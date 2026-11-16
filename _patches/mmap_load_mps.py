"""Fix: moving a memory-mapped model to MPS doubles its memory and pushes the Mac into swap.

safetensors hands ComfyUI tensors backed by a private file mapping. torch's CPU->MPS copy
wraps the source pages in a no-copy Metal buffer; wiring a private mapping for that makes
the kernel duplicate every page into anonymous memory, and the duplicates live until the
whole mapping goes (after the last weight has moved). A 20 GB model therefore needs 40 GB
plus its file cache mid-load, so macOS compresses and swaps the weights it just received.

When a plain CPU tensor that sits in a clean file mapping moves to MPS, read its bytes from
the file instead (uncached, into a reused scratch buffer) and move those. Same bytes, no
duplicate, no file cache growth. Anything unusual falls back to torch's own path.

Unloading has the mirror problem: ComfyUI moves weights "to the CPU" to free memory, which on
unified memory only copies them into ordinary RAM. A weight that arrived by the path above
and still holds the file's values goes back as the mapped tensor it came from instead: no
copy, and the memory really is released.

This only happens where memory is tight: when the model's file, counted twice for torch's
duplicate, would not fit beside what MPS already holds in 60% of the machine's RAM. With
room to spare torch's path is left alone, since it can reload from the file cache, which
matters on a slow disk.

  APPLESILICON_MMAP_LOAD=off      disable
  APPLESILICON_MMAP_LOAD=on       use it for every mapped weight, whatever the memory
"""

import ctypes
import fcntl
import mmap
import os
import sys
import threading
import time

import torch

TAG = "[AppleSilicon-int8-Fast/mmap_load]"

_installed = False
_orig_to = None

_PAGE = 16384
_SCRATCH_MAX = 256 << 20          # larger tensors move in chunks of this size
_PROC_PIDREGIONPATHINFO2 = 22    # vnode-backed regions only, without the per-page statistics
# mincore() bits saying a page may no longer equal the file: modified here or elsewhere,
# copied on write, or anonymous
_MINCORE_CHANGED = 0x04 | 0x10 | 0x40 | 0x80
_MAXPATHLEN = 1024

_TIGHT_FRACTION = 0.6             # of physical RAM; see _tight()
_DECISION_TTL = 5.0               # seconds a per-file decision holds between weights
_SAMPLES = 64                     # values compared before a weight is taken to be unchanged

stats = {"fast_bytes": 0, "fast_tensors": 0, "fallback_tensors": 0, "returned_tensors": 0}

# MPS storage -> (weak ref to it, the mapped tensor it was read from, sampled bytes).
# Holding the mapped tensor keeps the mapping alive; untouched, it costs no memory.
_sources = {}


class _RegionInfo(ctypes.Structure):
    _fields_ = [
        ("pri_protection", ctypes.c_uint32),
        ("pri_max_protection", ctypes.c_uint32),
        ("pri_inheritance", ctypes.c_uint32),
        ("pri_flags", ctypes.c_uint32),
        ("pri_offset", ctypes.c_uint64),
        ("pri_behavior", ctypes.c_uint32),
        ("pri_user_wired_count", ctypes.c_uint32),
        ("pri_user_tag", ctypes.c_uint32),
        ("pri_pages_resident", ctypes.c_uint32),
        ("pri_pages_shared_now_private", ctypes.c_uint32),
        ("pri_pages_swapped_out", ctypes.c_uint32),
        ("pri_pages_dirtied", ctypes.c_uint32),
        ("pri_ref_count", ctypes.c_uint32),
        ("pri_shadow_depth", ctypes.c_uint32),
        ("pri_share_mode", ctypes.c_uint32),
        ("pri_private_pages_resident", ctypes.c_uint32),
        ("pri_shared_pages_resident", ctypes.c_uint32),
        ("pri_obj_id", ctypes.c_uint32),
        ("pri_depth", ctypes.c_uint32),
        ("pri_address", ctypes.c_uint64),
        ("pri_size", ctypes.c_uint64),
    ]


class _VinfoStat(ctypes.Structure):
    _fields_ = [
        ("vst_dev", ctypes.c_uint32),
        ("vst_mode", ctypes.c_uint16),
        ("vst_nlink", ctypes.c_uint16),
        ("vst_ino", ctypes.c_uint64),
        ("vst_uid", ctypes.c_uint32),
        ("vst_gid", ctypes.c_uint32),
        ("vst_atime", ctypes.c_int64),
        ("vst_atimensec", ctypes.c_int64),
        ("vst_mtime", ctypes.c_int64),
        ("vst_mtimensec", ctypes.c_int64),
        ("vst_ctime", ctypes.c_int64),
        ("vst_ctimensec", ctypes.c_int64),
        ("vst_birthtime", ctypes.c_int64),
        ("vst_birthtimensec", ctypes.c_int64),
        ("vst_size", ctypes.c_int64),
        ("vst_blocks", ctypes.c_int64),
        ("vst_blksize", ctypes.c_int32),
        ("vst_flags", ctypes.c_uint32),
        ("vst_gen", ctypes.c_uint32),
        ("vst_rdev", ctypes.c_uint32),
        ("vst_qspare", ctypes.c_int64 * 2),
    ]


class _RegionWithPath(ctypes.Structure):
    _fields_ = [
        ("prp_prinfo", _RegionInfo),
        ("vi_stat", _VinfoStat),
        ("vi_type", ctypes.c_int),
        ("vi_pad", ctypes.c_int),
        ("vi_fsid", ctypes.c_int32 * 2),
        ("vip_path", ctypes.c_char * _MAXPATHLEN),
    ]


_libproc = None
_libc = None
_pid = None
_lock = threading.Lock()
_scratch = None          # anonymous, page-aligned, reused for every read
_scratch_view = None
_fds = {}                # (dev, ino) -> uncached read fd
_mode = "auto"           # "auto" or "on"
_ram = 0                 # physical memory in bytes
_decisions = {}          # (dev, ino) -> (tight?, time of the last weight from that file)


def _region(addr):
    """The VM region holding addr, or None."""
    info = _RegionWithPath()
    n = _libproc.proc_pidinfo(_pid, _PROC_PIDREGIONPATHINFO2, ctypes.c_uint64(addr),
                              ctypes.byref(info), ctypes.sizeof(info))
    if n != ctypes.sizeof(info):
        return None
    r = info.prp_prinfo
    if not (r.pri_address <= addr < r.pri_address + r.pri_size):
        return None          # addr is unmapped; the call answered with the next region
    return info


def _untouched(ptr, nbytes):
    """True when no page of [ptr, ptr+nbytes) was written or duplicated since it was mapped."""
    start = ptr - ptr % _PAGE
    length = -(-(ptr + nbytes - start) // _PAGE) * _PAGE
    vec = ctypes.create_string_buffer(length // _PAGE)
    if _libc.mincore(start, length, vec) != 0:
        return False
    return not any(b & _MINCORE_CHANGED for b in set(vec.raw))


def _file_span(ptr, nbytes):
    """(path, (dev, ino), file offset of ptr, file size) when [ptr, ptr+nbytes) is an
    untouched mapping of one file, else None."""
    first = None
    addr = ptr
    end = ptr + nbytes
    while addr < end:
        info = _region(addr)
        if info is None or not info.vip_path:
            return None
        r = info.prp_prinfo
        key = (info.vi_stat.vst_dev, info.vi_stat.vst_ino)
        offset = r.pri_offset + (addr - r.pri_address)
        if first is None:
            first = (info.vip_path, key, offset, info.vi_stat.vst_size)
        elif key != first[1] or offset != first[2] + (addr - ptr):
            return None
        addr = r.pri_address + r.pri_size
    if not _untouched(ptr, nbytes):
        return None
    path, key, offset, size = first
    return os.fsdecode(path), key, offset, size


def _tight(key, file_size):
    """Would torch's own move of this file run the machine short of memory?

    Its peak is what MPS holds already plus the file twice (the copy and the duplicate). One
    answer per file and load: MPS fills up as the file's weights arrive, and a load must not
    change paths halfway.
    """
    if _mode == "on":
        return True
    now = time.monotonic()
    known = _decisions.get(key)
    if known is not None and now - known[1] < _DECISION_TTL:
        tight = known[0]
    else:
        tight = torch.mps.driver_allocated_memory() + 2 * file_size > _TIGHT_FRACTION * _ram
    _decisions[key] = (tight, now)
    return tight


def _fd_for(path, key):
    fd = _fds.get(key)
    if fd is None:
        fd = os.open(path, os.O_RDONLY)
        st = os.fstat(fd)
        if (st.st_dev & 0xFFFFFFFF, st.st_ino) != (key[0] & 0xFFFFFFFF, key[1]):
            os.close(fd)     # the path now names another file than the one mapped
            return None
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        _fds[key] = fd
    return fd


def _read(fd, offset, nbytes):
    """nbytes of the file at offset, as a view into the scratch buffer. Reads are page-aligned
    so that F_NOCACHE holds; an unaligned read quietly goes through the file cache."""
    global _scratch, _scratch_view
    start = offset - offset % _PAGE
    length = -(-(offset + nbytes - start) // _PAGE) * _PAGE
    if _scratch is None or len(_scratch) < length:
        _scratch_view = None
        _scratch = mmap.mmap(-1, max(length, 16 << 20))
        _scratch_view = memoryview(_scratch)
    got = os.preadv(fd, [_scratch_view[:length]], start)
    if got < offset - start + nbytes:
        raise OSError("short read")
    return _scratch_view[offset - start: offset - start + nbytes]


def _typed(view, dtype):
    t = torch.frombuffer(view, dtype=torch.uint8)
    return t if dtype == torch.uint8 else t.view(dtype)


def _move(self, device):
    """self's bytes on `device`, read from the file self is mapped from; None to fall back."""
    nbytes = self.numel() * self.element_size()
    span = _file_span(self.data_ptr(), nbytes)
    if span is None:
        return None
    path, key, offset, file_size = span
    if not _tight(key, file_size):
        return None
    with _lock:
        fd = _fd_for(path, key)
        if fd is None:
            return None
        if nbytes <= _SCRATCH_MAX:
            host = _typed(_read(fd, offset, nbytes), self.dtype)
            out = _orig_to(host.reshape(self.shape), device)
            torch.mps.synchronize()      # the copy reads the scratch buffer asynchronously
            _remember(out, self, host)
            return out
        out = torch.empty(self.shape, dtype=self.dtype, device=device)
        flat = out.view(-1)
        item = self.element_size()
        step = _SCRATCH_MAX // item
        for i in range(0, self.numel(), step):
            n = min(step, self.numel() - i)
            flat[i:i + n].copy_(_typed(_read(fd, offset + i * item, n * item), self.dtype))
            torch.mps.synchronize()
        return out


def _sample(flat):
    """A fixed, evenly spread handful of flat's elements."""
    return flat[::max(1, flat.numel() // _SAMPLES)][:_SAMPLES]


def _remember(out, source, host):
    """Note which mapped tensor `out` (on MPS) was read from, with a sample of its values."""
    from torch.multiprocessing.reductions import StorageWeakRef
    if len(_sources) % 512 == 511:
        for k in [k for k, v in _sources.items() if v[0].expired()]:
            del _sources[k]
    storage = out.untyped_storage()
    sample = bytes(_sample(host).contiguous().view(torch.uint8).numpy())
    _sources[storage._cdata] = (StorageWeakRef(storage), source.detach(), sample)


def _return(self):
    """The mapped tensor self was read from, when self still holds its values; else None."""
    storage = self.untyped_storage()
    entry = _sources.get(storage._cdata)
    if entry is None:
        return None
    ref, source, sample = entry
    del _sources[storage._cdata]         # handed out once: the caller may write to it
    if ref.expired() or source.dtype != self.dtype or source.numel() != self.numel():
        return None
    if self.storage_offset() != 0 or storage.nbytes() != self.numel() * self.element_size():
        return None
    now = _orig_to(_sample(self.reshape(-1)), "cpu").contiguous().view(torch.uint8)
    if bytes(now.numpy()) != sample:
        return None                      # written to since it was loaded
    if not _untouched(source.data_ptr(), source.numel() * source.element_size()):
        return None
    return source.reshape(self.shape)


def _wants_plain_return(self, args, kwargs):
    """True when this .to() only moves a plain MPS tensor to the CPU."""
    if not _sources or self.device.type != "mps":
        return False
    if type(self) not in (torch.Tensor, torch.nn.Parameter) or self.requires_grad:
        return False
    device = kwargs.get("device")
    dtype = kwargs.get("dtype")
    for a in args:
        if isinstance(a, torch.dtype):
            dtype = a
        elif isinstance(a, (torch.device, str)):
            device = a
        elif isinstance(a, torch.Tensor):
            return False
    if kwargs.get("copy") or any(k not in ("device", "dtype", "non_blocking", "copy") for k in kwargs):
        return False
    if device is None or (dtype is not None and dtype != self.dtype):
        return False
    try:
        return torch.device(device).type == "cpu" and self.is_contiguous()
    except (TypeError, ValueError, RuntimeError):
        return False


def _self_test():
    """Map this file and check the lookup names it, at the right offset."""
    try:
        mm = _libc.mmap
        mm.restype = ctypes.c_void_p
        mm.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                       ctypes.c_int, ctypes.c_longlong]
        _libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        fd = os.open(__file__, os.O_RDONLY)
        try:
            st = os.fstat(fd)
            addr = mm(None, st.st_size, mmap.PROT_READ, mmap.MAP_PRIVATE, fd, 0)
        finally:
            os.close(fd)
        if addr in (None, ctypes.c_void_p(-1).value):
            return False
        try:
            span = _file_span(addr + 5, min(64, st.st_size - 5))
        finally:
            _libc.munmap(addr, st.st_size)
        return (span is not None and span[1] == (st.st_dev & 0xFFFFFFFF, st.st_ino)
                and span[2] == 5 and span[3] == st.st_size)
    except Exception:
        return False


def _wants_plain_move(self, args, kwargs):
    """The target device when this .to() only moves a plain CPU tensor to MPS, else None."""
    if type(self) not in (torch.Tensor, torch.nn.Parameter):
        return None
    if self.device.type != "cpu" or self.layout != torch.strided or self.requires_grad:
        return None
    device = kwargs.get("device")
    dtype = kwargs.get("dtype")
    for a in args:
        if isinstance(a, torch.dtype):
            dtype = a
        elif isinstance(a, (torch.device, str)):
            device = a
        elif isinstance(a, torch.Tensor):
            return None
    if any(k not in ("device", "dtype", "non_blocking", "copy") for k in kwargs):
        return None
    if device is None or (dtype is not None and dtype != self.dtype):
        return None
    try:
        device = torch.device(device)
    except (TypeError, ValueError, RuntimeError):
        return None
    if device.type != "mps" or self.numel() == 0 or not self.is_contiguous():
        return None
    return device


def install():
    global _installed, _orig_to, _libproc, _libc, _pid, _mode, _ram
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    setting = os.environ.get("APPLESILICON_MMAP_LOAD", "auto").lower()
    if setting in ("off", "0", "false"):
        print(f"{TAG} disabled by APPLESILICON_MMAP_LOAD.")
        return
    _mode = "on" if setting in ("on", "1", "true") else "auto"
    try:
        _ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return
    try:
        _libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        _libproc.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                          ctypes.c_void_p, ctypes.c_int]
        _libproc.proc_pidinfo.restype = ctypes.c_int
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
        _libc.mincore.restype = ctypes.c_int
    except OSError:
        return
    _pid = os.getpid()
    if not _self_test():
        print(f"{TAG} region lookup does not work on this macOS; not installed.")
        return

    _orig_to = torch.Tensor.to

    def _patched_to(self, *args, **kwargs):
        device = _wants_plain_move(self, args, kwargs)
        if device is not None:
            try:
                out = _move(self, device)
            except Exception:
                out = None
            if out is not None:
                stats["fast_tensors"] += 1
                stats["fast_bytes"] += out.numel() * out.element_size()
                return out
            stats["fallback_tensors"] += 1
        elif _wants_plain_return(self, args, kwargs):
            try:
                out = _return(self)
            except Exception:
                out = None
            if out is not None:
                stats["returned_tensors"] += 1
                return out
        return _orig_to(self, *args, **kwargs)

    torch.Tensor.to = _patched_to
    _installed = True
    when = "always" if _mode == "on" else "when memory is tight"
    print(f"{TAG} memory-mapped weights move to MPS by uncached file reads, and back by "
          f"reference ({when}): no duplicate pages, no file cache growth, no copy on unload.")
