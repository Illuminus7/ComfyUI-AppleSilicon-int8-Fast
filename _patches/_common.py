"""Shared helpers: FP8 decode, and rebinding helpers that other modules import by name.

MPS has no FP8 casts, so FP8 decodes via a 256-entry uint8 LUT built on CPU (where the
cast works) and gathered on-device, which is bit-exact with a real FP8->float cast.
"""

import sys

import torch

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

_lut_cache = {}


def fp8_to_float_lut(dtype, device, out_dtype=torch.float32):
    """Return a cached 256-entry LUT (`out_dtype`) for FP8 `dtype`, on `device`."""
    key = (dtype, out_dtype, device.type, getattr(device, "index", None))
    lut = _lut_cache.get(key)
    if lut is None:
        lut = torch.arange(256, dtype=torch.uint8).view(dtype).to(out_dtype).to(device)
        _lut_cache[key] = lut
    return lut


def decode_fp8(t, out_dtype=torch.float32):
    """Decode an FP8 tensor to `out_dtype` on its own device (MPS-safe)."""
    device = t.device
    lut = fp8_to_float_lut(t.dtype, device, out_dtype)
    if t.is_contiguous():
        idx = t.view(torch.uint8).to(torch.long)
    else:
        # MPS can't .contiguous() an FP8 tensor, so do it on CPU
        idx = t.cpu().contiguous().view(torch.uint8).to(torch.long).to(device)
    return lut[idx]


_rebinds = []
_first_load = []


def on_first_model_load(fn):
    """Run fn once more at the first model load, when every custom node (sorted after this
    one) has been imported."""
    if not _first_load:
        import comfy.model_management as mm
        orig = mm.load_models_gpu

        def load_models_gpu(*args, **kwargs):
            mm.load_models_gpu = orig
            for f in _first_load:
                f()
            return orig(*args, **kwargs)

        mm.load_models_gpu = load_models_gpu
    _first_load.append(fn)


def _sweep():
    for mod in list(sys.modules.values()):
        try:
            items = list(vars(mod).items())
        except TypeError:
            continue
        for name, val in items:
            for old, new in _rebinds:
                if val is old:
                    setattr(mod, name, new)


def rebind_everywhere(old, new):
    """Point every module-level reference to `old` at `new`: model code and custom nodes
    bind helpers by name at import, and custom nodes sorted after this one import later,
    so sweep now and once more at the first model load."""
    if not _rebinds:
        on_first_model_load(_sweep)
    _rebinds.append((old, new))
    _sweep()
