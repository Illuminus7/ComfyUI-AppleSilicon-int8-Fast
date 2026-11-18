"""Fix: a text encoder stays on MPS through video sampling and takes memory the sampler needs.

ComfyUI keeps a text encoder loaded beside the diffusion model while both fit, so MiniMax
H3's 15 GB encoder stays on the GPU through sampling although its work is done. A video's
activations then push a 48 GB Mac into swap.

Before a diffusion model loads on MPS, if what MPS holds, the incoming model and ComfyUI's
own estimate of the sampling memory would not fit under the MPS allocator's limit, unload the
text encoder first, and the VAE and other helpers with it. The helpers matter: loaded after
the encoder, their weights sit inside the encoder's 1 GiB MPS heaps, and a heap only goes back
to macOS once all of it is free; leaving the VAE pinned 12 of the encoder's 15 GB. With
mmap_load's return-by-reference the weights go back to the file mapping they came from, so
the memory really is released rather than copied into RAM.

After that has happened, a new prompt's encoder load would land on top of the diffusion
model still loaded from the last run: it swaps, and the encoder's weights end up in the gaps
between the model's, where they cannot be returned later. So while the encoder is being
offloaded, its next load unloads the diffusion model first when the two would not fit in 60%
of RAM (mmap_load's rule); the model reloads from the file in seconds.

Images are not affected: their sampling estimate is small, so the encoder stays loaded. The
model objects stay cached either way, so this works the same with --cache-classic,
--cache-lru, --cache-none and the default cache.

  APPLESILICON_TE_OFFLOAD=off      disable
  APPLESILICON_TE_OFFLOAD=on       offload before every diffusion model load
"""

import os
import sys

import torch
from . import _env

TAG = "[AppleSilicon-int8-Fast/te_offload]"

_installed = False
_mode = "auto"
_limit = 0
_ram = 0
_TIGHT_FRACTION = 0.6    # of physical RAM, as in mmap_load
_offloading = False      # the last diffusion load offloaded the encoder


def _on_mps(lm):
    return getattr(lm.device, "type", None) == "mps"


def _is_diffusion(patcher, model_base):
    return (not getattr(patcher, "is_clip", False)
            and isinstance(getattr(patcher, "model", None), model_base.BaseModel))


def _unload(mm, targets):
    """Fully unload these LoadedModels and return the bytes they held on MPS."""
    freed = 0
    for lm in targets:
        size = lm.model.loaded_size()
        if lm.model_unload():
            freed += size
            for i, other in enumerate(mm.current_loaded_models):
                if other is lm:
                    mm.current_loaded_models.pop(i)
                    break
    if freed:
        torch.mps.empty_cache()
    return freed


def _loaded(mm, keep, want):
    out = []
    for lm in list(mm.current_loaded_models):
        p = lm.model
        if p is None or id(p) in keep or not _on_mps(lm) or lm.is_dead():
            continue
        if want(p) and p.loaded_size() > 0:
            out.append(lm)
    return out


def _incoming(patchers):
    return sum(max(0, p.model_size() - p.loaded_size()) for p in patchers)


def _before_load(mm, requested, memory_required, model_base):
    """Make room before ComfyUI loads `requested`. Returns bytes unloaded."""
    global _offloading
    keep = {id(p) for p in requested}
    on_mps = [p for p in requested if getattr(p.load_device, "type", None) == "mps"]

    if any(_is_diffusion(p, model_base) for p in on_mps):
        need = torch.mps.driver_allocated_memory() + _incoming(on_mps) + memory_required
        _offloading = _mode == "on" or need > _limit
        if not _offloading:
            return 0
        targets = _loaded(mm, keep, lambda p: not _is_diffusion(p, model_base))
        if not any(getattr(lm.model, "is_clip", False) for lm in targets):
            return 0
        freed = _unload(mm, targets)
        if freed:
            print(f"{TAG} unloaded the text encoder and helpers ({freed / 2**30:.1f} GiB) "
                  f"before sampling.", flush=True)
        return freed

    if _offloading and any(getattr(p, "is_clip", False) for p in on_mps):
        tight = _TIGHT_FRACTION * _ram
        if _mode != "on" and torch.mps.driver_allocated_memory() + _incoming(on_mps) <= tight:
            return 0
        targets = _loaded(mm, keep, lambda p: _is_diffusion(p, model_base))
        freed = _unload(mm, targets)
        if freed:
            print(f"{TAG} unloaded the diffusion model ({freed / 2**30:.1f} GiB) so the text "
                  f"encoder loads without swapping; it reloads for sampling.", flush=True)
        return freed
    return 0


def install():
    global _installed, _mode, _limit, _ram
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    setting = _env.get("APPLESILICON_TE_OFFLOAD", "auto").lower()
    if setting in ("off", "0", "false"):
        print(f"{TAG} disabled by APPLESILICON_TE_OFFLOAD.")
        return
    _mode = "on" if setting in ("on", "1", "true") else "auto"
    from . import mmap_load_mps, mps_free_memory
    if not mmap_load_mps._installed:
        # Without return-by-reference, unloading only copies the weights into RAM.
        return
    try:
        import comfy.model_management as mm
        import comfy.model_base as model_base
        _limit = mps_free_memory._allocator_limit() or torch.mps.recommended_max_memory()
        _ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return

    orig = mm.load_models_gpu

    def load_models_gpu(models, memory_required=0, *args, **kwargs):
        try:
            _before_load(mm, list(models), memory_required or 0, model_base)
        except Exception:
            pass
        return orig(models, memory_required, *args, **kwargs)

    mm.load_models_gpu = load_models_gpu
    _installed = True
    print(f"{TAG} text encoders are unloaded before sampling when it needs the memory "
          f"(limit {_limit / 2**30:.1f} GiB).")
