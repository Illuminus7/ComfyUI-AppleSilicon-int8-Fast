"""Fix: switching workflows leaves the previous model's weights on MPS, and the next one swaps.

When a workflow switch drops a model from ComfyUI's cache, the model object goes, but its
weights can stay alive in reference cycles until Python's cyclic garbage collector happens
to run. ComfyUI only forces a collection when the model object itself survives, so the
weights of, say, a 20 GB video model stay on the GPU while the next workflow loads another
20 GB model, and the Mac swaps for minutes.

Before ComfyUI frees memory for a load, compare what MPS holds with what its loaded models
account for. When more than a couple of GiB is unaccounted for, run the collector once.

  APPLESILICON_MPS_ORPHAN_GC=off      disable
"""

import gc
import sys

import torch
from . import _env

TAG = "[AppleSilicon-int8-Fast/mps_orphan_gc]"

_THRESHOLD = 2 << 30   # bytes held beyond the loaded models before a collection is worth it

_installed = False
_baseline = 0          # unaccounted bytes a collection could not free; not collected for again


def _unaccounted(mm):
    """Bytes MPS tensors hold beyond what ComfyUI's loaded models account for."""
    loaded = 0
    for lm in list(mm.current_loaded_models):
        try:
            if lm.model is not None and getattr(lm.device, "type", None) == "mps":
                loaded += lm.model.loaded_size()
        except Exception:
            pass
    return torch.mps.current_allocated_memory() - loaded


def _collect_if_orphaned(mm):
    """Collect cyclic garbage when MPS holds much more than the loaded models."""
    global _baseline
    extra = _unaccounted(mm)
    if extra < _baseline + _THRESHOLD:
        if extra < _baseline:
            _baseline = max(0, extra)      # memory came back by itself; lower the bar again
        return 0
    gc.collect()
    torch.mps.empty_cache()
    after = _unaccounted(mm)
    _baseline = max(0, after)
    freed = extra - after
    if freed >= 1 << 30:
        print(f"{TAG} freed {freed / 2**30:.1f} GiB a dropped model left on the GPU.", flush=True)
    return freed


def install():
    global _installed
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    if _env.get("APPLESILICON_MPS_ORPHAN_GC", "auto").lower() in ("off", "0", "false"):
        print(f"{TAG} disabled by APPLESILICON_MPS_ORPHAN_GC.")
        return
    try:
        import comfy.model_management as mm
    except Exception:
        return

    orig = mm.free_memory

    def free_memory(memory_required, device, *args, **kwargs):
        if getattr(device, "type", None) == "mps":
            try:
                _collect_if_orphaned(mm)
            except Exception:
                pass
        return orig(memory_required, device, *args, **kwargs)

    mm.free_memory = free_memory
    _installed = True
    print(f"{TAG} weights a dropped model leaves on MPS are collected before the next load.")
