"""Fix: ComfyUI's /free takes minutes on a Mac that is short of memory.

/free with free_memory (what apps send before loading something of their own) first unloads
every model, moving its weights back to the CPU, then drops every cached node output, which
frees those models. Under memory pressure macOS has paged out part of the idle GPU memory, so
unloading, which reads every weight, pages it all back in: H3 image's 35 GB took 80-110 s on
a 48 GB Mac, while the app loading its own model waited or swapped beside it.

When a /free that drops the cache is pending, its unload stops tracking the loaded models
instead of moving their weights; the cache reset and garbage collection that follow free them
without reading anything. A model that something outside the cache still holds is unloaded
the normal way afterwards, so no GPU memory is left untracked.

  APPLESILICON_FAST_FREE=off      disable
"""

import sys
import weakref

import torch
from . import _env

TAG = "[AppleSilicon-int8-Fast/fast_free]"

_installed = False
_pending = False     # a /free that drops the cache is waiting for the prompt worker
_dropped = []        # weak refs to the models its unload stopped tracking


def _drop(mm):
    """Stop tracking every loaded model without moving its weights. False when a model is on
    another device than MPS, so the normal unload runs."""
    loaded = list(mm.current_loaded_models)
    if not loaded or any(getattr(lm.device, "type", None) != "mps" for lm in loaded):
        return False
    refs = []
    for lm in loaded:
        if lm.model is not None:
            refs.append(weakref.ref(lm.model))
        if lm.model_finalizer is not None:
            lm.model_finalizer.detach()
            lm.model_finalizer = None
    mm.current_loaded_models.clear()
    _dropped[:] = refs
    return True


def _finish():
    """After the cache reset and collection: unload, the normal way, any dropped model
    something still holds. Returns how many there were."""
    alive = [p for p in (r() for r in _dropped) if p is not None]
    _dropped.clear()
    for patcher in alive:
        try:
            patcher.detach(unpatch_all=True)
        except Exception as e:
            print(f"{TAG} could not unload a model that outlived /free: {e!r}", flush=True)
    return len(alive)


def install():
    global _installed
    if _installed or sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    if _env.get("APPLESILICON_FAST_FREE", "auto").lower() in ("off", "0", "false"):
        print(f"{TAG} disabled by APPLESILICON_FAST_FREE.")
        return
    try:
        import comfy.model_management as mm
        import execution
        queue_cls = execution.PromptQueue
    except Exception:
        return

    orig_set_flag = queue_cls.set_flag

    def set_flag(self, name, data):
        global _pending
        if name == "free_memory" and data:
            _pending = True
        return orig_set_flag(self, name, data)

    orig_unload_all = mm.unload_all_models

    def unload_all_models(*args, **kwargs):
        global _pending
        if _pending:
            _pending = False
            try:
                if _drop(mm):
                    return None
            except Exception as e:
                print(f"{TAG} fast free failed ({e!r}); unloading normally.", flush=True)
        return orig_unload_all(*args, **kwargs)

    orig_soft_empty_cache = mm.soft_empty_cache

    def soft_empty_cache(*args, **kwargs):
        if _dropped:
            try:
                _finish()
            except Exception:
                _dropped.clear()
        return orig_soft_empty_cache(*args, **kwargs)

    queue_cls.set_flag = set_flag
    mm.unload_all_models = unload_all_models
    mm.soft_empty_cache = soft_empty_cache
    _installed = True
    print(f"{TAG} /free drops cached models without reading their weights back.")
