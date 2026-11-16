"""Fix: ComfyUI keeps loading past the MPS allocator's limit, then sampling runs out of memory.

On MPS ComfyUI takes free memory to be the system's available RAM. The MPS allocator has
its own ceiling (PYTORCH_MPS_HIGH_WATERMARK_RATIO x the recommended working set, which
prestartup_script.py sets to 1.0), and ComfyUI does not know it. With plenty of RAM free it
leaves the text encoder beside the diffusion model, and the first big activation fails with
"MPS backend out of memory".

Report the smaller of the two: available RAM, or the room left under the allocator's limit.
ComfyUI then unloads enough before it samples.

  APPLESILICON_MPS_FREE_MEMORY=off      disable
"""

import os
import sys

import torch

TAG = "[AppleSilicon-int8-Fast/mps_free_memory]"

_installed = False
_limit = None


def _allocator_limit():
    """The MPS allocator's hard limit in bytes, or None when it has none."""
    try:
        ratio = float(os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.7"))
    except ValueError:
        return None
    if ratio <= 0:
        return None          # 0 disables the limit
    return int(ratio * torch.mps.recommended_max_memory())


def _room():
    """Bytes the allocator can still take from the system before it hits its limit.

    Measured against what it holds, not what tensors use: the limit applies to the former,
    which runs 1-2 GiB higher from block rounding and cached blocks of the wrong size.
    """
    return max(0, _limit - torch.mps.driver_allocated_memory())


def install():
    global _installed, _limit
    if _installed:
        return
    if sys.platform != "darwin":
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return
    if os.environ.get("APPLESILICON_MPS_FREE_MEMORY", "auto").lower() in ("off", "0", "false"):
        print(f"{TAG} disabled by APPLESILICON_MPS_FREE_MEMORY.")
        return
    try:
        import comfy.model_management as mm
    except Exception:
        return
    _limit = _allocator_limit()
    if _limit is None:
        return

    orig = mm.get_free_memory

    def get_free_memory(dev=None, torch_free_too=False):
        out = orig(dev, torch_free_too)
        if getattr(dev if dev is not None else mm.get_torch_device(), "type", None) != "mps":
            return out
        room = _room()
        if torch_free_too:
            return (min(out[0], room), min(out[1], room))
        return min(out, room)

    mm.get_free_memory = get_free_memory
    _installed = True
    print(f"{TAG} free memory on MPS is capped at the allocator's limit "
          f"({_limit / 2**30:.1f} GiB).")
