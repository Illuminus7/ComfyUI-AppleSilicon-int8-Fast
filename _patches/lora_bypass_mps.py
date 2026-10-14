"""Patch: accumulate a bypass-mode LoRA into the layer output in place on MPS.

ComfyUI's bypass forward computes base(x) + up(down(x)) * scale as a full-size up GEMM,
a scale pass and an add, i.e. three extra writes of the layer output. With MiniMax H3's
turbo LoRA at 16k tokens that is ~11% of every step. Here the up projection is one
addmm into the fresh base output (beta=1, alpha=scale). Plain linear LoRAs only;
conv/Tucker/DoRA adapters and custom g() keep the stock forward.
DEFAULT ON on MPS; ASFP8_LORA_BYPASS=off disables.
"""

import sys

import torch

TAG = "[AppleSilicon-int8-Fast/lora_bypass]"

_installed = False


def install():
    global _installed
    if _installed or sys.platform != "darwin":
        return
    from . import _caps
    if not _caps.resolve("ASFP8_LORA_BYPASS", default_on=True, cap=_caps.is_mps):
        return
    try:
        from comfy.weight_adapter.base import WeightAdapterBase
        from comfy.weight_adapter.bypass import BypassForwardHook
        from comfy.weight_adapter.lora import LoRAAdapter
    except Exception:
        return
    orig = BypassForwardHook._bypass_forward

    def _bypass_forward(self, x, *args, **kwargs):
        a = self.adapter
        w = getattr(a, "weights", None)
        if not (type(a) is LoRAAdapter and x.device.type == "mps" and not getattr(a, "is_conv", False)
                and type(a).g is WeightAdapterBase.g and isinstance(w, (list, tuple)) and len(w) >= 4
                and w[3] is None and all(v is None for v in w[4:6])):
            return orig(self, x, *args, **kwargs)
        up, down, alpha = w[0], w[1], w[2]
        scale = (alpha / down.shape[0] if alpha is not None else 1.0) * getattr(a, "multiplier", 1.0)
        out = self.original_forward(x, *args, **kwargs)
        if out.dtype != x.dtype or not out.is_contiguous():
            return out + a.h(x, out)
        hidden = torch.nn.functional.linear(x, down.to(x.dtype)).reshape(-1, down.shape[0])
        out.view(-1, out.shape[-1]).addmm_(hidden, up.to(x.dtype).t(), alpha=float(scale))
        return out

    BypassForwardHook._bypass_forward = _bypass_forward
    _installed = True
    print(f"{TAG} bypass-mode LoRA accumulated in place on MPS (one addmm per layer).")
