"""Patch: single-pass channel RMS_norm for the Wan VAE on MPS.

The Wan 2.1/2.2 VAE normalizes over the channel dim of [B, C, T, H, W] activations with
F.normalize(x, dim=1) * sqrt(C) * gamma (+ bias). On MPS that strided reduction plus the
broadcast multiply/add take ~45% of a 1536x1024 decode once its convs are fast. One
shader thread per 4 spatial positions reduces over C and writes the result in one pass.
DEFAULT ON where compile_shader works; ASFP8_WAN_VAE_NORM=off disables.
"""

import sys

import torch

TAG = "[AppleSilicon-int8-Fast/wan_vae_norm]"

_installed = False
_lib = None
_orig_forward = None

_MSL = r"""
#include <metal_stdlib>
using namespace metal;

// x, out: [B, C, S] contiguous; out = x / max(||x[:, s]||, 1e-12) * scale * gamma[c] (+ bias[c])
template <typename T>
inline void channel_norm(const device T* x, device T* out, const device float* gamma, const device float* bias,
                         uint C, uint S, float scale, uint has_bias, uint i) {
  const uint b = i / (S / 4), s = (i % (S / 4)) * 4;
  const ulong base = ulong(b) * C * S + s;
  float4 ss = 0.0f;
  for (uint c = 0; c < C; ++c) {
    const float4 v = float4(*(const device vec<T, 4>*)(x + base + ulong(c) * S));
    ss += v * v;
  }
  const float4 r = scale / max(sqrt(ss), 1e-12f);
  for (uint c = 0; c < C; ++c) {
    float4 v = float4(*(const device vec<T, 4>*)(x + base + ulong(c) * S)) * r * gamma[c];
    if (has_bias) v += bias[c];
    *(device vec<T, 4>*)(out + base + ulong(c) * S) = vec<T, 4>(v);
  }
}

#define NORM(NAME, T) \
kernel void NAME(const device T* x [[buffer(0)]], device T* out [[buffer(1)]], const device float* gamma [[buffer(2)]], \
                 const device float* bias [[buffer(3)]], constant uint& C [[buffer(4)]], constant uint& S [[buffer(5)]], \
                 constant float& scale [[buffer(6)]], constant uint& has_bias [[buffer(7)]], \
                 uint i [[thread_position_in_grid]]) { \
  channel_norm<T>(x, out, gamma, bias, C, S, scale, has_bias, i); \
}
NORM(channel_norm_bf16, bfloat)
NORM(channel_norm_f16, half)
NORM(channel_norm_f32, float)
"""

_SUFFIX = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}


def _channel_norm(x, gamma, bias, scale):
    B, C = x.shape[0], x.shape[1]
    S = x[0, 0].numel()
    x = x.contiguous()
    out = torch.empty_like(x)
    g = gamma.to(device=x.device, dtype=torch.float32).reshape(-1).contiguous()
    b = g if bias is None else bias.to(device=x.device, dtype=torch.float32).reshape(-1).contiguous()
    getattr(_lib, "channel_norm_" + _SUFFIX[x.dtype])(x, out, g, b, C, S, float(scale), int(bias is not None),
                                                     threads=(B * S // 4, 1, 1))
    return out


def _verify():
    global _lib
    try:
        _lib = torch.mps.compile_shader(_MSL)
    except Exception as e:
        print(f"{TAG} shader did not compile; using the stock RMS_norm: {e!r}")
        return False
    from comfy.ldm.wan.vae import RMS_norm
    g = torch.Generator().manual_seed(0)
    norm = RMS_norm(24, images=False, bias=True)
    norm.gamma.data = torch.randn(norm.gamma.shape, generator=g)
    norm.bias.data = torch.randn(norm.bias.shape, generator=g)
    x = torch.randn(2, 24, 1, 6, 10, generator=g)
    ref = _orig_forward(norm, x)
    out = _channel_norm(x.to("mps", torch.bfloat16), norm.gamma, norm.bias, norm.scale).cpu().float()
    err = ((out - ref).norm() / ref.norm()).item()
    if not err < 0.01:
        print(f"{TAG} self-check mismatch (rel err {err:.3g}); using the stock RMS_norm.")
        return False
    return True


def _forward(self, x):
    from . import _caps
    if (self.channel_first and x.device.type == "mps" and x.dtype in _SUFFIX and x.dim() >= 3
            and x[0, 0].numel() % 4 == 0 and _caps.kernel_ready("wan_vae_norm", _verify)):
        return _channel_norm(x, self.gamma, self.bias, self.scale)
    return _orig_forward(self, x)


def install():
    global _installed, _orig_forward
    if _installed or sys.platform != "darwin":
        return
    from . import _caps
    if not _caps.resolve("ASFP8_WAN_VAE_NORM", default_on=True, cap=_caps.has_compile_shader):
        return
    try:
        from comfy.ldm.wan.vae import RMS_norm
    except Exception:
        return
    _orig_forward = RMS_norm.forward
    RMS_norm.forward = _forward
    _installed = True
    print(f"{TAG} Wan VAE channel RMS_norm fused on MPS (self-check on first use).")
