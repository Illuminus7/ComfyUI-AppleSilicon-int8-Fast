"""Patch: single-pass shaders for MiniMax H3's elementwise work on MPS.

comfy_kitchen has no MPS kernel for the fused per-head q/k RMSNorm + split-half RoPE H3
runs in place on its qkv buffer, so its eager fallback costs ~40 ms per block at 20k
tokens; the per-segment adaLN modulate is two broadcast passes; and the "swiglu" input
act of the MLP down-projection is chunk + silu + mul. Each becomes one shader pass with
fp32 math. When fc2 runs the int8 kernel the swiglu shader also quantizes its output for it
(~2x faster than swiglu + the separate pass at 20k tokens), so the MLP forward is replaced
to hand that over. Masked (per-token) modulation rows and other shapes run the original code.
DEFAULT ON where compile_shader works; ASFP8_MINIMAX_H3_FUSED=off disables.
"""

import sys

import torch

from .int8_linear_kernel_mps import ROWQ_MSL, quant_target, shared_activation

TAG = "[AppleSilicon-int8-Fast/minimax_h3]"

_installed = False
_lib = None
_orig = {}

_MSL = r"""
#include <metal_stdlib>
using namespace metal;
""" + ROWQ_MSL + r"""

// In place on one head row of D = 32 * E elements (one simdgroup, E per lane): RMSNorm with
// weight w, then split-half RoPE over the first rot dims with the [S, rot/2, 2, 2] table f:
// y[i] = f[i][0][0]*x[i] + f[i][0][1]*x[i+h],  y[i+h] = f[i][1][0]*x[i] + f[i][1][1]*x[i+h].
template <typename T, int E>
inline void rms_rope_row(device T* x, const device float* w, const device T* f, uint half_, float eps, uint lane) {
  vec<float, E> v = vec<float, E>(*(device vec<T, E>*)(x + lane * E));
  float ss = 0.0f;
  for (short c = 0; c < E; ++c) ss += v[c] * v[c];
  v *= rsqrt(simd_sum(ss) / float(32 * E) + eps) * *(const device vec<float, E>*)(w + lane * E);
  const uint e = lane * E, hl = half_ / E;
  const bool lo = e < half_, hi = !lo && e < 2 * half_;
  const vec<float, E> p = simd_shuffle(v, ushort(lo ? lane + hl : (hi ? lane - hl : lane)));
  if (lo || hi) {
    const uint i0 = lo ? e : e - half_;
    for (short c = 0; c < E; ++c) {
      const device T* fr = f + (i0 + c) * 4;
      v[c] = lo ? float(fr[0]) * v[c] + float(fr[1]) * p[c] : float(fr[2]) * p[c] + float(fr[3]) * v[c];
    }
  }
  *(device vec<T, E>*)(x + lane * E) = vec<T, E>(v);
}

// h[r, c] = h[r, c] * (1 + scale[c]) + shift[c] over one contiguous segment, in place
template <typename T>
inline void mod_scale_shift(device T* h, const device float* scale, const device float* shift, uint D, uint i) {
  const uint e = i * 4, c = e % D;
  const float4 v = float4(*(device vec<T, 4>*)(h + e));
  *(device vec<T, 4>*)(h + e) = vec<T, 4>(v * (1.0f + *(const device float4*)(scale + c)) + *(const device float4*)(shift + c));
}

#define KERNELS(S, T) \
/* q/k: [B, S, H, D] with strides P = {sb, ss, sh}; table batch stride P[3] (0 when shared) */ \
kernel void rms_rope_##S(device T* q [[buffer(0)]], device T* k [[buffer(1)]], const device T* f [[buffer(2)]], \
                         const device float* qw [[buffer(3)]], const device float* kw [[buffer(4)]], constant uint& S_ [[buffer(5)]], \
                         constant uint& H [[buffer(6)]], constant long* P [[buffer(7)]], constant uint& half_ [[buffer(8)]], \
                         constant float& eps [[buffer(9)]], constant uint& rows [[buffer(10)]], constant uint& D [[buffer(11)]], \
                         uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
  const uint g = t / 32, r = g % rows, b = r / (S_ * H), s = (r / H) % S_, h = r % H; \
  const bool is_k = g >= rows; \
  device T* x = (is_k ? k : q) + b * P[0] + s * P[1] + h * P[2]; \
  const device T* fs = f + b * P[3] + ulong(s) * half_ * 4; \
  if (D == 128) rms_rope_row<T, 4>(x, is_k ? kw : qw, fs, half_, eps, lane); \
  else rms_rope_row<T, 2>(x, is_k ? kw : qw, fs, half_, eps, lane); \
} \
kernel void mod_scale_shift_##S(device T* h [[buffer(0)]], const device float* scale [[buffer(1)]], \
                                const device float* shift [[buffer(2)]], constant uint& D [[buffer(3)]], \
                                uint i [[thread_position_in_grid]]) { \
  mod_scale_shift<T>(h, scale, shift, D, i); \
} \
/* out[r, c] = silu(x[r, c]) * x[r, F + c] */ \
kernel void swiglu_##S(const device T* x [[buffer(0)]], device T* out [[buffer(1)]], constant uint& F [[buffer(2)]], \
                       uint i [[thread_position_in_grid]]) { \
  const ulong e = ulong(i) * 4, r = e / F, c = e % F; \
  const float4 g = float4(*(const device vec<T, 4>*)(x + r * 2 * F + c)); \
  *(device vec<T, 4>*)(out + e) = vec<T, 4>(g / (1.0f + exp(-g)) * float4(*(const device vec<T, 4>*)(x + r * 2 * F + F + c))); \
} \
/* swiglu one row per threadgroup, then the int8 quantization of it (rowq) */ \
kernel void swiglu_q_##S(const device T* x [[buffer(0)]], device T* out [[buffer(1)]], device char* q [[buffer(2)]], \
                         device float* qs [[buffer(3)]], constant uint& F [[buffer(4)]], constant uint& rotate [[buffer(5)]], \
                         uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                         uint nt [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], \
                         uint lane [[thread_index_in_simdgroup]]) { \
  threadgroup float red[32]; \
  const ulong r = row; \
  for (uint c = tid * 4; c < F; c += nt * 4) { \
    const float4 g = float4(*(const device vec<T, 4>*)(x + r * 2 * F + c)); \
    *(device vec<T, 4>*)(out + r * F + c) = vec<T, 4>(g / (1.0f + exp(-g)) * float4(*(const device vec<T, 4>*)(x + r * 2 * F + F + c))); \
  } \
  rowq<T>(out + r * F, q + r * F, qs + row, F, rotate, sg, nt / 32, lane, red); \
}
KERNELS(bf16, bfloat)
KERNELS(f16, half)
"""

_SUFFIX = {torch.bfloat16: "bf16", torch.float16: "f16"}


def _fn(name, dtype):
    return getattr(_lib, f"{name}_{_SUFFIX[dtype]}")


def _ready():
    from . import _caps
    return _caps.kernel_ready("minimax_h3", _verify)


def _rope_eligible(q, k, freqs_cis, q_scale, k_scale, rot_dim):
    if not (q.device.type == "mps" and q.dtype in _SUFFIX and k.dtype == q.dtype and freqs_cis.dtype == q.dtype):
        return False
    D = q.shape[-1]
    half = rot_dim // 2
    E = D // 32
    return (q.dim() == 4 and D in (64, 128) and k.shape == q.shape and q.stride() == k.stride() and q.stride(-1) == 1
            and all(st % E == 0 for st in q.stride()[:3]) and half % E == 0 and 0 < rot_dim <= D
            and freqs_cis.is_contiguous() and freqs_cis.numel() in (q.shape[1] * half * 4, q.numel() // D * half * 4 // q.shape[2])
            and freqs_cis.shape[-3] == half
            and q_scale.numel() == D and k_scale.numel() == D and q.numel() > 0)


def _rms_rope_split_half_(q, k, freqs_cis, q_scale, k_scale=None, epsilon=1e-6, rot_dim=0):
    """comfy_kitchen.rms_rope_split_half_ on MPS: q/k [B, S, H, D] strided views (e.g. of a qkv buffer)."""
    k_scale = q_scale if k_scale is None else k_scale
    rot = rot_dim or q.shape[-1]
    if not (_rope_eligible(q, k, freqs_cis, q_scale, k_scale, rot) and _ready()):
        return _orig["rms_rope_split_half_"](q, k, freqs_cis, q_scale, k_scale, epsilon=epsilon, rot_dim=rot_dim)
    B, S, H, D = q.shape
    rows = B * S * H
    qw, kw = (w.to(device=q.device, dtype=torch.float32).contiguous() for w in (q_scale, k_scale))
    fbs = 0 if freqs_cis.numel() == S * (rot // 2) * 4 else S * (rot // 2) * 4
    _fn("rms_rope", q.dtype)(q, k, freqs_cis, qw, kw, S, H, [*q.stride()[:3], fbs], rot // 2, float(epsilon), rows, D,
                             threads=(2 * rows * 32, 1, 1), group_size=(256, 1, 1))
    return q, k


def _mod_scale_shift(h, shift, scale, segments):
    if not (h.device.type == "mps" and h.dtype in _SUFFIX and h.is_contiguous() and h.shape[-1] % 4 == 0
            and all(isinstance(row, int) for _, _, row in segments) and _ready()):
        return _orig["_mod_scale_shift"](h, shift, scale, segments)
    D = h.shape[-1]
    fn = _fn("mod_scale_shift", h.dtype)
    for a, b, row in segments:
        if b > a:
            fn(h[a:b], scale[row].float().contiguous(), shift[row].float().contiguous(), D, threads=((b - a) * D // 4, 1, 1))
    return h


def _swiglu_q(x, linear):
    """(_swiglu(x), its int8 quantization for linear) for shared_activation, or None unless
    linear runs the int8 kernel and the shader takes x."""
    F = x.shape[-1] // 2
    if x.shape[-1] % 2 or not (x.device.type == "mps" and x.dtype in _SUFFIX and x.is_contiguous()):
        return None
    convrot = quant_target(x[..., :F], linear)
    if convrot is None or not _ready():
        return None
    rows = x.numel() // (2 * F)
    out = torch.empty(*x.shape[:-1], F, device=x.device, dtype=x.dtype)
    q = torch.empty(rows, F, device=x.device, dtype=torch.int8)
    scale = torch.empty(rows, device=x.device)
    nt = min(1024, max(32, F // 4 // 32 * 32))
    _fn("swiglu_q", x.dtype)(x, out, q, scale, F, int(convrot), threads=(rows * nt, 1, 1), group_size=(nt, 1, 1))
    return out, (convrot, q, scale)


def _mlp_forward(self, x):
    """fc2(swiglu(fc1(x))) with the swiglu shader quantizing for fc2 when that runs the int8
    kernel; otherwise the stock linear_input_act."""
    import comfy.ops
    h = self.fc1(x)
    fused = _swiglu_q(h, self.fc2)
    if fused is None:
        return comfy.ops.linear_input_act(self.fc2, h, "swiglu")
    a, quantized = fused
    with shared_activation(a, quantized):
        return self.fc2(a)


def _swiglu(x):
    F = x.shape[-1] // 2
    if not (x.device.type == "mps" and x.dtype in _SUFFIX and x.is_contiguous() and F % 4 == 0 and _ready()):
        return _orig["swiglu"](x)
    out = torch.empty(*x.shape[:-1], F, device=x.device, dtype=x.dtype)
    _fn("swiglu", x.dtype)(x, out, F, threads=(out.numel() // 4, 1, 1))
    return out


def _verify():
    global _lib
    try:
        _lib = torch.mps.compile_shader(_MSL)
    except Exception as e:
        print(f"{TAG} shader did not compile; using the stock H3 ops: {e!r}")
        return False
    import comfy.quant_ops
    g = torch.Generator().manual_seed(0)
    errs = []
    # q/k as strided views of a qkv buffer: H3's DiT (D 128, rot 96, [S, H*D] per projection)
    # and its VAE (D 64, rot 48, heads interleaved as [S, H, 3*D])
    for D, rot, interleaved in ((128, 96, False), (64, 48, True)):
        S_, H = 37, 4
        qkv = torch.randn(1, S_, 3 * H * D, generator=g).to(torch.bfloat16)
        ang = torch.rand(S_, rot // 2, generator=g) * 6
        f = torch.stack([ang.cos(), -ang.sin(), ang.sin(), ang.cos()], -1).reshape(1, S_, 1, rot // 2, 2, 2).to(torch.bfloat16)
        qw, kw = torch.randn(D, generator=g).to(torch.bfloat16), torch.randn(D, generator=g)

        def views(t):
            if interleaved:
                q, k, _ = t.view(1, S_, H, 3 * D).chunk(3, dim=-1)
                return q, k
            q, k, _ = t.split(H * D, dim=-1)
            return q.view(1, S_, H, D), k.view(1, S_, H, D)

        ref = qkv.clone()
        _orig["rms_rope_split_half_"](*views(ref), f, qw.float(), kw, epsilon=1e-5, rot_dim=rot)
        out = qkv.to("mps")
        q, k = views(out)
        rows = S_ * H
        _fn("rms_rope", torch.bfloat16)(q, k, f.to("mps"), qw.float().to("mps"), kw.to("mps"), S_, H, [*q.stride()[:3], 0],
                                        rot // 2, 1e-5, rows, D, threads=(2 * rows * 32, 1, 1), group_size=(256, 1, 1))
        errs.append(((out.cpu().float() - ref.float()).norm() / ref.float().norm()).item())
    h = torch.randn(30, 256, generator=g)
    sc, sh = torch.randn(3, 256, generator=g) * 0.1, torch.randn(3, 256, generator=g) * 0.1
    segs = [(0, 7, 1), (7, 30, 2)]
    want = _orig["_mod_scale_shift"](h.clone(), sh, sc, segs)
    got = h.to("mps", torch.bfloat16)
    for a, b, row in segs:
        _fn("mod_scale_shift", torch.bfloat16)(got[a:b], sc[row].to("mps"), sh[row].to("mps"), 256, threads=((b - a) * 64, 1, 1))
    errs.append(((got.cpu().float() - want).norm() / want.norm()).item())
    x = torch.randn(9, 2 * 64, generator=g)
    got = torch.empty(9, 64, device="mps", dtype=torch.bfloat16)
    _fn("swiglu", torch.bfloat16)(x.to("mps", torch.bfloat16), got, 64, threads=(9 * 16, 1, 1))
    want = _orig["swiglu"](x)
    errs.append(((got.cpu().float() - want).norm() / want.norm()).item())
    if not max(errs) < 0.01:
        print(f"{TAG} self-check mismatch (rel errs {[f'{e:.3g}' for e in errs]}); using the stock H3 ops.")
        return False
    return True


def install():
    global _installed
    if _installed or sys.platform != "darwin":
        return
    from . import _caps
    if not _caps.resolve("ASFP8_MINIMAX_H3_FUSED", default_on=True, cap=_caps.has_compile_shader):
        return
    try:
        import comfy.ldm.minimax.model as hm
        import comfy.ops
        import comfy.quant_ops
    except Exception:
        return
    ck = comfy.quant_ops.ck
    _orig["rms_rope_split_half_"] = ck.rms_rope_split_half_
    _orig["_mod_scale_shift"] = hm._mod_scale_shift
    _orig["swiglu"] = comfy.ops.INPUT_ACT_EAGER["swiglu"]
    ck.rms_rope_split_half_ = _rms_rope_split_half_
    comfy.ops.INPUT_ACT_EAGER["swiglu"] = _swiglu
    _orig["MLP"] = hm.MLP.forward
    hm.MLP.forward = _mlp_forward
    # block forwards (stock, and nodes that copy it such as KJ's MiniMaxLowVRAMAttention)
    # bind _mod_scale_shift by name at import
    from ._common import rebind_everywhere
    rebind_everywhere(_orig["_mod_scale_shift"], _mod_scale_shift)
    _installed = True
    print(f"{TAG} MiniMax H3 q/k norm+RoPE, adaLN modulate and SwiGLU fused on MPS, SwiGLU also int8-quantizing "
          f"for an int8 fc2 (self-check on first use).")
