"""Patch: fuse Krea2's per-block elementwise work into single-pass shaders on MPS.

Around its matmuls a Krea2 block spends ~23 ms (of ~110) at 1536x1024 in separate MPS
passes: fp32 copies for every RMSNorm, broadcast adaLN multiply/adds, q/k norm + RoPE,
repeat_interleave of the GQA k/v, and the sigmoid/SiLU/residual gating. Each fused shader
here reads its inputs once, computes in fp32 and writes once (~3 ms per block); the GQA
k/v go to attention un-repeated, and the projections of one input share one int8
activation quantize. Anything the fast path doesn't cover (reference latents, attention
patches, masks, other dtypes/devices) runs the original forward.
DEFAULT ON where compile_shader works; ASFP8_KREA2_FUSED=off disables.
"""

import sys

import torch

from .int8_linear_kernel_mps import shared_activation

TAG = "[AppleSilicon-int8-Fast/krea2]"

_installed = False
_lib = None

_MSL = r"""
#include <metal_stdlib>
using namespace metal;

// out = x * rsqrt(mean(x^2) + eps) * (1 + w) [* (1 + scale[b]) + shift[b]], one threadgroup
// per row; w is the RMSNorm's zero-centred scale, scale/shift the adaLN modulation rows
template <typename T>
inline void norm_mod(const device T* x, device T* out, const device float* w, const device T* scale,
                     const device T* shift, uint D, uint rows_per_b, uint mod_bstride, float eps,
                     uint has_mod, uint row, uint tid, uint nt, uint sg, uint lane, threadgroup float* red) {
  const device T* xr = x + ulong(row) * D;
  float ss = 0.0f;
  for (uint c = tid * 4; c < D; c += nt * 4) {
    const float4 v = float4(*(const device vec<T, 4>*)(xr + c));
    ss += dot(v, v);
  }
  ss = simd_sum(ss);
  if (lane == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ss = 0.0f;
  for (uint i = 0; i < (nt + 31) / 32; ++i) ss += red[i];
  const float r = rsqrt(ss / float(D) + eps);
  const ulong mb = ulong(row / rows_per_b) * mod_bstride;
  for (uint c = tid * 4; c < D; c += nt * 4) {
    float4 v = float4(*(const device vec<T, 4>*)(xr + c)) * r * (1.0f + *(const device float4*)(w + c));
    if (has_mod)
      v = v * (1.0f + float4(*(const device vec<T, 4>*)(scale + mb + c))) + float4(*(const device vec<T, 4>*)(shift + mb + c));
    *(device vec<T, 4>*)(out + ulong(row) * D + c) = vec<T, 4>(v);
  }
}

// x: rows of H heads x 128 (row stride in_stride) -> out [B, H, L, 128], RMSNorm with
// (1 + w), then RoPE out[2i+j] = f[i][j][0] * x[2i] + f[i][j][1] * x[2i+1] with f the
// [B, L, 64, 2, 2] fp32 table. One simdgroup per (b, l, h), 4 elements per lane.
template <typename T>
inline void qk_norm_rope(const device T* x, device T* out, const device float* w, const device float* f,
                         uint L, uint H, uint in_stride, uint f_bstride, float eps, uint has_rope, uint gid, uint lane) {
  const uint h = gid % H, l = (gid / H) % L, b = gid / (H * L);
  float4 v = float4(*(const device vec<T, 4>*)(x + ulong(b * L + l) * in_stride + h * 128 + lane * 4));
  const float r = rsqrt(simd_sum(dot(v, v)) / 128.0f + eps);
  v = v * r * (1.0f + *(const device float4*)(w + lane * 4));
  if (has_rope) {
    const device float4* fr = (const device float4*)(f + ulong(b) * f_bstride + ulong(l) * 256 + lane * 8);
    const float4 f0 = fr[0], f1 = fr[1];  // pair 2*lane: (f00, f01, f10, f11), pair 2*lane + 1
    v = float4(f0.x * v.x + f0.y * v.y, f0.z * v.x + f0.w * v.y, f1.x * v.z + f1.y * v.w, f1.z * v.z + f1.w * v.w);
  }
  *(device vec<T, 4>*)(out + (ulong(b * H + h) * L + l) * 128 + lane * 4) = vec<T, 4>(v);
}

#define KERNELS(S, T) \
kernel void norm_mod_##S(const device T* x [[buffer(0)]], device T* out [[buffer(1)]], const device float* w [[buffer(2)]], \
                         const device T* scale [[buffer(3)]], const device T* shift [[buffer(4)]], constant uint& D [[buffer(5)]], \
                         constant uint& rows_per_b [[buffer(6)]], constant uint& mod_bstride [[buffer(7)]], \
                         constant float& eps [[buffer(8)]], constant uint& has_mod [[buffer(9)]], \
                         uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                         uint nt [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]], \
                         uint lane [[thread_index_in_simdgroup]]) { \
  threadgroup float red[32]; \
  norm_mod<T>(x, out, w, scale, shift, D, rows_per_b, mod_bstride, eps, has_mod, row, tid, nt, sg, lane, red); \
} \
kernel void qk_norm_rope_##S(const device T* x [[buffer(0)]], device T* out [[buffer(1)]], const device float* w [[buffer(2)]], \
                             const device float* f [[buffer(3)]], constant uint& L [[buffer(4)]], constant uint& H [[buffer(5)]], \
                             constant uint& in_stride [[buffer(6)]], constant uint& f_bstride [[buffer(7)]], \
                             constant float& eps [[buffer(8)]], constant uint& has_rope [[buffer(9)]], \
                             uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
  qk_norm_rope<T>(x, out, w, f, L, H, in_stride, f_bstride, eps, has_rope, t / 32, lane); \
} \
/* out = x + g[b] * y, g broadcast over the rows of each batch */ \
kernel void gated_add_##S(const device T* x [[buffer(0)]], const device T* g [[buffer(1)]], const device T* y [[buffer(2)]], \
                          device T* out [[buffer(3)]], constant uint& D [[buffer(4)]], constant uint& per_b [[buffer(5)]], \
                          constant uint& g_bstride [[buffer(6)]], uint i [[thread_position_in_grid]]) { \
  const ulong e = ulong(i) * 4; \
  const float4 gv = float4(*(const device vec<T, 4>*)(g + ulong(e / per_b) * g_bstride + e % D)); \
  *(device vec<T, 4>*)(out + e) = vec<T, 4>(float4(*(const device vec<T, 4>*)(x + e)) + gv * float4(*(const device vec<T, 4>*)(y + e))); \
} \
kernel void sigmoid_mul_##S(const device T* y [[buffer(0)]], const device T* g [[buffer(1)]], device T* out [[buffer(2)]], \
                            uint i [[thread_position_in_grid]]) { \
  const float4 gv = float4(((const device vec<T, 4>*)g)[i]); \
  ((device vec<T, 4>*)out)[i] = vec<T, 4>(float4(((const device vec<T, 4>*)y)[i]) / (1.0f + exp(-gv))); \
} \
kernel void silu_mul_##S(const device T* g [[buffer(0)]], const device T* u [[buffer(1)]], device T* out [[buffer(2)]], \
                         uint i [[thread_position_in_grid]]) { \
  const float4 gv = float4(((const device vec<T, 4>*)g)[i]); \
  ((device vec<T, 4>*)out)[i] = vec<T, 4>(gv / (1.0f + exp(-gv)) * float4(((const device vec<T, 4>*)u)[i])); \
}
KERNELS(bf16, bfloat)
KERNELS(f16, half)
"""

_SUFFIX = {torch.bfloat16: "bf16", torch.float16: "f16"}

_orig = {}


def _fn(name, dtype):
    return getattr(_lib, f"{name}_{_SUFFIX[dtype]}")


def _norm_mod(x, norm, scale=None, shift=None):
    """norm(x) [* (1 + scale) + shift] for a Krea2 RMSNorm; scale/shift are [B, 1, D] rows."""
    D = x.shape[-1]
    x = x.contiguous()
    out = torch.empty_like(x)
    w = norm.scale.to(device=x.device, dtype=torch.float32).contiguous()
    rows = x.numel() // D
    if scale is None:
        scale = shift = x
        rows_per_b, bstride, has_mod = rows, 0, 0
    else:
        rows_per_b, bstride, has_mod = rows // scale.shape[0], scale.stride(0), 1
    nt = min(1024, max(32, D // 4 // 32 * 32))
    _fn("norm_mod", x.dtype)(x, out, w, scale, shift, D, rows_per_b, bstride, float(norm.eps), has_mod,
                             threads=(rows * nt, 1, 1), group_size=(nt, 1, 1))
    return out


def _qk_norm_rope(x, heads, norm, freqs):
    """[B, L, heads*128] -> RMSNorm(+RoPE) -> [B, heads, L, 128] contiguous."""
    x = x.contiguous()
    B, L = x.shape[0], x.shape[1]
    out = torch.empty(B, heads, L, 128, device=x.device, dtype=x.dtype)
    w = norm.scale.to(device=x.device, dtype=torch.float32).contiguous()
    if freqs is None:
        f, fb, has_rope = w, 0, 0
    else:
        f = freqs.contiguous()
        fb, has_rope = (f.stride(0) if f.shape[0] > 1 else 0), 1
    _fn("qk_norm_rope", x.dtype)(x, out, w, f, L, heads, x.shape[-1], fb, float(norm.eps), has_rope,
                                 threads=(B * L * heads * 32, 1, 1), group_size=(256, 1, 1))
    return out


def _gated_add(x, g, y):
    """x + g * y with g a [B, 1, D] row per batch."""
    out = torch.empty_like(x)
    D = x.shape[-1]
    _fn("gated_add", x.dtype)(x, g, y.contiguous(), out, D, x.numel() // x.shape[0], g.stride(0),
                              threads=(x.numel() // 4, 1, 1))
    return out


def _sigmoid_mul(y, g):
    out = torch.empty_like(y)
    _fn("sigmoid_mul", y.dtype)(y.contiguous(), g.contiguous(), out, threads=(y.numel() // 4, 1, 1))
    return out


def _silu_mul(g, u):
    out = torch.empty_like(g)
    _fn("silu_mul", g.dtype)(g.contiguous(), u.contiguous(), out, threads=(g.numel() // 4, 1, 1))
    return out


def _verify():
    global _lib
    try:
        _lib = torch.mps.compile_shader(_MSL)
    except Exception as e:
        print(f"{TAG} shader did not compile; using the stock Krea2 forward: {e!r}")
        return False
    import comfy.ldm.krea2.model as km
    from comfy.ldm.flux.math import _apply_rope1
    g = torch.Generator().manual_seed(0)
    norm = km.RMSNorm(256)
    norm.scale = torch.nn.Parameter(torch.randn(256, generator=g) * 0.1)
    x = torch.randn(2, 5, 256, generator=g)
    s, b = (torch.randn(2, 1, 256, generator=g) * 0.1 for _ in range(2))
    ref = (1 + s) * _orig["RMSNorm"](norm, x) + b
    out = _norm_mod(x.to("mps", torch.bfloat16), norm, s.to("mps", torch.bfloat16), b.to("mps", torch.bfloat16)).cpu().float()
    errs = [((out - ref).norm() / ref.norm()).item()]
    qn = km.RMSNorm(128)
    qn.scale = torch.nn.Parameter(torch.randn(128, generator=g) * 0.1)
    q = torch.randn(2, 7, 3 * 128, generator=g)
    f = torch.randn(2, 1, 7, 64, 2, 2, generator=g)
    ref = _apply_rope1(_orig["RMSNorm"](qn, q.unflatten(-1, (3, 128)).transpose(1, 2)), f)
    out = _qk_norm_rope(q.to("mps", torch.bfloat16), 3, qn, f.to("mps")).cpu().float()
    errs.append(((out - ref).norm() / ref.norm()).item())
    y, gate = (torch.randn(2, 5, 256, generator=g) for _ in range(2))
    for got, want in ((_gated_add(x.to("mps", torch.bfloat16), s.to("mps", torch.bfloat16), y.to("mps", torch.bfloat16)), x + s * y),
                      (_sigmoid_mul(y.to("mps", torch.bfloat16), gate.to("mps", torch.bfloat16)), y * torch.sigmoid(gate)),
                      (_silu_mul(gate.to("mps", torch.bfloat16), y.to("mps", torch.bfloat16)), torch.nn.functional.silu(gate) * y)):
        errs.append(((got.cpu().float() - want).norm() / want.norm()).item())
    if not max(errs) < 0.01:
        print(f"{TAG} self-check mismatch (rel errs {[f'{e:.3g}' for e in errs]}); using the stock Krea2 forward.")
        return False
    return True


def _fast(x):
    from . import _caps
    return (x.device.type == "mps" and x.dtype in _SUFFIX and x.shape[-1] % 4 == 0
            and _caps.kernel_ready("krea2", _verify))


def _rmsnorm_forward(self, x):
    if not _fast(x):
        return _orig["RMSNorm"](self, x)
    return _norm_mod(x, self)


def _swiglu_forward(self, x):
    if not _fast(x):
        return _orig["SwiGLU"](self, x)
    with shared_activation(x):
        g, u = self.gate(x), self.up(x)
    return self.down(_silu_mul(g, u))


def _attention_forward(self, x, freqs=None, mask=None, transformer_options={}):
    patches = transformer_options.get("patches", {})
    if (mask is not None or self.headdim != 128 or not _fast(x)
            or ("block_index" in transformer_options and ("attn1_patch" in patches or "attn1_output_patch" in patches))):
        return _orig["Attention"](self, x, freqs, mask, transformer_options=transformer_options)
    import comfy.ldm.krea2.model as km
    with shared_activation(x):
        q, k, v, gate = self.wq(x), self.wk(x), self.wv(x), self.gate(x)
    q = _qk_norm_rope(q, self.heads, self.qknorm.qnorm, freqs)
    k = _qk_norm_rope(k, self.kvheads, self.qknorm.knorm, freqs)
    v = v.unflatten(-1, (self.kvheads, self.headdim)).transpose(1, 2)
    out = km.optimized_attention_masked(
        km.AttentionTensorContainer(q), km.AttentionTensorContainer(k), km.AttentionTensorContainer(v), self.heads,
        skip_reshape=True, enable_gqa=self.kvheads != self.heads, preferred_attention=self.comfy_attention,
        transformer_options=transformer_options)
    return self.wo(_sigmoid_mul(out, gate))


def _block_forward(self, x, vec, freqs, mask=None, timestep_zero_index=None, transformer_options={}):
    if timestep_zero_index is not None or not _fast(x):
        return _orig["SingleStreamBlock"](self, x, vec, freqs, mask, timestep_zero_index, transformer_options)
    prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
    h = _norm_mod(x, self.prenorm, prescale, preshift)
    x = _gated_add(x, pregate, self.attn(h, freqs, mask, transformer_options=transformer_options))
    h = _norm_mod(x, self.postnorm, postscale, postshift)
    return _gated_add(x, postgate, self.mlp(h))


def _last_forward(self, x, tvec):
    if not _fast(x):
        return _orig["LastLayer"](self, x, tvec)
    scale, shift = self.modulation(tvec)
    return self.linear(_norm_mod(x, self.norm, scale, shift))


def install():
    global _installed
    if _installed or sys.platform != "darwin":
        return
    from . import _caps
    if not _caps.resolve("ASFP8_KREA2_FUSED", default_on=True, cap=_caps.has_compile_shader):
        return
    try:
        import comfy.ldm.krea2.model as km
    except Exception:
        return
    for cls, fn in ((km.RMSNorm, _rmsnorm_forward), (km.SwiGLU, _swiglu_forward), (km.Attention, _attention_forward),
                    (km.SingleStreamBlock, _block_forward), (km.LastLayer, _last_forward)):
        _orig[cls.__name__] = cls.forward
        cls.forward = fn
    _installed = True
    print(f"{TAG} Krea2 norm/modulation/RoPE/gating fused on MPS (self-check on first use).")
