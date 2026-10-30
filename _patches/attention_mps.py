"""Patch: back ComfyUI's optimized attention on MPS with a Neural-Accelerator flash kernel.

On MPS ComfyUI selects attention_sub_quad (chunked baddbmm + softmax, ~23 TFLOPS on an
M5 Max). This shader runs S = QK^T and O += PV as matmul2d on the tensor units with the
online softmax in threadgroup memory (~47 TFLOPS at Krea2/Flux shapes), never forms the
score matrix, and reads GQA k/v heads in place. Masked or otherwise unsupported calls go
to the attention function ComfyUI picked. DEFAULT ON where the shader builds and passes
its self-check; ASFP8_ATTENTION=off disables.
"""

import sys

import torch

TAG = "[AppleSilicon-int8-Fast/attention]"

_installed = False
_libs = {}

# BQ query rows x BKV keys per step, 4 simdgroups: the fastest tiling on M5 Max at
# Krea2/Flux shapes; S (fp32) + P (fp16) use 24 KB of threadgroup memory
_MSL = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int BQ = 32, BKV = 128, NSG = 4, D = @D@;
constant constexpr int TPR = NSG * 32 / BQ, CPT = BKV / TPR;  // threads per row, columns per thread
constant constexpr int OCAP = BQ * D / (NSG * 32);           // O elements per thread

// q/k/v: [B, H, L, D], any batch/head/row strides (D contiguous); o: [B, Lq, H, D].
// P = {Lq, Lk, H, gqa, q strides (b, h, l), k strides, v strides}
template <typename T>
inline void attn(device T* Q, device T* K, device T* V, device T* O, constant long* P, float scale,
                 uint3 tgid, uint tid, threadgroup float* Ss, threadgroup half* Ps, threadgroup float* rowbuf,
                 threadgroup uint* rescale) {
  const int Lq = P[0], Lk = P[1], H = P[2], gqa = P[3];
  // x enumerates (kv head, q tile, q head in its GQA group) so the q heads sharing a k/v
  // tile run back to back and hit it in cache
  const int b = tgid.z, tiles = (Lq + BQ - 1) / BQ, grp = int(tgid.x) % (tiles * gqa);
  const int h = int(tgid.x) / (tiles * gqa) * gqa + grp % gqa, q0 = grp / gqa * BQ;
  const int qr = min(BQ, Lq - q0);
  device T* Qp = Q + b * P[4] + h * P[5] + q0 * P[6];
  device T* Kp = K + b * P[7] + (h / gqa) * P[8];
  device T* Vp = V + b * P[10] + (h / gqa) * P[11];
  const int qsl = P[6], ksl = P[9], vsl = P[12];

  constexpr auto dqk = matmul2d_descriptor(BQ, BKV, D, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<dqk, execution_simdgroups<NSG>> qk;
  constexpr auto dpv = matmul2d_descriptor(BQ, D, BKV, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dpv, execution_simdgroups<NSG>> pv;

  auto tQ = tensor(Qp, dextents<int, 2>{D, qr}, array<int, 2>{1, qsl});
  auto tS = tensor(Ss, dextents<int, 2>{BKV, BQ}, array<int, 2>{1, BKV});
  auto tP0 = tensor(Ps, dextents<int, 2>{BKV, BQ}, array<int, 2>{1, BKV});
  auto tV0 = tensor(Vp, dextents<int, 2>{D, BKV}, array<int, 2>{1, vsl});
  using PT = __tensor_ops_detail::__remove_addrspace_t<decltype(tP0)>;
  using VT = __tensor_ops_detail::__remove_addrspace_t<decltype(tV0)>;
  auto Oc = pv.get_destination_cooperative_tensor<PT, VT, float>();
  for (uint16_t i = 0; i < Oc.get_capacity(); ++i) if (Oc.is_valid_element(i)) Oc[i] = 0.0f;
  // the element -> row map is fixed for the kernel, so look it up once
  ushort orow[OCAP];
  for (uint16_t i = 0; i < OCAP; ++i) orow[i] = Oc.get_multidimensional_index(i)[1];

  const int row = tid / TPR, c0 = (tid % TPR) * CPT;
  float m = -FLT_MAX, l = 0.0f;
  const float sl2 = scale * M_LOG2E_F;
  if (tid == 0) rescale[0] = 0;
  for (int kb = 0, it = 0; kb < Lk; kb += BKV, ++it) {
    const int kr = min(BKV, Lk - kb);
    auto tK = tensor(Kp + kb * ksl, dextents<int, 2>{D, kr}, array<int, 2>{1, ksl});
    qk.run(tQ, tK, tS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) rescale[(it + 1) & 1] = 0;   // last read in the previous block, before this barrier
    const threadgroup float4* sr = (const threadgroup float4*)(Ss + row * BKV + c0);
    float sv[CPT];
    for (int j = 0; j < CPT / 4; ++j) {
      const float4 t = sr[j];
      sv[4 * j] = t.x; sv[4 * j + 1] = t.y; sv[4 * j + 2] = t.z; sv[4 * j + 3] = t.w;
    }
    if (kr < BKV) for (int j = 0; j < CPT; ++j) if (c0 + j >= kr) sv[j] = -FLT_MAX;
    float bm = -FLT_MAX;
    for (int j = 0; j < CPT; ++j) bm = max(bm, sv[j]);
    for (int o = 1; o < TPR; o <<= 1) bm = max(bm, simd_shuffle_xor(bm, ushort(o)));
    // keep the stale max unless the block's beats it by 2^8: p <= 256 still fits P, O and l share
    // the offset so the result is unchanged, and most blocks then skip rescaling O
    const bool moved = bm * sl2 > m + 8.0f;
    const float mn = moved ? bm * sl2 : m;
    const float alpha = fast::exp2(m - mn);
    m = mn;
    float bs = 0.0f;
    threadgroup half4* pr = (threadgroup half4*)(Ps + row * BKV + c0);
    for (int j = 0; j < CPT / 4; ++j) {
      const float4 p = float4(fast::exp2(sv[4 * j] * sl2 - mn), fast::exp2(sv[4 * j + 1] * sl2 - mn),
                              fast::exp2(sv[4 * j + 2] * sl2 - mn), fast::exp2(sv[4 * j + 3] * sl2 - mn));
      bs += p.x + p.y + p.z + p.w;
      pr[j] = half4(p);
    }
    for (int o = 1; o < TPR; o <<= 1) bs += simd_shuffle_xor(bs, ushort(o));
    l = l * alpha + bs;
    if (tid % TPR == 0) {
      rowbuf[row] = alpha;
      if (moved) rescale[it & 1] = 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rescale[it & 1]) for (uint16_t i = 0; i < OCAP; ++i) Oc[i] *= rowbuf[orow[i]];
    // the next block's first barrier also orders this PV read of Ps before Ps is rewritten
    auto tP = tensor(Ps, dextents<int, 2>{kr, BQ}, array<int, 2>{1, BKV});
    auto tV = tensor(Vp + kb * vsl, dextents<int, 2>{D, kr}, array<int, 2>{1, vsl});
    pv.run(tP, tV, Oc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid % TPR == 0) rowbuf[row] = 1.0f / l;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device T* Op = O + (long(b) * Lq + q0) * H * D + h * D;
  for (uint16_t i = 0; i < Oc.get_capacity(); ++i) {
    if (!Oc.is_valid_element(i)) continue;
    auto idx = Oc.get_multidimensional_index(i);
    if (idx[1] < qr) Op[long(idx[1]) * H * D + idx[0]] = T(Oc[i] * rowbuf[idx[1]]);
  }
}

#define ATTN(NAME, T) \
kernel void NAME(device T* Q [[buffer(0)]], device T* K [[buffer(1)]], device T* V [[buffer(2)]], \
                 device T* O [[buffer(3)]], constant long* P [[buffer(4)]], constant float& scale [[buffer(5)]], \
                 uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
  threadgroup float Ss[BQ * BKV]; \
  threadgroup half Ps[BQ * BKV]; \
  threadgroup float rowbuf[BQ]; \
  threadgroup uint rescale[2]; \
  attn<T>(Q, K, V, O, P, scale, tgid, tid, Ss, Ps, rowbuf, rescale); \
}
ATTN(attn_bf16, bfloat)
ATTN(attn_f16, half)
"""

# int8 attention, on by default (INT8_ATTENTION=off disables it), slightly lossy: S = QK^T runs as int8
# matmul2d, the rest is the bf16 kernel (fp16 P, PV on the input dtype). Q and K are rotated by
# the normalized Hadamard H128 ((QH)(KH)^T = QK^T, outliers spread over the head) and quantized
# per row, K minus its sequence mean first (softmax is unchanged by that per-query constant).
# int8 P instead lost 4-22% of the output on Krea2's long-tailed attention; this costs ~0.5%.
_MSL_I8 = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int BQ = 32, BKV = 128, NSG = 4, D = @D@, E = D / 32;
constant constexpr int TPR = NSG * 32 / BQ, CPT = BKV / TPR;
constant constexpr int OCAP = BQ * D / (NSG * 32);

// One simdgroup per row r of x [B, H, L, D] (P = {L, H, strides b, h, l}): x8 = int8 of
// (x - mu[b, h] when has_mu) H_D / sqrt(D), contiguous [B, H, L, D], with the row's scale in xs.
// Lane t holds elements t*E..t*E+E-1: H_D = H_E (x) H_32, H_E in lane, H_32 across lanes.
template <typename T>
inline void quant8(const device T* x, device int8_t* x8, device float* xs, const device float* mu,
                   constant long* P, uint has_mu, uint r, uint lane) {
  const long L = P[0], H = P[1], l = r % L, h = (r / L) % H, b = r / (L * H);
  const device T* xr = x + b * P[2] + h * P[3] + l * P[4] + lane * E;
  const long mb = (b * H + h) * D + lane * E;
  float v[E];
  for (int e = 0; e < E; ++e) v[e] = float(xr[e]) - (has_mu ? mu[mb + e] : 0.0f);
  for (int s = 1; s < E; s <<= 1)
    for (int e = 0; e < E; ++e) if (!(e & s)) { const float a = v[e], c = v[e + s]; v[e] = a + c; v[e + s] = a - c; }
  for (ushort s = 1; s < 32; s <<= 1)
    for (int e = 0; e < E; ++e) { const float o = simd_shuffle_xor(v[e], s); v[e] = (lane & s) ? o - v[e] : v[e] + o; }
  float amax = 0.0f;
  for (int e = 0; e < E; ++e) { v[e] *= rsqrt(float(D)); amax = max(amax, fabs(v[e])); }
  amax = simd_max(amax);
  const float s = max(amax / 127.0f, 1e-30f), inv = 1.0f / s;
  if (lane == 0) xs[r] = s;
  for (int e = 0; e < E; ++e) x8[long(r) * D + lane * E + e] = int8_t(clamp(rint(v[e] * inv), -127.0f, 127.0f));
}

// The bf16 kernel (attention_mps._MSL) with S = (q8 k8^T) * qs[row] * ks[key]: q8 [B, H, Lq, D],
// k8 [B, Hk, Lk, D] contiguous, qs [B, H, Lq], ks [B, Hk, Lk]; v [B, Hk, Lk, D] any strides,
// o [B, Lq, H, D]. P = {Lq, Lk, H, gqa, v strides (b, h, l)}
template <typename T>
inline void attn_qk8(device int8_t* Q, device int8_t* K, device T* V, device T* O, const device float* qs,
                     const device float* ks, constant long* P, float scale, uint3 tgid, uint tid,
                     threadgroup int* Si, threadgroup half* Ps, threadgroup float* rowbuf, threadgroup uint* rescale) {
  const int Lq = P[0], Lk = P[1], H = P[2], gqa = P[3], Hk = H / gqa;
  const int b = tgid.z, tiles = (Lq + BQ - 1) / BQ, grp = int(tgid.x) % (tiles * gqa);
  const int h = int(tgid.x) / (tiles * gqa) * gqa + grp % gqa, q0 = grp / gqa * BQ, hk = h / gqa;
  const int qr = min(BQ, Lq - q0);
  device int8_t* Qp = Q + ((long(b) * H + h) * Lq + q0) * D;
  device int8_t* Kp = K + (long(b) * Hk + hk) * Lk * D;
  device T* Vp = V + b * P[4] + hk * P[5];
  const int vsl = P[6];
  const device float* ksp = ks + (long(b) * Hk + hk) * Lk;

  constexpr auto dqk = matmul2d_descriptor(BQ, BKV, D, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<dqk, execution_simdgroups<NSG>> qk;
  constexpr auto dpv = matmul2d_descriptor(BQ, D, BKV, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<dpv, execution_simdgroups<NSG>> pv;

  auto tQ = tensor(Qp, dextents<int, 2>{D, qr}, array<int, 2>{1, D});
  auto tS = tensor(Si, dextents<int, 2>{BKV, BQ}, array<int, 2>{1, BKV});
  auto tP0 = tensor(Ps, dextents<int, 2>{BKV, BQ}, array<int, 2>{1, BKV});
  auto tV0 = tensor(Vp, dextents<int, 2>{D, BKV}, array<int, 2>{1, vsl});
  using PT = __tensor_ops_detail::__remove_addrspace_t<decltype(tP0)>;
  using VT = __tensor_ops_detail::__remove_addrspace_t<decltype(tV0)>;
  auto Oc = pv.get_destination_cooperative_tensor<PT, VT, float>();
  for (uint16_t i = 0; i < Oc.get_capacity(); ++i) if (Oc.is_valid_element(i)) Oc[i] = 0.0f;
  ushort orow[OCAP];
  for (uint16_t i = 0; i < OCAP; ++i) orow[i] = Oc.get_multidimensional_index(i)[1];

  const int row = tid / TPR, c0 = (tid % TPR) * CPT;
  const float qsl = (row < qr ? qs[(long(b) * H + h) * Lq + q0 + row] : 0.0f) * scale * M_LOG2E_F;
  float m = -FLT_MAX, l = 0.0f;
  if (tid == 0) rescale[0] = 0;
  for (int kb = 0, it = 0; kb < Lk; kb += BKV, ++it) {
    const int kr = min(BKV, Lk - kb);
    auto tK = tensor(Kp + long(kb) * D, dextents<int, 2>{D, kr}, array<int, 2>{1, D});
    qk.run(tQ, tK, tS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) rescale[(it + 1) & 1] = 0;   // last read in the previous block, before this barrier
    float sv[CPT];
    float bm = -FLT_MAX;
    if (kr == BKV) {
      const threadgroup int4* sr = (const threadgroup int4*)(Si + row * BKV + c0);
      const device float4* kq = (const device float4*)(ksp + kb + c0);
      for (int j = 0; j < CPT / 4; ++j) {
        const float4 t = float4(sr[j]) * qsl * kq[j];
        sv[4 * j] = t.x; sv[4 * j + 1] = t.y; sv[4 * j + 2] = t.z; sv[4 * j + 3] = t.w;
        bm = max(bm, max(max(t.x, t.y), max(t.z, t.w)));
      }
    } else {
      for (int j = 0; j < CPT; ++j) {
        sv[j] = c0 + j < kr ? float(Si[row * BKV + c0 + j]) * qsl * ksp[kb + c0 + j] : -FLT_MAX;
        bm = max(bm, sv[j]);
      }
    }
    for (int o = 1; o < TPR; o <<= 1) bm = max(bm, simd_shuffle_xor(bm, ushort(o)));
    // the bf16 kernel's lazy max: p <= 256 fits fp16 P, and most blocks skip rescaling O
    const bool moved = bm > m + 8.0f;
    const float mn = moved ? bm : m;
    const float alpha = fast::exp2(m - mn);
    m = mn;
    float bs = 0.0f;
    threadgroup half4* pr = (threadgroup half4*)(Ps + row * BKV + c0);
    for (int j = 0; j < CPT / 4; ++j) {
      const float4 p = float4(fast::exp2(sv[4 * j] - mn), fast::exp2(sv[4 * j + 1] - mn),
                              fast::exp2(sv[4 * j + 2] - mn), fast::exp2(sv[4 * j + 3] - mn));
      bs += p.x + p.y + p.z + p.w;
      pr[j] = half4(p);
    }
    for (int o = 1; o < TPR; o <<= 1) bs += simd_shuffle_xor(bs, ushort(o));
    l = l * alpha + bs;
    if (tid % TPR == 0) {
      rowbuf[row] = alpha;
      if (moved) rescale[it & 1] = 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rescale[it & 1]) for (uint16_t i = 0; i < OCAP; ++i) Oc[i] *= rowbuf[orow[i]];
    auto tP = tensor(Ps, dextents<int, 2>{kr, BQ}, array<int, 2>{1, BKV});
    auto tV = tensor(Vp + long(kb) * vsl, dextents<int, 2>{D, kr}, array<int, 2>{1, vsl});
    pv.run(tP, tV, Oc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid % TPR == 0) rowbuf[row] = 1.0f / l;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device T* Op = O + (long(b) * Lq + q0) * H * D + h * D;
  for (uint16_t i = 0; i < Oc.get_capacity(); ++i) {
    if (!Oc.is_valid_element(i)) continue;
    auto idx = Oc.get_multidimensional_index(i);
    if (idx[1] < qr) Op[long(idx[1]) * H * D + idx[0]] = T(Oc[i] * rowbuf[idx[1]]);
  }
}

#define KERNELS(S, T) \
kernel void quant8_##S(const device T* x [[buffer(0)]], device int8_t* x8 [[buffer(1)]], device float* xs [[buffer(2)]], \
                       const device float* mu [[buffer(3)]], constant long* P [[buffer(4)]], constant uint& has_mu [[buffer(5)]], \
                       uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
  quant8<T>(x, x8, xs, mu, P, has_mu, t / 32, lane); \
} \
kernel void attn_qk8_##S(device int8_t* Q [[buffer(0)]], device int8_t* K [[buffer(1)]], device T* V [[buffer(2)]], \
                         device T* O [[buffer(3)]], const device float* qs [[buffer(4)]], const device float* ks [[buffer(5)]], \
                         constant long* P [[buffer(6)]], constant float& scale [[buffer(7)]], \
                         uint3 tgid [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) { \
  threadgroup int Si[BQ * BKV]; \
  threadgroup half Ps[BQ * BKV]; \
  threadgroup float rowbuf[BQ]; \
  threadgroup uint rescale[2]; \
  attn_qk8<T>(Q, K, V, O, qs, ks, P, scale, tgid, tid, Si, Ps, rowbuf, rescale); \
}
KERNELS(bf16, bfloat)
KERNELS(f16, half)
"""
_libs_i8 = {}
_use_i8 = False

_SUFFIX = {torch.bfloat16: "bf16", torch.float16: "f16"}


def _flash(q, k, v, scale):
    """q: [B, H, Lq, D], k/v: [B, Hkv, Lk, D] (any strides, D contiguous) -> [B, Lq, H, D]."""
    if _use_i8 and q.shape[-1] % 32 == 0 and q.shape[-1] & (q.shape[-1] // 32 - 1) == 0:
        from . import _caps
        D = q.shape[-1]
        if _caps.kernel_ready(f"attention_i8_d{D}", lambda: _verify_i8(D)):
            return _flash_i8(q, k, v, scale)
    return _flash_bf16(q, k, v, scale)


def _flash_bf16(q, k, v, scale):
    B, H, Lq, D = q.shape
    Hk, Lk = k.shape[1], k.shape[2]
    o = torch.empty(B, Lq, H, D, device=q.device, dtype=q.dtype)
    P = [Lq, Lk, H, H // Hk, *q.stride()[:3], *k.stride()[:3], *v.stride()[:3]]
    getattr(_libs[D], "attn_" + _SUFFIX[q.dtype])(
        q, k, v, o, P, float(scale), threads=(-(-Lq // 32) * H * 128, 1, B), group_size=(128, 1, 1))
    return o


def _flash_i8(q, k, v, scale):
    """_flash with S = QK^T as int8 matmul2d (see _MSL_I8)."""
    B, H, Lq, D = q.shape
    Hk, Lk = k.shape[1], k.shape[2]
    lib, sfx, dev = _libs_i8[D], _SUFFIX[q.dtype], q.device
    mk = k.mean(2, dtype=torch.float32).contiguous()
    q8 = torch.empty(B, H, Lq, D, device=dev, dtype=torch.int8)
    k8 = torch.empty(B, Hk, Lk, D, device=dev, dtype=torch.int8)
    qs = torch.empty(B, H, Lq, device=dev)
    ks = torch.empty(B, Hk, Lk, device=dev)
    quant = getattr(lib, "quant8_" + sfx)
    quant(q, q8, qs, mk, [Lq, H, *q.stride()[:3]], 0, threads=(B * H * Lq * 32, 1, 1), group_size=(256, 1, 1))
    quant(k, k8, ks, mk, [Lk, Hk, *k.stride()[:3]], 1, threads=(B * Hk * Lk * 32, 1, 1), group_size=(256, 1, 1))
    o = torch.empty(B, Lq, H, D, device=dev, dtype=q.dtype)
    getattr(lib, "attn_qk8_" + sfx)(q8, k8, v, o, qs, ks, [Lq, Lk, H, H // Hk, *v.stride()[:3]], float(scale),
                                    threads=(-(-Lq // 32) * H * 128, 1, B), group_size=(128, 1, 1))
    return o


def _verify_i8(D):
    try:
        _libs_i8[D] = torch.mps.compile_shader(_MSL_I8.replace("@D@", str(D)))
    except Exception as e:
        print(f"{TAG} int8 attention head_dim {D} shader did not compile; using the bf16 kernel: {e!r}")
        return False
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 4, 70, D, generator=g).to(torch.bfloat16)
    k, v = (torch.randn(1, 2, 150, D, generator=g).to(torch.bfloat16) for _ in range(2))
    k = k + 3.0   # a large shared component, which the K-centering must absorb
    out = _flash_i8(q.to("mps"), k.to("mps"), v.to("mps"), D ** -0.5).cpu().float().transpose(1, 2)
    kr, vr = k.float().repeat_interleave(2, 1), v.float().repeat_interleave(2, 1)
    ref = torch.softmax(q.float() @ kr.transpose(-1, -2) * D ** -0.5, -1) @ vr
    err = ((out - ref).norm() / ref.norm()).item()
    if not err < 0.02:
        print(f"{TAG} int8 attention head_dim {D} self-check mismatch (rel err {err:.3g}); using the bf16 kernel.")
        return False
    return True


def _verify(D):
    try:
        _libs[D] = torch.mps.compile_shader(_MSL.replace("@D@", str(D)))
    except Exception as e:
        print(f"{TAG} head_dim {D} shader did not compile; using ComfyUI's attention for it: {e!r}")
        return False
    g = torch.Generator().manual_seed(0)
    # 2 kv heads for 4 q heads, Lk not a multiple of the tile: covers GQA and the ragged edges
    q = torch.randn(1, 4, 70, D, generator=g).to(torch.bfloat16)
    k, v = (torch.randn(1, 2, 150, D, generator=g).to(torch.bfloat16) for _ in range(2))
    out = _flash_bf16(q.to("mps"), k.to("mps"), v.to("mps"), D ** -0.5).cpu().float().transpose(1, 2)
    kr, vr = k.float().repeat_interleave(2, 1), v.float().repeat_interleave(2, 1)
    ref = torch.softmax(q.float() @ kr.transpose(-1, -2) * D ** -0.5, -1) @ vr
    err = ((out - ref).norm() / ref.norm()).item()
    if not err < 0.01:
        print(f"{TAG} head_dim {D} self-check mismatch (rel err {err:.3g}); using ComfyUI's attention for it.")
        return False
    return True


def _make_attention(orig):
    from comfy.ldm.modules.attention import wrap_attn

    @wrap_attn
    def attention_mps(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
                      skip_output_reshape=False, **kwargs):
        if mask is None and q.device.type == "mps":
            if skip_reshape:
                qh, kh, vh = q, k, v
            else:
                dh = q.shape[-1] // heads
                qh, kh, vh = (t.unflatten(-1, (-1, dh)).transpose(1, 2) for t in (q, k, v))
            if _eligible(qh, kh, vh):
                out = _flash(qh, kh, vh, kwargs.get("scale", qh.shape[-1] ** -0.5))
                if skip_output_reshape:
                    return out.transpose(1, 2)
                return out.view(out.shape[0], out.shape[1], -1)
        return orig(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape, **kwargs)

    return attention_mps


def _eligible(q, k, v):
    """Shapes/dtypes the kernel takes: [B, H, L, D] with D contiguous, GQA by head ratio."""
    from . import _caps
    D = q.shape[-1]
    return (q.device.type == "mps" and q.dim() == 4 and k.dim() == 4 and v.dim() == 4
            and q.dtype in _SUFFIX and k.dtype == q.dtype and v.dtype == q.dtype
            and D % 16 == 0 and 16 <= D <= 256 and k.shape[-1] == D and v.shape[-1] == D
            and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
            and q.shape[0] == k.shape[0] == v.shape[0] and k.shape[1] == v.shape[1] and k.shape[2] == v.shape[2]
            and q.shape[1] % k.shape[1] == 0
            and _caps.kernel_ready(f"attention_d{D}", lambda: _verify(D)))


def _make_sdpa(orig):
    """F.scaled_dot_product_attention on MPS: unmasked non-causal calls go to the kernel.
    This is the seam ComfyUI's "pytorch attention" (attention_pytorch, ModelAttentionBackend)
    and custom nodes reach, which the optimized_attention rebinding above never sees."""
    def sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kwargs):
        if (attn_mask is None and not dropout_p and not is_causal and not kwargs
                and (enable_gqa or query.shape[1] == key.shape[1]) and _eligible(query, key, value)):
            out = _flash(query, key, value, scale if scale is not None else query.shape[-1] ** -0.5)
            return out.transpose(1, 2)
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                    scale=scale, enable_gqa=enable_gqa, **kwargs)
    return sdpa


def install():
    global _installed
    if _installed or sys.platform != "darwin":
        return
    from . import _caps
    if not _caps.resolve("ASFP8_ATTENTION", default_on=True, cap=_caps.shader_gate):
        return
    try:
        import comfy.ldm.modules.attention as attention
    except Exception as e:
        print(f"{TAG} could not import comfy attention: {e!r}")
        return

    # model modules and custom nodes bind these by name at import
    from ._common import rebind_everywhere
    fn = attention.optimized_attention
    rebind_everywhere(fn, _make_attention(fn))
    if attention.optimized_attention_masked is not attention.optimized_attention:
        fn = attention.optimized_attention_masked
        rebind_everywhere(fn, _make_attention(fn))

    torch.nn.functional.scaled_dot_product_attention = _make_sdpa(torch.nn.functional.scaled_dot_product_attention)

    global _use_i8
    _use_i8 = _caps.resolve("INT8_ATTENTION", default_on=True, cap=_caps.shader_gate)
    if _use_i8:
        print(f"{TAG} int8 attention on: QK^T as int8 matmul2d, slightly lossy (INT8_ATTENTION=off disables it).")

    _installed = True
    print(f"{TAG} ComfyUI attention and F.scaled_dot_product_attention on MPS -> matmul2d flash "
          f"kernel (unmasked calls; the shader compiles and runs its self-check on first use, per head_dim).")
