"""Patch: Sol-Attn block-sparse attention (ComfyUI's "Model Sparse Attention" node) on MPS.

comfy_kitchen ships Sol-Attn for CUDA/HIP only, and the node keeps every non-CUDA call
dense. This runs comfy_kitchen's eager reference algorithm (tau or top-k routing, sinks,
pooled tail) with shaders: one pass pools each 64-token block, the routing and the pooled
tail of every query block come from small block-level tensor ops, and a flash kernel
starts each query tile's online softmax from that pooled state and visits only the key
blocks routed exact. MiniMax H3 takes the node's generic attention-override path here
(its chunked qkv producer is CUDA-only). DEFAULT ON where the shaders build and pass their
self-check; ASFP8_SOL_ATTN=off disables.
"""

import sys

import torch

TAG = "[AppleSilicon-int8-Fast/sol_attn]"

_installed = False
_lib = None
_orig = {}
BLOCK = 64
_LOG2E = 1.4426950408889634

_MSL = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int BQ = 32, BKV = 64, NSG = 4, D = 128, BLOCK = 64;
constant constexpr int TPR = NSG * 32 / BQ, CPT = BKV / TPR;

// P = {T, H, n, q strides (b, t, h), k strides, v strides}; q/k/v [B, T, H, 128]
// per (b, h, 64-block): centroid (mean q), kc (mean k), vc (sum v), fp32 [B*H, n, D]
template <typename T>
inline void pool(const device T* Q, const device T* K, const device T* V, device float* qc, device float* kc,
                 device float* vc, constant long* P, uint g, uint lane) {
  const int Tn = P[0], H = P[1], n = P[2];
  const int j = g % n, h = (g / n) % H, b = g / (n * H);
  const int r0 = j * BLOCK, r1 = min(Tn, r0 + BLOCK);
  float4 sq = 0.0f, sk = 0.0f, sv = 0.0f;
  for (int r = r0; r < r1; ++r) {
    sq += float4(*(const device vec<T, 4>*)(Q + b * P[3] + r * P[4] + h * P[5] + lane * 4));
    sk += float4(*(const device vec<T, 4>*)(K + b * P[6] + r * P[7] + h * P[8] + lane * 4));
    sv += float4(*(const device vec<T, 4>*)(V + b * P[9] + r * P[10] + h * P[11] + lane * 4));
  }
  const ulong o = (ulong(b * H + h) * n + j) * D + lane * 4;
  const float inv = 1.0f / float(r1 - r0);
  *(device float4*)(qc + o) = sq * inv;
  *(device float4*)(kc + o) = sk * inv;
  *(device float4*)(vc + o) = sv;
}

// One BQ-row query tile (inside one 64-row Sol-Attn query block). The online softmax starts
// from the block's pooled tail (pm: max, pl: denominator, po: numerator, log2 domain) and
// visits only the key blocks the routing mask marks exact; exact logits use the centred
// key, q.(k - kmean), like the reference.
template <typename T>
inline void sattn(device T* Q, device T* K, device T* V, device T* O, constant long* P,
                  const device uchar* mask, const device float* kmean, const device float* pm,
                  const device float* pl, const device float* po, float scale,
                  uint3 tgid, uint tid, threadgroup float* Ss, threadgroup half* Ps, threadgroup float* rowbuf) {
  const int Tn = P[0], H = P[1], n = P[2];
  const int b = tgid.z, tiles = (Tn + BQ - 1) / BQ, h = int(tgid.x) / tiles, q0 = int(tgid.x) % tiles * BQ;
  const int qr = min(BQ, Tn - q0), qb = q0 / BLOCK, bh = b * H + h;
  device T* Qp = Q + b * P[3] + q0 * P[4] + h * P[5];
  device T* Kp = K + b * P[6] + h * P[8];
  device T* Vp = V + b * P[9] + h * P[11];
  const int qsl = P[4], ksl = P[7], vsl = P[10];
  const device uchar* mrow = mask + (long(bh) * n + qb) * n;
  const long pr_ = long(bh) * n + qb;

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
  for (uint16_t i = 0; i < Oc.get_capacity(); ++i)
    if (Oc.is_valid_element(i)) Oc[i] = po[pr_ * D + Oc.get_multidimensional_index(i)[0]];

  const int row = tid / TPR, c0 = (tid % TPR) * CPT;
  // q.kmean for this row: TPR threads x 32 dims, combined over the row's lanes
  float qkm = 0.0f;
  if (row < qr) {
    const device T* qrow = Qp + row * qsl + (tid % TPR) * (D / TPR);
    const device float* km = kmean + long(bh) * D + (tid % TPR) * (D / TPR);
    for (int d = 0; d < D / TPR; d += 4) qkm += dot(float4(*(const device vec<T, 4>*)(qrow + d)), *(const device float4*)(km + d));
  }
  for (int o = 1; o < TPR; o <<= 1) qkm += simd_shuffle_xor(qkm, ushort(o));
  float m = pm[pr_], l = pl[pr_];
  const float sl2 = scale * M_LOG2E_F;
  for (int kb = 0; kb < n; ++kb) {
    if (!mrow[kb]) continue;   // uniform over the tile: all its rows share one query block
    const int k0 = kb * BLOCK, kr = min(BKV, Tn - k0);
    auto tK = tensor(Kp + k0 * ksl, dextents<int, 2>{D, kr}, array<int, 2>{1, ksl});
    qk.run(tQ, tK, tS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup float4* sr = (const threadgroup float4*)(Ss + row * BKV + c0);
    float sv[CPT];
    for (int j = 0; j < CPT / 4; ++j) {
      const float4 t = sr[j];
      sv[4 * j] = t.x; sv[4 * j + 1] = t.y; sv[4 * j + 2] = t.z; sv[4 * j + 3] = t.w;
    }
    float bm = -FLT_MAX;
    for (int j = 0; j < CPT; ++j) {
      sv[j] = (c0 + j < kr) ? (sv[j] - qkm) * sl2 : -FLT_MAX;
      bm = max(bm, sv[j]);
    }
    for (int o = 1; o < TPR; o <<= 1) bm = max(bm, simd_shuffle_xor(bm, ushort(o)));
    const float mn = max(m, bm);
    const float alpha = fast::exp2(m - mn);
    m = mn;
    float bs = 0.0f;
    threadgroup half4* pw = (threadgroup half4*)(Ps + row * BKV + c0);
    for (int j = 0; j < CPT / 4; ++j) {
      const float4 p = float4(fast::exp2(sv[4 * j] - mn), fast::exp2(sv[4 * j + 1] - mn),
                              fast::exp2(sv[4 * j + 2] - mn), fast::exp2(sv[4 * j + 3] - mn));
      bs += p.x + p.y + p.z + p.w;
      pw[j] = half4(p);
    }
    for (int o = 1; o < TPR; o <<= 1) bs += simd_shuffle_xor(bs, ushort(o));
    l = l * alpha + bs;
    if (tid % TPR == 0) rowbuf[row] = alpha;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint16_t i = 0; i < Oc.get_capacity(); ++i)
      if (Oc.is_valid_element(i)) Oc[i] *= rowbuf[Oc.get_multidimensional_index(i)[1]];
    auto tP = tensor(Ps, dextents<int, 2>{kr, BQ}, array<int, 2>{1, BKV});
    auto tV = tensor(Vp + k0 * vsl, dextents<int, 2>{D, kr}, array<int, 2>{1, vsl});
    pv.run(tP, tV, Oc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid % TPR == 0) rowbuf[row] = 1.0f / max(l, 1e-30f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device T* Op = O + (long(b) * Tn + q0) * H * D + h * D;
  for (uint16_t i = 0; i < Oc.get_capacity(); ++i) {
    if (!Oc.is_valid_element(i)) continue;
    auto idx = Oc.get_multidimensional_index(i);
    if (idx[1] < qr) Op[long(idx[1]) * H * D + idx[0]] = T(Oc[i] * rowbuf[idx[1]]);
  }
}

#define KERNELS(S, T) \
kernel void pool_##S(const device T* Q [[buffer(0)]], const device T* K [[buffer(1)]], const device T* V [[buffer(2)]], \
                     device float* qc [[buffer(3)]], device float* kc [[buffer(4)]], device float* vc [[buffer(5)]], \
                     constant long* P [[buffer(6)]], uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
  pool<T>(Q, K, V, qc, kc, vc, P, t / 32, lane); \
} \
kernel void sattn_##S(device T* Q [[buffer(0)]], device T* K [[buffer(1)]], device T* V [[buffer(2)]], device T* O [[buffer(3)]], \
                      constant long* P [[buffer(4)]], const device uchar* mask [[buffer(5)]], const device float* kmean [[buffer(6)]], \
                      const device float* pm [[buffer(7)]], const device float* pl [[buffer(8)]], const device float* po [[buffer(9)]], \
                      constant float& scale [[buffer(10)]], uint3 tgid [[threadgroup_position_in_grid]], \
                      uint tid [[thread_index_in_threadgroup]]) { \
  threadgroup float Ss[BQ * BKV]; \
  threadgroup half Ps[BQ * BKV]; \
  threadgroup float rowbuf[BQ]; \
  sattn<T>(Q, K, V, O, P, mask, kmean, pm, pl, po, scale, tgid, tid, Ss, Ps, rowbuf); \
}
KERNELS(bf16, bfloat)
KERNELS(f16, half)
"""

_SUFFIX = {torch.bfloat16: "bf16", torch.float16: "f16"}


def _route(qc, kc, vc, T, n, tau, log2s, sink_blocks, sink_q, topk_ratio, tail):
    """Block-level part of the reference: routing mask and the pooled tail of every query
    block. qc/kc/vc [BH, n, D] fp32 -> (mask uint8, kmean, pm, pl, po)."""
    dev = qc.device
    lengths = torch.full((n,), float(BLOCK), device=dev)
    if T - (n - 1) * BLOCK < BLOCK:
        lengths[-1] = float(T - (n - 1) * BLOCK)
    kmean = kc.mean(dim=1, keepdim=True)
    kcc = kc - kmean
    colmean = torch.bmm(qc, kcc.transpose(1, 2)) * log2s                    # [BH, nq, nk]
    s0, s1 = (sink_blocks or [0, 0])[:2]
    q0, q1 = (sink_q or [0, 0])[:2]
    idx = torch.arange(n, device=dev)
    if topk_ratio:
        ranked = colmean.clone()
        ranked[..., s0:s1] = float("-inf")
        n_sink = max(0, min(s1, n) - min(s0, n))
        kk = max(0, min(n - n_sink - 1, max(1, round(topk_ratio * (n - n_sink)))))
        exact = ranked >= ranked.topk(kk, dim=-1).values[..., -1:] if kk else torch.zeros_like(ranked, dtype=torch.bool)
    else:
        var = (qc.pow(2) * kcc.pow(2).mean(dim=1, keepdim=True)).sum(-1)    # [BH, nq]
        exact = colmean > (tau * torch.sqrt(var * log2s * log2s + 1e-6)).unsqueeze(-1)
    exact |= ((idx.view(1, -1) - idx.view(-1, 1)).abs() <= 1)
    exact |= ((idx >= s0) & (idx < s1)).view(1, 1, n)
    exact |= ((idx >= q0) & (idx < q1)).view(1, n, 1)
    if tail:
        logits = colmean.masked_fill(exact, float("-inf"))
        pm = logits.amax(dim=-1)
        p = torch.exp2(logits - pm.clamp_min(-1e30).unsqueeze(-1))
        pl = (p * lengths).sum(-1)
        po = torch.bmm(p, vc)
        pm = pm.clamp_min(-3.0e38)                                          # empty tail: -FLT_MAX, l = 0
    else:
        pm = torch.full(colmean.shape[:2], -3.0e38, device=dev)
        pl = torch.zeros_like(pm)
        po = torch.zeros(*colmean.shape[:2], qc.shape[-1], device=dev)
    return exact.to(torch.uint8).contiguous(), kmean.contiguous(), pm.contiguous(), pl.contiguous(), po.contiguous()


def _sol_attn_mps(q, k, v, tau, scale, sink_blocks, sink_q, topk_ratio, tail):
    B, T, H, D = q.shape
    n = (T + BLOCK - 1) // BLOCK
    scale = D ** -0.5 if scale is None else scale
    P = [T, H, n, q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
         v.stride(0), v.stride(1), v.stride(2)]
    sfx = _SUFFIX[q.dtype]
    qc, kc, vc = (torch.empty(B * H, n, D, device=q.device) for _ in range(3))
    getattr(_lib, "pool_" + sfx)(q, k, v, qc, kc, vc, P, threads=(B * H * n * 32, 1, 1), group_size=(256, 1, 1))
    mask, kmean, pm, pl, po = _route(qc, kc, vc, T, n, tau, scale * _LOG2E, sink_blocks, sink_q, topk_ratio, tail)
    out = torch.empty(B, T, H, D, device=q.device, dtype=q.dtype)
    getattr(_lib, "sattn_" + sfx)(q, k, v, out, P, mask, kmean, pm, pl, po, float(scale),
                                  threads=(-(-T // 32) * H * 128, 1, B), group_size=(128, 1, 1))
    return out


def _eligible(q, k, v, key_bias, block_len, coarse_gate):
    return (q.device.type == "mps" and q.dim() == 4 and q.shape[-1] == 128 and q.dtype in _SUFFIX
            and k.shape == q.shape and v.shape == q.shape and k.dtype == q.dtype and v.dtype == q.dtype
            and q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
            and key_bias is None and block_len is None and coarse_gate is None)


def sol_attn(q, k, v, tau=1.0, scale=None, sink_blocks=None, sink_q=None, key_bias=None, topk_ratio=0.0,
             tail=True, block_len=None, coarse_gate=None, token_aug=0):
    """comfy_kitchen.sol_attn on MPS (``token_aug`` is CUDA-only refinement; the reference ignores it too)."""
    from . import _caps
    if _eligible(q, k, v, key_bias, block_len, coarse_gate) and _caps.kernel_ready("sol_attn", _verify):
        return _sol_attn_mps(q, k, v, tau, scale, sink_blocks, sink_q, topk_ratio, tail)
    return _orig["sol_attn"](q, k, v, tau=tau, scale=scale, sink_blocks=sink_blocks, sink_q=sink_q,
                             key_bias=key_bias, topk_ratio=topk_ratio, tail=tail, block_len=block_len,
                             coarse_gate=coarse_gate, token_aug=token_aug)


def _verify():
    global _lib
    try:
        _lib = torch.mps.compile_shader(_MSL)
    except Exception as e:
        print(f"{TAG} shader did not compile; sparse attention stays dense: {e!r}")
        return False
    from comfy_kitchen.backends.eager.sol_attn import sol_attn as ref_sol_attn
    g = torch.Generator().manual_seed(0)
    errs = []
    # ragged length, sinks, both routing modes, a strided BHTD-transposed layout like the node passes
    for T, kw in ((700, dict(tau=0.7, sink_blocks=[0, 2], sink_q=[1, 3])),
                  (450, dict(topk_ratio=0.3, sink_blocks=[0, 1])),
                  (300, dict(tau=1.0, tail=False))):
        q, k, v = (torch.randn(1, 3, T, 128, generator=g).to(torch.bfloat16).transpose(1, 2) for _ in range(3))
        ref = ref_sol_attn(q.float(), k.float(), v.float(), **kw)
        got = _sol_attn_mps(q.to("mps"), k.to("mps"), v.to("mps"), kw.get("tau", 1.0), None, kw.get("sink_blocks"),
                            kw.get("sink_q"), kw.get("topk_ratio", 0.0), kw.get("tail", True)).cpu().float()
        errs.append(((got - ref).norm() / ref.norm()).item())
    if not max(errs) < 0.01:
        print(f"{TAG} self-check mismatch (rel errs {[f'{e:.3g}' for e in errs]}); sparse attention stays dense.")
        return False
    return True


def _patch_node():
    """The node keeps non-CUDA calls dense in _ineligible(); accept MPS there. Its module
    is loaded by ComfyUI under a path-derived name, so find it by its contents."""
    for mod in list(sys.modules.values()):
        d = getattr(mod, "__dict__", None)
        if not isinstance(d, dict) or "make_attention_override" not in d:
            continue
        f = d.get("_ineligible")
        if f is None or getattr(f, "_asfp8", False):
            continue

        def _ineligible(q, k, v, dim_head, _orig_f=f):
            if q.device.type == "mps":
                if dim_head != 128:
                    return f"head_dim {dim_head} != 128"
                if q.shape != k.shape or q.shape != v.shape:
                    return "cross-attention or GQA (kept dense)"
                if q.dtype not in (torch.bfloat16, torch.float16, torch.float32) or k.dtype != q.dtype or v.dtype != q.dtype:
                    return f"dtypes {q.dtype}/{k.dtype}/{v.dtype}"
                return None
            return _orig_f(q, k, v, dim_head)

        _ineligible._asfp8 = True
        mod._ineligible = _ineligible


def install():
    global _installed
    if _installed or sys.platform != "darwin":
        return
    from . import _caps
    if not _caps.resolve("ASFP8_SOL_ATTN", default_on=True, cap=_caps.shader_gate):
        return
    try:
        import comfy_kitchen as ck
    except Exception:
        return
    if not hasattr(ck, "sol_attn"):
        return
    _orig["sol_attn"] = ck.sol_attn
    ck.sol_attn = sol_attn
    _patch_node()
    from ._common import on_first_model_load
    on_first_model_load(_patch_node)
    _installed = True
    print(f"{TAG} Model Sparse Attention (Sol-Attn) runs on MPS shaders (self-check on first use).")
