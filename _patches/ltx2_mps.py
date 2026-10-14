"""Patch: single-pass shaders for LTX-2.x's elementwise work on MPS.

Transformer: LTX normalizes q and k across all heads (RMSNorm over the full inner dim),
then rotates each head with a per-token bf16 table; on MPS that is the fused norm kernel
plus comfy_kitchen's eager RoPE, ~7% of a 1280x720 step. One shader reads each q/k row
once, normalizes, rotates in fp32 and writes it back, and q/k/v of one input share one
int8 activation quantize.

Diffusion VAE decoder (2.4/2.5): q/k/v are projected into one buffer and a shader does the
per-head RMSNorm + 3-axis RoPE in place from the small per-axis cos/sin tables (the eager
path builds a per-token rotation table and copies q/k/v out first), na3d reads the
buffer as strided views; the AdaLN pre-norms go through the fused norm kernel and SwiGLU's
silu(gate) * up is one pass.

Masks, STG, training, LoRA-patched norm weights and other layouts/dtypes run the original
forwards. DEFAULT ON where compile_shader works; ASFP8_LTX2_FUSED=off disables.
"""

import sys

import torch

from .int8_linear_kernel_mps import shared_activation

TAG = "[AppleSilicon-int8-Fast/ltx2]"

_installed = False
_lib = None
_orig = {}

_MSL = r"""
#include <metal_stdlib>
using namespace metal;

constant constexpr uint TG = 128;

// x[row, C] in place: rmsnorm over C (weight w), then per head of 2 * half dims
// y_lo = f00 * lo + f01 * hi, y_hi = f10 * lo + f11 * hi with the token's [H, half, 2, 2]
// table f. A thread owns 4 consecutive pairs (quads) at a time. P = {C, half, N, table batch rows}
template <typename T>
inline void rms_rope_row(device T* x, const device T* w, const device T* f, constant long* P, float eps,
                         uint row, uint tid, uint sg, uint lane, threadgroup float* part) {
  const uint C = P[0], half_ = P[1], N = P[2], quads = C / 8;
  x += ulong(row) * C;
  f += (ulong(row / N) * P[3] + row % N) * C * 2;
  float4 lo[8], hi[8];
  float ss = 0.0f;
  for (uint j = 0; j < 8; ++j) {
    const uint q = tid + j * TG;
    if (q >= quads) break;
    const uint p = q * 4, e = p / half_ * 2 * half_ + p % half_;
    lo[j] = float4(*(device vec<T, 4>*)(x + e));
    hi[j] = float4(*(device vec<T, 4>*)(x + e + half_));
    ss += dot(lo[j], lo[j]) + dot(hi[j], hi[j]);
  }
  ss = simd_sum(ss);
  if (lane == 0) part[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = rsqrt((part[0] + part[1] + part[2] + part[3]) / float(C) + eps);
  for (uint j = 0; j < 8; ++j) {
    const uint q = tid + j * TG;
    if (q >= quads) break;
    const uint p = q * 4, e = p / half_ * 2 * half_ + p % half_;
    const float4 a = lo[j] * inv * float4(*(const device vec<T, 4>*)(w + e));
    const float4 b = hi[j] * inv * float4(*(const device vec<T, 4>*)(w + e + half_));
    const device vec<T, 4>* fr = (const device vec<T, 4>*)(f + p * 4);
    float4 ylo, yhi;
    for (uint c = 0; c < 4; ++c) {
      const float4 m = float4(fr[c]);
      ylo[c] = m.x * a[c] + m.y * b[c];
      yhi[c] = m.z * a[c] + m.w * b[c];
    }
    *(device vec<T, 4>*)(x + e) = vec<T, 4>(ylo);
    *(device vec<T, 4>*)(x + e + half_) = vec<T, 4>(yhi);
  }
}

// VAE decoder: q and k heads of 64 inside qkv[row, 3 * NH * 64], one simdgroup each with lane
// j holding interleaved pair j: rmsnorm over the head (weight qw / kw), then rotate the pair
// by its axis' angle (j < pt: frame, < pt + ph: row, else column) from the per-axis tables
// cs / sn = [T * pt | H * ph | W * pw]. P = {T, H, W, NH, pt, ph, pw, row stride}
template <typename T>
inline void na_rms_rope(device T* qkv, const device T* qw, const device T* kw, const device float* cs,
                        const device float* sn, constant long* P, float eps, uint gid, uint lane) {
  const uint Tn = P[0], Hn = P[1], Wn = P[2], NH = P[3], pt = P[4], ph = P[5];
  const uint s = gid / 32, which = s % 2, head = s / 2 % NH, row = s / 2 / NH;
  const uint wi = row % Wn, hi = row / Wn % Hn, ti = row / (Wn * Hn) % Tn;
  device vec<T, 2>* x = (device vec<T, 2>*)(qkv + ulong(row) * P[7] + (which * NH + head) * 64) + lane;
  const device vec<T, 2>* w = (const device vec<T, 2>*)(which ? kw : qw) + lane;
  float2 v = float2(*x);
  v *= rsqrt(simd_sum(dot(v, v)) / 64.0f + eps) * float2(*w);
  const uint t = lane < pt ? ti * pt + lane
               : lane < pt + ph ? Tn * pt + hi * ph + lane - pt
               : Tn * pt + Hn * ph + wi * P[6] + lane - pt - ph;
  const float c = cs[t], sv = sn[t];
  *x = vec<T, 2>(float2(c * v.x - sv * v.y, sv * v.x + c * v.y));
}

// g = silu(g) * u, in place
template <typename T>
inline void silu_mul(device T* g, const device T* u, uint i) {
  const float4 a = float4(((device vec<T, 4>*)g)[i]);
  ((device vec<T, 4>*)g)[i] = vec<T, 4>(a / (1.0f + exp(-a)) * float4(((const device vec<T, 4>*)u)[i]));
}

#define KERNELS(S, T) \
kernel void rms_rope_##S(device T* x [[buffer(0)]], const device T* w [[buffer(1)]], const device T* f [[buffer(2)]], \
                         constant long* P [[buffer(3)]], constant float& eps [[buffer(4)]], \
                         uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  threadgroup float part[TG / 32]; \
  rms_rope_row<T>(x, w, f, P, eps, row, tid, sg, lane, part); \
} \
kernel void na_rms_rope_##S(device T* qkv [[buffer(0)]], const device T* qw [[buffer(1)]], const device T* kw [[buffer(2)]], \
                            const device float* cs [[buffer(3)]], const device float* sn [[buffer(4)]], \
                            constant long* P [[buffer(5)]], constant float& eps [[buffer(6)]], \
                            uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
  na_rms_rope<T>(qkv, qw, kw, cs, sn, P, eps, gid, lane); \
} \
kernel void silu_mul_##S(device T* g [[buffer(0)]], const device T* u [[buffer(1)]], uint i [[thread_position_in_grid]]) { \
  silu_mul<T>(g, u, i); \
}
KERNELS(bf16, bfloat)
KERNELS(f16, half)
"""

_SUFFIX = {torch.bfloat16: "bf16", torch.float16: "f16"}


def _fn(name, dtype):
    return getattr(_lib, f"{name}_{_SUFFIX[dtype]}")


def _ready():
    from . import _caps
    return _caps.kernel_ready("ltx2", _verify)


# --- transformer attention ---

def _norm_ok(norm, dim):
    return (norm.weight is not None and tuple(norm.normalized_shape) == (dim,)
            and not len(getattr(norm, "weight_function", ())) and not len(getattr(norm, "bias_function", ())))


def _rope_ok(x, freqs):
    if freqs is None:
        return False
    rot, split = freqs
    B, N, C = x.shape
    return (split and rot.dtype == x.dtype and rot.is_contiguous() and rot.dim() == 6 and rot.shape[0] in (1, B)
            and rot.shape[1] == N and rot.shape[2] * rot.shape[3] * 2 == C and rot.shape[3] % 4 == 0
            and C % 8 == 0 and C <= 8 * 128 * 8)


def _rms_rope_(x, norm, freqs):
    import comfy.model_management
    rot = freqs[0]
    B, N, C = x.shape
    w = comfy.model_management.cast_to(norm.weight, dtype=x.dtype, device=x.device)
    _fn("rms_rope", x.dtype)(
        x, w, rot, [C, rot.shape[3], N, 0 if rot.shape[0] == 1 else N], float(norm.eps if norm.eps is not None else torch.finfo(x.dtype).eps),
        threads=(B * N * 128, 1, 1), group_size=(128, 1, 1))
    return x


def _attention_forward(self, x, context=None, mask=None, pe=None, k_pe=None, transformer_options={}):
    import comfy.ldm.modules.attention
    import comfy.model_management
    from comfy.ldm.lightricks.model import GuideAttentionMask, _attention_with_guide_mask
    from comfy.ldm.modules.attention import AttentionTensorContainer

    self_attn = context is None
    if (x.device.type != "mps" or x.dtype not in _SUFFIX or comfy.model_management.in_training or pe is None
            or (self_attn and transformer_options.get("stg_skip_self_attn", False))
            or not (_norm_ok(self.q_norm, self.heads * self.dim_head) and _norm_ok(self.k_norm, self.heads * self.dim_head))
            or not _ready()):
        return _orig["attention"](self, x, context, mask, pe, k_pe, transformer_options)
    context = x if context is None else context
    with shared_activation(context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
    k_freqs = pe if k_pe is None else k_pe
    if not (q.is_contiguous() and k.is_contiguous() and _rope_ok(q, pe) and _rope_ok(k, k_freqs)):
        # same math as the stock forward, minus the q/k/v projections already done
        from comfy.ldm.lightricks.model import apply_rotary_emb, apply_rotary_emb_qk
        q, k = self.q_norm(q), self.k_norm(k)
        if k_pe is None and q.shape == k.shape:
            q, k = apply_rotary_emb_qk(q, k, pe)
        else:
            q, k = apply_rotary_emb(q, pe), apply_rotary_emb(k, k_freqs)
    else:
        _rms_rope_(q, self.q_norm, pe)
        _rms_rope_(k, self.k_norm, k_freqs)

    if isinstance(mask, GuideAttentionMask):
        out = _attention_with_guide_mask(q, k, v, self.heads, mask, attn_precision=self.attn_precision, transformer_options=transformer_options, preferred_attention=self.comfy_attention)
    else:
        q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
        out = comfy.ldm.modules.attention.optimized_attention(q, k, v, self.heads, mask=mask, attn_precision=self.attn_precision, transformer_options=transformer_options, preferred_attention=self.comfy_attention)

    if self.to_gate_logits is not None:
        gate_logits = self.to_gate_logits(x)
        b, t, _ = out.shape
        out = out.view(b, t, self.heads, self.dim_head) * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
        out = out.view(b, t, self.heads * self.dim_head)
    return self.to_out(out)


# --- diffusion VAE decoder ---

def _na_rms_rope_(qkv, num_heads, q_weight, k_weight, tables, split):
    """In place on the q/k parts of qkv [B, T, H, W, 3 * NH * 64], eps as comfy_kitchen.rms_rope_'s default."""
    B, T, H, W, _ = qkv.shape
    cs = torch.cat([c.reshape(-1) for c, _ in tables])
    sn = torch.cat([s.reshape(-1) for _, s in tables])
    _fn("na_rms_rope", qkv.dtype)(
        qkv, q_weight, k_weight, cs, sn, [T, H, W, num_heads, *(d // 2 for d in split), qkv.shape[-1]], 1e-6,
        threads=(B * T * H * W * num_heads * 2 * 32, 1, 1), group_size=(256, 1, 1))
    return qkv


def _na_attention_forward(self, x, pre=None, add_to=None):
    import comfy_kitchen
    import comfy.ldm.lightricks.vae.na_diffusion_decoder as nd

    if not (x.device.type == "mps" and x.dtype in _SUFFIX and self.head_dim == 64 and sum(self.rope_split) == 64
            and _ready()):
        return _orig["na_attention"](self, x, pre, add_to)
    batch, t, h, w, _ = x.shape
    inv_freqs = tuple(nd.rope_inv_freqs(d, self.rope_base, device=x.device) for d in self.rope_split)
    tables = nd._rope_tables((t, h, w), inv_freqs, x.device)
    qkv = torch.empty(batch, t, h, w, 3 * self.dim, dtype=x.dtype, device=x.device)
    chunk = max(1, (2 ** 25) // max(h * w * self.dim, 1))
    for t0 in range(0, t, chunk):
        sl = x[:, t0:t0 + chunk] if pre is None else pre(x[:, t0:t0 + chunk])
        qkv[:, t0:t0 + chunk] = self.qkv(sl)
    q_weight = (self.q_norm.weight.detach() * self.scale).to(x.dtype)  # scale commutes with the rotation
    k_weight = self.k_norm.weight.detach().to(x.dtype)
    _na_rms_rope_(qkv, self.num_heads, q_weight, k_weight, tables, self.rope_split)
    q, k, v = qkv.view(batch, t, h, w, 3, self.num_heads, self.head_dim).unbind(4)
    out = comfy_kitchen.na3d(q, k, v, list(self.kernel_size), None, 1.0)
    del q, k, v, qkv
    out = out.reshape(batch, t, h, w, self.dim)
    res = add_to if add_to is not None else torch.empty_like(out)
    for t0 in range(0, t, chunk):
        t1 = min(t0 + chunk, t)
        if add_to is not None:
            res[:, t0:t1] += self.proj(out[:, t0:t1])
        else:
            res[:, t0:t1] = self.proj(out[:, t0:t1])
    return res


def _silu_mul(g, u):
    """silu(g) * u into g; g and u fresh contiguous projections."""
    if not (g.device.type == "mps" and g.dtype in _SUFFIX and g.is_contiguous() and u.is_contiguous()
            and g.shape == u.shape and u.dtype == g.dtype and g.numel() % 4 == 0 and _ready()):
        return torch.nn.functional.silu(g) * u
    _fn("silu_mul", g.dtype)(g, u, threads=(g.numel() // 4, 1, 1))
    return g


def _swiglu_forward(self, x, pre=None, add_to=None):
    import comfy.ldm.lightricks.vae.na_diffusion_decoder as nd

    _, t, h, w, _ = x.shape
    chunk = max(1, nd.MLP_TOKEN_CHUNK // max(h * w, 1))
    out = add_to if add_to is not None else torch.empty_like(x)
    for t0 in range(0, t, chunk):
        t1 = min(t0 + chunk, t)
        sl = x[:, t0:t1] if pre is None else pre(x[:, t0:t1])
        y = self.w_down(_silu_mul(self.w_gate(sl), self.w_up(sl)))
        if add_to is not None:
            out[:, t0:t1] += y
        else:
            out[:, t0:t1] = y
    return out


def _norm_modulate(s, norm, scale, shift):
    """modulate(norm(s), scale, shift) for per-batch scale/shift [B, 1, 1, 1, C], in one pass."""
    from . import fused_norm_mps
    if s.device.type == "mps" and s.dtype in _SUFFIX and scale.dtype == s.dtype and shift.dtype == s.dtype:
        return fused_norm_mps.fused_rmsnorm_modulate(s, norm.weight.to(s.dtype), norm.eps, scale, shift)
    import comfy.ldm.lightricks.vae.na_diffusion_decoder as nd
    return nd.modulate(norm(s), scale, shift)


def _diffusion_block_forward(self, x, latent_context, modulation):
    import comfy.ldm.lightricks.vae.na_diffusion_decoder as nd

    scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
        modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(nd.AdaLNZero.NUM_CHUNKS)
    ]
    chunk = max(1, nd.MLP_TOKEN_CHUNK // max(x.shape[2] * x.shape[3], 1))
    for t0 in range(0, x.shape[1], chunk):
        x[:, t0:t0 + chunk] += self.context_proj(latent_context[:, t0:t0 + chunk])
    x = self.attn(x, pre=lambda s: _norm_modulate(s, self.norm1, scale_msa, shift_msa), add_to=x)
    return self.mlp(x, pre=lambda s: _norm_modulate(s, self.norm2, scale_mlp, shift_mlp), add_to=x)


def _verify():
    global _lib
    try:
        _lib = torch.mps.compile_shader(_MSL)
    except Exception as e:
        print(f"{TAG} shader did not compile; using the stock LTX ops: {e!r}")
        return False
    import comfy_kitchen.backends.eager.rope as er
    import comfy.ldm.lightricks.vae.na_diffusion_decoder as nd
    g = torch.Generator().manual_seed(0)
    errs = []
    # transformer q/k: one shared table (video, D 128), and a per-batch table (D 64)
    for B, Bt, N, H, Dh in ((1, 1, 37, 4, 128), (2, 2, 19, 3, 64)):
        x = (torch.randn(B, N, H * Dh, generator=g) * 3).to(torch.bfloat16)
        w = torch.randn(H * Dh, generator=g).to(torch.bfloat16)
        ang = torch.rand(Bt, N, H, Dh // 2, generator=g) * 6
        rot = torch.stack([ang.cos(), -ang.sin(), ang.sin(), ang.cos()], -1).reshape(Bt, N, H, Dh // 2, 2, 2).to(torch.bfloat16)
        xn = torch.nn.functional.rms_norm(x.float(), (H * Dh,), w.float(), 1e-5)
        ref = er.apply_rope_split_half1(xn.reshape(B, N, H, Dh), rot.float()).reshape(B, N, H * Dh)
        norm = torch.nn.RMSNorm(H * Dh, eps=1e-5, device="mps", dtype=torch.bfloat16)
        norm.weight.data.copy_(w)
        out = _rms_rope_(x.to("mps"), norm, (rot.to("mps"), True))
        errs.append(((out.cpu().float() - ref).norm() / ref.norm()).item())
    # VAE decoder q/k inside a qkv buffer, against comfy_kitchen's rms_rope_ on the per-token table
    B, T, H, W, NH = 2, 3, 5, 6, 2
    split = nd.default_rope_dim_split(64)
    qkv = (torch.randn(B, T, H, W, 3 * NH * 64, generator=g) * 2).to(torch.bfloat16)
    qw, kw = (torch.randn(64, generator=g).to(torch.bfloat16) for _ in range(2))
    tables = nd._rope_tables((T, H, W), tuple(nd.rope_inv_freqs(d, device="cpu") for d in split), "cpu")
    freqs = nd._rope_matrices_slice(tables, 0, T, H, W)
    ref = qkv.float().view(B, T * H * W, 3, NH, 64).clone()
    for b in range(B):
        er.rms_rope_(ref[b:b + 1, :, 0], ref[b:b + 1, :, 1], freqs, qw.float(), kw.float())
    out = _na_rms_rope_(qkv.to("mps"), NH, qw.to("mps"), kw.to("mps"), [(c.to("mps"), s.to("mps")) for c, s in tables], split)
    errs.append(((out.cpu().float().view_as(ref) - ref).norm() / ref.norm()).item())
    gt, up = (torch.randn(7, 64, generator=g).to(torch.bfloat16).to("mps") for _ in range(2))
    ref = torch.nn.functional.silu(gt.float()) * up.float()
    _fn("silu_mul", gt.dtype)(gt, up, threads=(gt.numel() // 4, 1, 1))
    errs.append(((gt.cpu().float() - ref.cpu()).norm() / ref.norm().cpu()).item())
    if not max(errs) < 0.01:
        print(f"{TAG} self-check mismatch (rel errs {[f'{e:.3g}' for e in errs]}); using the stock LTX ops.")
        return False
    return True


def install():
    global _installed
    if _installed or sys.platform != "darwin":
        return
    from . import _caps
    if not _caps.resolve("ASFP8_LTX2_FUSED", default_on=True, cap=_caps.has_compile_shader):
        return
    try:
        import comfy.ldm.lightricks.model as lm
        import comfy.ldm.lightricks.vae.na_diffusion_decoder as nd
    except Exception:
        return
    _orig["attention"] = lm.CrossAttention.forward
    _orig["na_attention"] = nd.NeighborhoodAttention3D.forward
    lm.CrossAttention.forward = _attention_forward
    nd.NeighborhoodAttention3D.forward = _na_attention_forward
    nd.SwiGLU.forward = _swiglu_forward
    nd.DiffusionNABlock.forward = _diffusion_block_forward
    _installed = True
    print(f"{TAG} LTX-2.x attention q/k RMSNorm + RoPE and the diffusion VAE decoder's norm/RoPE/SwiGLU "
          f"fused on MPS (self-check on first use).")
