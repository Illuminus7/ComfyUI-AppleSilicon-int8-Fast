"""Patch: run comfy_kitchen INT8 (convrot) Linears as W8A8 on the M5 Neural Accelerators.

comfy treats int8 as a disabled format on MPS and dequantizes (and un-rotates) the whole
weight on every call. Instead one shader rotates + per-row quantizes the activation, and
an int8 x int8 matmul2d GEMM applies both scales and the bias in its epilogue (int8 runs
at twice the bf16 rate on M5). DEFAULT ON where the shader builds and passes its
self-check; ASFP8_INT8_EXT=off disables.
"""

import contextlib
import sys

import torch

TAG = "[AppleSilicon-int8-Fast/int8_kernel]"

_installed = False
_lib = None

_MSL = r"""
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// One simdgroup per row, two passes over its 256-wide groups (rotate + absmax, then rotate +
// quantize; the second read hits cache). Lane t holds group elements t*8..t*8+7. ConvRot's
// Hadamard is H4 (x) H4 (x) H4 (x) H4 / 16, and since H4 = J - 2*antidiag every base-4 digit
// transforms as y_k = sum - 2*x_{3-k}.
inline void h4_lanes(thread float (&v)[8], ushort m1, ushort m2, ushort m3) {
  for (short j = 0; j < 8; ++j) {
    float a = simd_shuffle_xor(v[j], m1), b = simd_shuffle_xor(v[j], m2), c = simd_shuffle_xor(v[j], m3);
    v[j] = v[j] + a + b - c;
  }
}

inline void hadamard256(thread float (&r)[8]) {
  for (short h = 0; h < 8; h += 4) {
    float a0 = r[h], a1 = r[h + 1], a2 = r[h + 2], a3 = r[h + 3], s = a0 + a1 + a2 + a3;
    r[h] = s - 2.0f * a3; r[h + 1] = s - 2.0f * a2; r[h + 2] = s - 2.0f * a1; r[h + 3] = s - 2.0f * a0;
  }
  float p[8], o[8];
  for (short j = 0; j < 8; ++j) { p[j] = simd_shuffle_xor(r[j], ushort(1)); o[j] = r[j]; }
  for (short j = 0; j < 8; ++j) r[j] = o[j] + o[j ^ 4] + p[j] - p[j ^ 4];
  h4_lanes(r, 2, 4, 6);
  h4_lanes(r, 8, 16, 24);
  for (short j = 0; j < 8; ++j) r[j] *= 0.0625f;
}

template <typename T>
inline void load_group(const device T* x, uint lane, uint rotate, thread float (&v)[8]) {
  const device vec<T, 4>* xv = (const device vec<T, 4>*)(x + lane * 8);
  const float4 a = float4(xv[0]), b = float4(xv[1]);
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
  if (rotate) hadamard256(v);
}

#define ROTQ(NAME, T) \
kernel void NAME(const device T* x [[buffer(0)]], device char* q [[buffer(1)]], device float* scale [[buffer(2)]], \
                 constant uint& M [[buffer(3)]], constant uint& K [[buffer(4)]], constant uint& rotate [[buffer(5)]], \
                 uint t [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) { \
  const uint row = t / 32; \
  if (row >= M) return; \
  x += ulong(row) * K; \
  q += ulong(row) * K; \
  float v[8]; \
  float amax = 0.0f; \
  for (uint g = 0; g < K; g += 256) { \
    load_group<T>(x + g, lane, rotate, v); \
    for (short j = 0; j < 8; ++j) amax = max(amax, fabs(v[j])); \
  } \
  amax = simd_max(amax); \
  const float s = max(amax / 127.0f, 1e-30f), inv = 1.0f / s; \
  if (lane == 0) scale[row] = s; \
  for (uint g = 0; g < K; g += 256) { \
    load_group<T>(x + g, lane, rotate, v); \
    device char4* qv = (device char4*)(q + g + lane * 8); \
    qv[0] = char4(clamp(rint(float4(v[0], v[1], v[2], v[3]) * inv), -127.0f, 127.0f)); \
    qv[1] = char4(clamp(rint(float4(v[4], v[5], v[6], v[7]) * inv), -127.0f, 127.0f)); \
  } \
}
ROTQ(rotq_bf16, bfloat)
ROTQ(rotq_f16, half)
ROTQ(rotq_f32, float)

// D[M,N] = (A[M,K] @ B[N,K]^T) * a_scale[m] * w_scale[n or 0] (+ bias[n]), int8 operands, int32 acc.
// Tiles run in bands of GM row tiles walking N, so threadgroups in flight share B (weight)
// tiles in cache instead of streaming the whole weight once per row tile: 1.3-2x at 6k-16k rows.
// Full tiles run in their own kernel with fixed-extent slices, compile-time scale layout and bias,
// and an inline store: ~1.2x at 4k-10k rows, but only all together (a bounded path, runtime flags
// or an out-of-line store each cost it back). The partial tiles take i8mme. K is a multiple of 256
// (the rotate/quantize group), so the k-steps are always whole.
constant constexpr int BM = 64, BN = 128, BK = 128, NSG = 4, GM = 32;

template <typename OutT, bool PER_N, bool BIAS, typename C>
inline void i8mm_store(thread C& c, device OutT* D, const device float* a_scale, const device float* w_scale,
                       const device OutT* bias, int m0, int n0, int mr, int nr, int N) {
  #pragma clang loop unroll(full)
  for (uint16_t i = 0; i < c.get_capacity(); ++i) {
    if (!c.is_valid_element(i)) continue;
    auto idx = c.get_multidimensional_index(i);
    const int r = int(idx[1]), cn = int(idx[0]), n = n0 + cn;
    if (r >= mr || cn >= nr) continue;
    float y = float(c[i]) * a_scale[m0 + r] * w_scale[PER_N ? n : 0];
    if (BIAS) y += float(bias[n]);
    D[ulong(m0 + r) * N + n] = OutT(y);
  }
}

template <typename OutT, bool PER_N, bool BIAS>
inline void i8mm(device int8_t* A, device int8_t* B, device OutT* D, const device float* a_scale,
                 const device float* w_scale, const device OutT* bias, uint M_, uint N_, uint K_, uint3 tgid) {
  const int M = M_, N = N_, K = K_;
  const int tiles_m = M / BM, tiles_n = N / BN;
  const int per = GM * tiles_n, first = int(tgid.x) / per * GM, t = int(tgid.x) % per, gsz = min(GM, tiles_m - first);
  const int n0 = t / gsz * BN, m0 = (first + t % gsz) * BM;
  constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  auto tA = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(A + ulong(m0) * K, dextents<int32_t, 2>(K, BM));
  auto tB = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>(B + ulong(n0) * K, dextents<int32_t, 2>(K, BN));
  auto a0 = tA.slice<BK, BM>(0, 0);
  auto b0 = tB.slice<BK, BN>(0, 0);
  auto c = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), int>();
  #pragma clang loop unroll(full)
  for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    auto a = tA.slice<BK, BM>(k0, 0);
    auto b = tB.slice<BK, BN>(k0, 0);
    op.run(a, b, c);
  }
  // the store stays inline with one base pointer: through i8mm_store it costs this kernel ~20%
  device OutT* Dp = D + ulong(m0) * N + n0;
  #pragma clang loop unroll(full)
  for (uint16_t i = 0; i < c.get_capacity(); ++i) {
    if (c.is_valid_element(i)) {
      auto idx = c.get_multidimensional_index(i);
      float y = float(c[i]) * a_scale[m0 + idx[1]] * w_scale[PER_N ? n0 + idx[0] : 0];
      if (BIAS) y += float(bias[n0 + idx[0]]);
      Dp[ulong(idx[1]) * N + idx[0]] = OutT(y);
    }
  }
}

// the tiles i8mm leaves: the partial last row band, then the partial last column
template <typename OutT, bool PER_N, bool BIAS>
inline void i8mme(device int8_t* A, device int8_t* B, device OutT* D, const device float* a_scale,
                  const device float* w_scale, const device OutT* bias, uint M_, uint N_, uint K_, uint3 tgid) {
  const int M = M_, N = N_, K = K_;
  const int band = M % BM ? (N + BN - 1) / BN : 0, e = int(tgid.x);
  const int m0 = e < band ? M / BM * BM : (e - band) * BM;
  const int n0 = e < band ? e * BN : N / BN * BN;
  constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<NSG>> op;
  const int mr = min(BM, M - m0), nr = min(BN, N - n0);
  device int8_t* Ap = A + ulong(m0) * K;
  device int8_t* Bp = B + ulong(n0) * K;
  auto tA0 = tensor(Ap, dextents<int, 2>{min(BK, K), mr}, array<int, 2>{1, K});
  auto tB0 = tensor(Bp, dextents<int, 2>{min(BK, K), nr}, array<int, 2>{1, K});
  using AT = __tensor_ops_detail::__remove_addrspace_t<decltype(tA0)>;
  using BT = __tensor_ops_detail::__remove_addrspace_t<decltype(tB0)>;
  auto c = op.get_destination_cooperative_tensor<AT, BT, int>();
  #pragma clang loop unroll(full)
  for (uint16_t i = 0; i < c.get_capacity(); ++i) if (c.is_valid_element(i)) c[i] = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    const int kk = min(BK, K - k0);
    auto tA = tensor(Ap + k0, dextents<int, 2>{kk, mr}, array<int, 2>{1, K});
    auto tB = tensor(Bp + k0, dextents<int, 2>{kk, nr}, array<int, 2>{1, K});
    op.run(tA, tB, c);
  }
  i8mm_store<OutT, PER_N, BIAS>(c, D, a_scale, w_scale, bias, m0, n0, mr, nr, N);
}

// i8mm[e]_<dtype>_<n: per-output-channel weight scale | t: per-tensor>[b: bias]
#define I8MM_ONE(NAME, T, PER_N, BIAS) \
kernel void i8mm_##NAME(device int8_t* A [[buffer(0)]], device int8_t* B [[buffer(1)]], device T* D [[buffer(2)]], \
                 const device float* a_scale [[buffer(3)]], const device float* w_scale [[buffer(4)]], \
                 const device T* bias [[buffer(5)]], constant uint& M [[buffer(6)]], constant uint& N [[buffer(7)]], \
                 constant uint& K [[buffer(8)]], uint3 tgid [[threadgroup_position_in_grid]]) { \
  i8mm<T, PER_N, BIAS>(A, B, D, a_scale, w_scale, bias, M, N, K, tgid); \
} \
kernel void i8mme_##NAME(device int8_t* A [[buffer(0)]], device int8_t* B [[buffer(1)]], device T* D [[buffer(2)]], \
                 const device float* a_scale [[buffer(3)]], const device float* w_scale [[buffer(4)]], \
                 const device T* bias [[buffer(5)]], constant uint& M [[buffer(6)]], constant uint& N [[buffer(7)]], \
                 constant uint& K [[buffer(8)]], uint3 tgid [[threadgroup_position_in_grid]]) { \
  i8mme<T, PER_N, BIAS>(A, B, D, a_scale, w_scale, bias, M, N, K, tgid); \
}
#define I8MM(S, T) \
I8MM_ONE(S##_n, T, true, false) I8MM_ONE(S##_nb, T, true, true) \
I8MM_ONE(S##_t, T, false, false) I8MM_ONE(S##_tb, T, false, true)
I8MM(bf16, bfloat)
I8MM(f16, half)
I8MM(f32, float)
"""

_SUFFIX = {torch.bfloat16: "bf16", torch.float16: "f16", torch.float32: "f32"}
def _quantize_activation(x2d, rotate):
    """Rotate (ConvRot Hadamard, group 256) and per-row int8-quantize, in one pass."""
    M, K = x2d.shape
    q = torch.empty(M, K, device=x2d.device, dtype=torch.int8)
    scale = torch.empty(M, device=x2d.device, dtype=torch.float32)
    getattr(_lib, "rotq_" + _SUFFIX[x2d.dtype])(
        x2d, q, scale, M, K, int(rotate), threads=(M * 32, 1, 1), group_size=(256, 1, 1))
    return q, scale


def _int8_matmul(a8, a_scale, w8, w_scale, bias, out_dtype):
    M, K = a8.shape
    N = w8.shape[0]
    out = torch.empty(M, N, device=a8.device, dtype=out_dtype)
    name = f"{_SUFFIX[out_dtype]}_{'n' if w_scale.numel() > 1 else 't'}{'' if bias is None else 'b'}"
    args = (a8, w8, out, a_scale, w_scale, out if bias is None else bias, M, N, K)
    full_m, full_n = M // 64, N // 128
    if full_m and full_n:
        getattr(_lib, "i8mm_" + name)(*args, threads=(full_m * full_n * 128, 1, 1), group_size=(128, 1, 1))
    edge = (-(-N // 128) if M % 64 else 0) + (full_m if N % 128 else 0)
    if edge:
        getattr(_lib, "i8mme_" + name)(*args, threads=(edge * 128, 1, 1), group_size=(128, 1, 1))
    return out


_shared = None   # [x, convrot, q, scale] inside a shared_activation(x) block


@contextlib.contextmanager
def shared_activation(x):
    """The caller promises x is not modified inside this block, so every int8 projection of
    it (q/k/v of one input) shares one rotate + quantize."""
    global _shared
    _shared = [x, None, None, None]
    try:
        yield
    finally:
        _shared = None


def _int8_linear_kernel(x, weight, weight_scale, bias, convrot):
    s = _shared
    if s is not None and s[0] is x:
        if s[2] is None or s[1] != convrot:
            s[1:] = convrot, *_quantize_activation(x.reshape(-1, x.shape[-1]).contiguous(), convrot)
        q, scale = s[2], s[3]
    else:
        q, scale = _quantize_activation(x.reshape(-1, x.shape[-1]).contiguous(), convrot)
    if bias is not None and bias.dtype != x.dtype:
        bias = bias.to(x.dtype)
    out = _int8_matmul(q, scale, weight, weight_scale.reshape(-1).float(), bias, x.dtype)
    return out.reshape(*x.shape[:-1], weight.shape[0])


def _verify():
    """The contract _caps.kernel_ready expects: build the library, then check numerics
    (rotation and GEMM, remainder tiles included) against a float reference."""
    global _lib
    try:
        _lib = torch.mps.compile_shader(_MSL)
    except Exception as e:
        print(f"{TAG} int8 shader did not compile; using comfy's int8 path: {e!r}")
        return False
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation
    g = torch.Generator().manual_seed(0)
    x = torch.randn(70, 512, generator=g).to(torch.bfloat16)
    w = torch.randint(-127, 128, (200, 512), generator=g, dtype=torch.int8)
    xr = _rotate_activation(x.float(), _build_hadamard(256, dtype=torch.float32), 256)
    ref = xr @ (w.float() * 0.01).t()
    q, scale = _quantize_activation(x.to("mps"), True)
    out = _int8_matmul(q, scale, w.to("mps"), torch.tensor([0.01], device="mps"), None, torch.float32).cpu()
    err = ((out - ref).norm() / ref.norm()).item()
    if not err < 0.02:
        print(f"{TAG} self-check mismatch (rel err {err:.3g}); using comfy's int8 path.")
        return False
    return True


def _try_int8_kernel_forward(self, input):
    """Return the layer output via the kernel W8A8 path, or None to fall back.

    The shader is compiled only after every cheap eligibility gate passes, so a model
    with no int8 layers never pays for it.
    """
    try:
        from comfy_kitchen.tensor import QuantizedTensor

        if not isinstance(input, torch.Tensor) or input.device.type != "mps":
            return None
        if isinstance(input, QuantizedTensor) or input.dtype not in _SUFFIX:
            return None
        if input.numel() == 0 or input.shape[-1] % 256:
            return None

        w = self.weight
        if not isinstance(w, QuantizedTensor) or getattr(w, "_layout_cls", None) != "TensorWiseINT8Layout":
            return None
        # offloaded weight: transient, so let comfy's forward cast it in rather than
        # latch mark_kernel_failed for the session
        if w.device.type != "mps":
            return None
        # only the checkpoint's own full_precision_matrix_mult opts out: comfy also sets
        # _full_precision_mm for every int8 layer on MPS (format "disabled" there), which
        # is exactly the case this kernel exists for
        if getattr(self, "_full_precision_mm_config", False):
            return None
        # deliberately no comfy_force_cast_weights check: on a quantized weight that
        # cast IS the per-call dequant/un-rotation this path exists to bypass
        if len(getattr(self, "weight_function", [])) or len(getattr(self, "bias_function", [])):
            return None

        params = w._params
        if getattr(params, "transposed", False):
            return None
        convrot = getattr(params, "convrot", False)
        if convrot and getattr(params, "convrot_groupsize", 256) != 256:
            return None

        from . import _caps
        if not _caps.kernel_ready("int8", _verify):
            return None

        return _int8_linear_kernel(input, w._qdata, params.scale, self.bias, convrot)
    except Exception as e:
        # latch off: verification already passed, so this repeats on every later layer
        from . import _caps
        if _caps._kernel_ready.get("int8"):
            print(f"{TAG} kernel forward failed ({e!r}); using comfy's int8 path "
                  f"for the rest of this session.")
        _caps.mark_kernel_failed("int8")
        return None


def _kernel_owns(linear, x):
    """Would this layer's forward take the kernel? Cheap checks first; the build last."""
    from comfy_kitchen.tensor import QuantizedTensor
    w = getattr(linear, "weight", None)
    if not (isinstance(w, QuantizedTensor) and getattr(w, "_layout_cls", None) == "TensorWiseINT8Layout"
            and w.device.type == "mps" and isinstance(x, torch.Tensor) and x.device.type == "mps"
            and x.dtype in _SUFFIX and not getattr(linear, "_full_precision_mm_config", False)):
        return False
    from . import _caps
    return _caps.kernel_ready("int8", _verify)


_DEQUANT_MODE = None


def _dequant_enabled():
    """Opt-in gate for once-at-load weight dequant, plus a >= 48 GiB total-RAM check.

    Opt-in because the plain copy lands after comfy's model_management has budgeted
    around the loaded int8 size.
    """
    global _DEQUANT_MODE
    if _DEQUANT_MODE is None:
        import os
        v = os.environ.get("ASFP8_INT8_DEQUANT", "").strip().lower()
        _DEQUANT_MODE = False
        if v in ("1", "on", "true"):
            try:
                import psutil
                _DEQUANT_MODE = psutil.virtual_memory().total >= 48 * (1 << 30)
                if not _DEQUANT_MODE:
                    print(f"{TAG} ASFP8_INT8_DEQUANT=1 ignored: needs >= 48 GiB RAM "
                          f"for the dequantised weight copy.")
            except Exception:
                _DEQUANT_MODE = False
    return _DEQUANT_MODE


def _maybe_dequant_weight(self, input):
    """One-off per layer: replace an eligible int8 (or fp16) weight with a plain
    tensor in the compute dtype, resident on MPS.

    At M=1..2 the per-call rotate/quant/rescale dispatch overhead dominates the matmul,
    so paying the dequant once is faster and numerically identical to comfy's W8A16.
    """
    if getattr(self, "_asfp8_deq_done", False):
        return
    try:
        from comfy_kitchen.tensor import QuantizedTensor

        # checks before the flag are properties of this call, so leave it unset for a
        # later one; everything past it is a stable property of the layer
        if not isinstance(input, torch.Tensor) or isinstance(input, QuantizedTensor):
            return
        if input.device.type != "mps":
            return
        # half precision only: fp32 would double the copy the RAM gate budgets for
        if input.dtype not in (torch.float16, torch.bfloat16):
            return
        # offloaded weight: dequantising would pull a bigger copy back onto the GPU
        # and undo the offload
        w = getattr(self, "weight", None)
        if w is None or getattr(w, "device", None) is None or w.device.type != "mps":
            return
        self._asfp8_deq_done = True
        if getattr(self, "_full_precision_mm", False):
            return
        if len(getattr(self, "weight_function", [])) or len(getattr(self, "bias_function", [])):
            return

        if isinstance(w, QuantizedTensor):
            if getattr(w, "_layout_cls", None) != "TensorWiseINT8Layout":
                return
            if getattr(w._params, "transposed", False):
                return
            plain = w.dequantize()
        elif isinstance(w, torch.Tensor) and w.dtype == torch.float16 and w.dtype != input.dtype:
            plain = w.detach()
        else:
            return
        if plain.dtype != input.dtype:
            plain = plain.to(input.dtype)
        if plain.device.type != "mps":
            plain = plain.to("mps")
        self.weight = torch.nn.Parameter(plain.contiguous(), requires_grad=False)
        b = getattr(self, "bias", None)
        if isinstance(b, torch.Tensor) and not isinstance(b, QuantizedTensor) and b.dtype != input.dtype:
            self.bias = torch.nn.Parameter(b.detach().to(device="mps", dtype=input.dtype), requires_grad=False)
        self.comfy_force_cast_weights = False
    except Exception as e:
        # latch on failure too, or the same error prints on every call
        self._asfp8_deq_done = True
        print(f"{TAG} weight dequant skipped ({e!r})")


def _maybe_dequant_embedding(self, dtype_hint):
    """One-off per Embedding: replace an int8 (or fp16) table with a plain tensor in
    the compute dtype, so cast_bias_weight stops casting it on every lookup."""
    if getattr(self, "_asfp8_deq_done", False):
        return
    try:
        from comfy_kitchen.tensor import QuantizedTensor

        w = self.weight
        # off-MPS is transient (offload), so leave the flag unset for a later call
        if w is None or w.device.type != "mps":
            return
        self._asfp8_deq_done = True
        if len(getattr(self, "weight_function", [])) or len(getattr(self, "bias_function", [])):
            return
        # half precision only, same memory budget as _maybe_dequant_weight
        if dtype_hint not in (torch.float16, torch.bfloat16):
            dtype_hint = None
        target = dtype_hint if dtype_hint is not None else torch.bfloat16
        if isinstance(w, QuantizedTensor):
            plain = w.dequantize().to(target)
        elif isinstance(w, torch.Tensor) and w.dtype == torch.float16 and w.dtype != target:
            plain = w.detach().to(target)
        else:
            return
        self.weight = torch.nn.Parameter(plain.contiguous(), requires_grad=False)
        self.comfy_force_cast_weights = False
    except Exception as e:
        self._asfp8_deq_done = True
        print(f"{TAG} embedding dequant skipped ({e!r})")


def install():
    global _installed
    if _installed:
        return
    if sys.platform != "darwin":
        return
    from . import _caps
    if not _caps.resolve("ASFP8_INT8_EXT", default_on=True, cap=_caps.shader_gate):
        return
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return

    # pick_operations() calls the factory by module name at model load, so wrapping
    # the module attribute is enough
    try:
        import comfy.ops as ops

        orig_factory = ops.mixed_precision_ops
        if getattr(orig_factory, "_asfp8_wrapped", False):
            _installed = True
            return

        def wrapped_factory(*a, **k):
            cls = orig_factory(*a, **k)
            linear_cls = getattr(cls, "Linear", None)
            if linear_cls is not None and not getattr(linear_cls, "_asfp8_int8_patched", False):
                orig_forward = linear_cls.forward

                def forward(self, input, *args, **kwargs):
                    if _dequant_enabled():
                        _maybe_dequant_weight(self, input)
                    res = _try_int8_kernel_forward(self, input)
                    if res is not None:
                        return res
                    return orig_forward(self, input, *args, **kwargs)

                linear_cls.forward = forward
                linear_cls._asfp8_int8_patched = True
            emb_cls = getattr(cls, "Embedding", None)
            if emb_cls is not None and not getattr(emb_cls, "_asfp8_int8_patched", False):
                orig_emb_forward = emb_cls.forward

                def emb_forward(self, input, *args, **kwargs):
                    if _dequant_enabled():
                        hint = kwargs.get("out_dtype", args[0] if args else None)
                        _maybe_dequant_embedding(self, hint)
                    return orig_emb_forward(self, input, *args, **kwargs)

                emb_cls.forward = emb_forward
                emb_cls._asfp8_int8_patched = True
            return cls

        wrapped_factory._asfp8_wrapped = True
        ops.mixed_precision_ops = wrapped_factory

        # ops.linear_input_act reads linear.weight directly and never calls Linear.forward,
        # so the dequant has to hook here too, and an int8 layer whose format comfy did not
        # disable (e.g. the MiniMax H3 VAE) would run comfy_kitchen's int8_linear, i.e.
        # torch._int_mm in fp32 on MPS. getattr: a comfy without it must not abort install().
        orig_lia = getattr(ops, "linear_input_act", None)
        if orig_lia is not None and not getattr(orig_lia, "_asfp8_deq_wrapped", False):
            def linear_input_act(linear, x, input_act, act_weight=None, act_eps=0.0,
                                 residual=None, residual_scale=None):
                if _dequant_enabled():
                    _maybe_dequant_weight(linear, x)
                if _kernel_owns(linear, x):
                    # the module call keeps instance-level hooks (LoRA bypass) in the path
                    out = linear(ops._eager_input_act(x, input_act, act_weight, act_eps))
                    return out if residual is None else torch.addcmul(residual, out, residual_scale)
                return orig_lia(linear, x, input_act, act_weight, act_eps, residual, residual_scale)

            linear_input_act._asfp8_deq_wrapped = True
            ops.linear_input_act = linear_input_act
    except Exception as e:
        print(f"{TAG} could not wrap mixed_precision_ops: {e!r}")
        return

    _installed = True
    deq = "on" if _dequant_enabled() else "off"
    print(
        f"{TAG} INT8 Linear seam installed on MPS (W8A8: fused rotate+quant, "
        f"int8 matmul2d; per-call weight dequant bypassed). The shader compiles and "
        f"runs its self-check on the first int8 layer, and only routes if that passes. "
        f"Once-at-load weight dequant: {deq} (opt-in: ASFP8_INT8_DEQUANT=1, "
        f"needs >= 48 GiB RAM)."
    )
