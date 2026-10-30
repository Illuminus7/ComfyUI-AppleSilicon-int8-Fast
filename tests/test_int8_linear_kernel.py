"""Tests for patch #17: INT8 Linears as W8A8 via the compile_shader rotate+quant and
int8 matmul2d kernels."""
import pytest
import torch

from conftest import requires_mps

from _patches import int8_linear_kernel_mps as patch

from _patches import _caps

_int8_enabled = torch.backends.mps.is_available() and _caps.resolve(
    "ASFP8_INT8_EXT", default_on=True, cap=_caps.shader_gate
)

requires_int8 = pytest.mark.skipif(not _int8_enabled, reason="int8 kernel not enabled on this machine")


@pytest.fixture(autouse=True)
def _clear_int8_kernel_memo():
    """Clear _caps.kernel_ready's per-process memo, or the first test to verify the
    kernel decides the answer for every later one."""
    _caps._kernel_ready.pop("int8", None)
    yield
    _caps._kernel_ready.pop("int8", None)


def _layer(weight, **attrs):
    class Holder:
        bias = None
        _full_precision_mm_config = False
        comfy_force_cast_weights = False
        weight_function = []
        bias_function = []

    h = Holder()
    h.weight = weight
    for k, v in attrs.items():
        setattr(h, k, v)
    return h


def _qweight(n=64, k=256):
    from comfy_kitchen.tensor import QuantizedTensor
    return QuantizedTensor.from_float((torch.randn(n, k) * 0.1).to(torch.bfloat16), "TensorWiseINT8Layout").to("mps")


def test_install_noop_when_explicitly_off(monkeypatch):
    """ASFP8_INT8_EXT=off force-disables even on capable hardware."""
    monkeypatch.setattr(_caps, "shader_gate", lambda: True)
    monkeypatch.setenv("ASFP8_INT8_EXT", "off")
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    patch.install()
    assert patch._installed is False


def test_install_noop_when_not_capable(monkeypatch):
    """DEFAULT ON but gated: unsupported hardware stays inert."""
    monkeypatch.delenv("ASFP8_INT8_EXT", raising=False)
    monkeypatch.setattr(_caps, "shader_gate", lambda: False)
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    patch.install()
    assert patch._installed is False


@requires_mps
def test_shader_compiles_on_first_eligible_forward_only(monkeypatch):
    """install() never compiles; the first int8 layer does, once, and a plain layer never."""
    calls = []
    monkeypatch.setattr(patch, "_verify", lambda: calls.append(1) or True)
    out = torch.full((1,), 7.0)
    monkeypatch.setattr(patch, "_int8_linear_kernel", lambda *a, **k: out)
    x = torch.randn(8, 256, dtype=torch.bfloat16, device="mps")

    assert patch._try_int8_kernel_forward(_layer(torch.zeros(8, 256)), x) is None
    assert calls == []
    assert patch._try_int8_kernel_forward(_layer(_qweight()), x) is out
    assert patch._try_int8_kernel_forward(_layer(_qweight()), x) is out
    assert calls == [1]


@requires_mps
def test_disabled_format_flag_still_routes(monkeypatch):
    """comfy sets _full_precision_mm on every int8 layer on MPS (int8 is a "disabled"
    format there); only the checkpoint's own full_precision_matrix_mult opts out."""
    monkeypatch.setattr(patch, "_verify", lambda: True)
    out = torch.full((1,), 7.0)
    monkeypatch.setattr(patch, "_int8_linear_kernel", lambda *a, **k: out)
    x = torch.randn(8, 256, dtype=torch.bfloat16, device="mps")
    assert patch._try_int8_kernel_forward(_layer(_qweight(), _full_precision_mm=True), x) is out
    assert patch._try_int8_kernel_forward(_layer(_qweight(), _full_precision_mm_config=True), x) is None


@requires_int8
@pytest.mark.parametrize("convrot", [False, True])
@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("M, N", [(77, 200), (20, 100), (64, 256)])
def test_kernel_matches_float_reference(convrot, per_channel, bias, M, N):
    """W8A8 output == float math on the rotated activation, within int8 quantization noise.
    2*77 x 200 runs the full-tile kernel plus both edge launches (row band and column), 2*20 x 100
    has no full tile, and 2*64 x 256 has no edge."""
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation
    assert patch._verify()
    g = torch.Generator().manual_seed(3)
    K = 768
    x = torch.randn(2, M, K, generator=g).to(torch.bfloat16)
    w = torch.randint(-127, 128, (N, K), generator=g, dtype=torch.int8)
    ws = torch.rand(N if per_channel else 1, generator=g) * 0.01 + 0.001
    b = torch.randn(N, generator=g).to(torch.bfloat16) if bias else None
    xf = x.float()
    if convrot:
        xf = _rotate_activation(xf, _build_hadamard(256, dtype=torch.float32), 256)
    ref = xf @ (w.float() * ws.reshape(-1, 1)).t()
    if bias:
        ref = ref + b.float()
    out = patch._int8_linear_kernel(x.to("mps"), w.to("mps"), ws.to("mps"), b.to("mps") if bias else None, convrot)
    assert out.shape == (2, M, N) and out.dtype == torch.bfloat16
    err = ((out.cpu().float() - ref).norm() / ref.norm()).item()
    assert err < 0.02, err


@requires_int8
def test_int8_kernel_compiles_when_enabled():
    """Canary: a kernel the node turns on actually builds and verifies (#13)."""
    assert patch._verify()


@requires_mps
def test_failed_verification_disables_the_kernel_and_is_not_retried(monkeypatch):
    """A kernel that fails verification costs one attempt, not one per layer (#14)."""
    attempts = []
    monkeypatch.setattr(patch, "_verify", lambda: attempts.append(1) or False)
    x = torch.randn(8, 256, dtype=torch.bfloat16, device="mps")
    for _ in range(3):
        assert patch._try_int8_kernel_forward(_layer(_qweight()), x) is None
    assert attempts == [1]
    assert _caps._kernel_ready["int8"] is False


@requires_mps
def test_a_forward_failure_after_verification_disables_the_kernel(monkeypatch):
    """A dispatch failure after verification passed is latched off for the session (#13)."""
    calls = []

    def exploding_kernel(*a, **k):
        calls.append(1)
        raise RuntimeError("dispatch blew up")

    monkeypatch.setattr(patch, "_verify", lambda: True)
    monkeypatch.setattr(patch, "_int8_linear_kernel", exploding_kernel)
    x = torch.randn(8, 256, dtype=torch.bfloat16, device="mps")
    for _ in range(4):
        assert patch._try_int8_kernel_forward(_layer(_qweight()), x) is None
    assert calls == [1]
    assert _caps._kernel_ready["int8"] is False


# --- the hardware gate: issues #25 and #27 ----------------------------------
# One defect from both ends: the old gate measured the torch build's default MSL
# rather than the GPU, so it answered yes on an M4 Pro and no on an M5 Max.


def test_install_is_inert_on_a_pre_m5_chip(monkeypatch):
    """A chip positively named as pre-M5 never reaches the build at all (#25)."""
    monkeypatch.setattr(_caps, "_chip_gen", _caps._UNPROBED)
    monkeypatch.delenv("ASFP8_INT8_EXT", raising=False)
    monkeypatch.setattr(_caps, "_cpu_brand_string", lambda: "Apple M4 Pro")
    monkeypatch.setattr(_caps, "_compile_shader", True)
    monkeypatch.setattr(patch, "_installed", False, raising=False)

    patch.install()

    assert patch._installed is False


def _stub_comfy_ops(monkeypatch):
    """Stand in for `comfy.ops`, which isn't importable from the repo root: without it
    install() bails on the import before the gate tests can observe anything."""
    import sys
    import types

    class _Linear:
        def forward(self, x):
            return x

    def mixed_precision_ops(*a, **k):
        return type("Ops", (), {"Linear": type("Linear", (_Linear,), {})})

    comfy = types.ModuleType("comfy")
    ops = types.ModuleType("comfy.ops")
    ops.mixed_precision_ops = mixed_precision_ops
    comfy.ops = ops
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.ops", ops)
    return ops


def test_install_survives_a_failing_compile_shader_probe(monkeypatch):
    """A failing compile_shader probe does not disable the kernel (#27)."""
    _stub_comfy_ops(monkeypatch)
    monkeypatch.setattr(_caps, "_chip_gen", _caps._UNPROBED)
    monkeypatch.delenv("ASFP8_INT8_EXT", raising=False)
    monkeypatch.setattr(_caps, "_cpu_brand_string", lambda: "Apple M5 Max")
    monkeypatch.setattr(_caps, "has_tensor_ops_matmul2d", lambda: False)
    monkeypatch.setattr(_caps, "_compile_shader", True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(patch, "_installed", False, raising=False)

    patch.install()

    assert patch._installed is True


def test_install_banner_does_not_claim_an_unverified_kernel(monkeypatch, capsys):
    """The startup banner promises a check, not a result nothing has verified yet (#25)."""
    _stub_comfy_ops(monkeypatch)
    monkeypatch.setattr(_caps, "_chip_gen", _caps._UNPROBED)
    monkeypatch.delenv("ASFP8_INT8_EXT", raising=False)
    monkeypatch.setattr(_caps, "_cpu_brand_string", lambda: "Apple M5 Max")
    monkeypatch.setattr(_caps, "_compile_shader", True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(patch, "_installed", False, raising=False)

    patch.install()

    out = capsys.readouterr().out
    assert "routed through" not in out, f"banner claims a result it hasn't got: {out}"
    assert "self-check" in out.lower(), f"banner doesn't mention verification: {out}"


def test_linear_input_act_wrapper_forwards_the_extended_signature(monkeypatch):
    """The wrapper accepts ComfyUI v0.36's extended signature (#36)."""
    ops = _stub_comfy_ops(monkeypatch)
    calls = []

    def linear_input_act(linear, x, input_act, act_weight=None, act_eps=0.0,
                         residual=None, residual_scale=None):
        calls.append((input_act, act_weight, act_eps, residual, residual_scale))
        return x

    ops.linear_input_act = linear_input_act
    monkeypatch.setattr(_caps, "_chip_gen", _caps._UNPROBED)
    monkeypatch.delenv("ASFP8_INT8_EXT", raising=False)
    monkeypatch.setattr(_caps, "_cpu_brand_string", lambda: "Apple M5 Max")
    monkeypatch.setattr(_caps, "_compile_shader", True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(patch, "_installed", False, raising=False)
    monkeypatch.setattr(patch, "_dequant_enabled", lambda: False)

    patch.install()
    assert ops.linear_input_act is not linear_input_act, "the seam was never wrapped"

    x = torch.zeros(2, 4)
    ops.linear_input_act(object(), x, "rms_norm", "norm_w", 1e-6)
    ops.linear_input_act(object(), x, "rms_norm", "norm_w", 1e-6,
                         residual="res", residual_scale="scale")
    ops.linear_input_act(object(), x, "swiglu")

    assert calls == [
        ("rms_norm", "norm_w", 1e-6, None, None),
        ("rms_norm", "norm_w", 1e-6, "res", "scale"),
        ("swiglu", None, 0.0, None, None),
    ]


@requires_mps
def test_force_cast_weights_does_not_disqualify_the_kernel(monkeypatch):
    """comfy_force_cast_weights does not push a layer off the kernel route (#26).

    On a QuantizedTensor that cast is the per-call dequant this kernel bypasses.
    """
    from comfy_kitchen.tensor import QuantizedTensor

    monkeypatch.setattr(patch, "_verify", lambda: True)

    out = torch.full((1,), 7.0)
    monkeypatch.setattr(patch, "_int8_linear_kernel", lambda *a, **k: out, raising=False)

    qw = QuantizedTensor.from_float(
        (torch.randn(64, 256) * 0.1).to(torch.bfloat16), "TensorWiseINT8Layout"
    ).to("mps")

    class Holder:
        weight = qw
        bias = None
        _full_precision_mm_config = False
        comfy_force_cast_weights = True
        weight_function = []
        bias_function = []

    x = torch.randn(8, 256, dtype=torch.bfloat16, device="mps")
    assert patch._try_int8_kernel_forward(Holder(), x) is out, (
        "comfy_force_cast_weights pushed an int8 layer off the kernel route"
    )


@requires_mps
def test_dequant_retries_after_a_transient_off_mps_call(monkeypatch):
    """An off-MPS first call doesn't latch _asfp8_deq_done, so a later MPS call still
    dequantises."""
    from comfy_kitchen.tensor import QuantizedTensor

    monkeypatch.setattr(patch, "_DEQUANT_MODE", True, raising=False)

    qw = QuantizedTensor.from_float(
        (torch.randn(64, 256) * 0.1).to(torch.bfloat16), "TensorWiseINT8Layout"
    ).to("mps")

    class Holder:
        _full_precision_mm_config = False
        comfy_force_cast_weights = True
        weight_function = []
        bias_function = []

    layer = Holder()
    layer.weight = qw
    layer.bias = None

    patch._maybe_dequant_weight(layer, torch.randn(1, 256, dtype=torch.bfloat16))
    assert isinstance(layer.weight, QuantizedTensor), "dequantised from a CPU-input call"
    assert not getattr(layer, "_asfp8_deq_done", False), "off-MPS call latched the flag"

    x = torch.randn(1, 256, dtype=torch.bfloat16, device="mps")
    patch._maybe_dequant_weight(layer, x)
    assert not isinstance(layer.weight, QuantizedTensor), "MPS call did not dequantise"
    assert layer.weight.dtype == torch.bfloat16
    assert layer.weight.device.type == "mps"
    assert layer._asfp8_deq_done is True

    ref = qw.dequantize().to(torch.bfloat16)
    assert torch.equal(layer.weight.detach().cpu(), ref.cpu()), (
        "dequantised weight differs from QuantizedTensor.dequantize()"
    )


# --- the ASFP8_INT8_DEQUANT gate: the switch people actually touch --------------


def _resolve_dequant_gate(monkeypatch, env, total_ram):
    import psutil

    class _VM:
        total = total_ram

    monkeypatch.setattr(patch, "_DEQUANT_MODE", None, raising=False)
    if env is None:
        monkeypatch.delenv("ASFP8_INT8_DEQUANT", raising=False)
    else:
        monkeypatch.setenv("ASFP8_INT8_DEQUANT", env)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _VM())
    return patch._dequant_enabled()


def test_dequant_gate_is_opt_in(monkeypatch):
    """Unset stays OFF regardless of RAM."""
    assert _resolve_dequant_gate(monkeypatch, None, 128 * (1 << 30)) is False


def test_dequant_gate_off_stays_off(monkeypatch):
    assert _resolve_dequant_gate(monkeypatch, "off", 128 * (1 << 30)) is False


def test_dequant_gate_on_with_enough_ram(monkeypatch):
    assert _resolve_dequant_gate(monkeypatch, "1", 128 * (1 << 30)) is True


def test_dequant_gate_ram_check_overrides_opt_in(monkeypatch, capsys):
    """=1 on a small box is refused, and says so."""
    assert _resolve_dequant_gate(monkeypatch, "1", 16 * (1 << 30)) is False
    assert "48 GiB" in capsys.readouterr().out


def test_dequant_gate_memoises(monkeypatch):
    """The gate resolves once, so a later env change can't flip a live session."""
    assert _resolve_dequant_gate(monkeypatch, "1", 128 * (1 << 30)) is True
    monkeypatch.setenv("ASFP8_INT8_DEQUANT", "off")
    assert patch._dequant_enabled() is True


@requires_mps
def test_offloaded_cpu_weight_does_not_latch_the_kernel_off(monkeypatch):
    """An offloaded CPU weight doesn't latch the kernel off for the session (#26).

    Reaching the kernel with an MPS input and a CPU weight would throw and call
    mark_kernel_failed.
    """
    from comfy_kitchen.tensor import QuantizedTensor
    from _patches import _caps

    qw_cpu = QuantizedTensor.from_float(
        (torch.randn(64, 256) * 0.1).to(torch.bfloat16), "TensorWiseINT8Layout"
    )
    assert qw_cpu.device.type == "cpu", "fixture must stay off-device"

    class Holder:
        weight = qw_cpu
        bias = None
        _full_precision_mm_config = False
        comfy_force_cast_weights = True
        weight_function = []
        bias_function = []

    x = torch.randn(8, 256, dtype=torch.bfloat16, device="mps")

    assert patch._try_int8_kernel_forward(Holder(), x) is None
    assert _caps._kernel_ready.get("int8") is not False, (
        "an offloaded weight latched the int8 kernel off for the session"
    )


@requires_mps
def test_dequant_leaves_an_offloaded_weight_on_the_cpu(monkeypatch):
    """An offloaded weight is left on the CPU, with the flag unset so the layer still
    dequantises once comfy brings it back."""
    from comfy_kitchen.tensor import QuantizedTensor

    monkeypatch.setattr(patch, "_DEQUANT_MODE", True, raising=False)

    qw_cpu = QuantizedTensor.from_float(
        (torch.randn(64, 256) * 0.1).to(torch.bfloat16), "TensorWiseINT8Layout"
    )
    assert qw_cpu.device.type == "cpu", "fixture must stay offloaded"

    class Holder:
        _full_precision_mm_config = False
        comfy_force_cast_weights = True
        weight_function = []
        bias_function = []

    layer = Holder()
    layer.weight = qw_cpu
    layer.bias = None

    x = torch.randn(1, 256, dtype=torch.bfloat16, device="mps")
    patch._maybe_dequant_weight(layer, x)

    assert isinstance(layer.weight, QuantizedTensor), "offloaded weight was dequantised"
    assert layer.weight.device.type == "cpu", "offloaded weight was pulled onto the GPU"
    assert not getattr(layer, "_asfp8_deq_done", False), (
        "a transient offload latched the layer out of ever dequantising"
    )


@requires_int8
def test_shared_activation_quantizes_once_and_matches(monkeypatch):
    """q/k/v projections of one input inside shared_activation reuse one rotate+quantize,
    with outputs identical to quantizing per call; other inputs are unaffected."""
    assert patch._verify()
    g = torch.Generator().manual_seed(4)
    x = torch.randn(3, 40, 512, generator=g).to("mps", torch.bfloat16)
    y = torch.randn(3, 40, 512, generator=g).to("mps", torch.bfloat16)
    ws = [(torch.randint(-127, 128, (96, 512), generator=g, dtype=torch.int8).to("mps"), torch.tensor([0.01], device="mps")) for _ in range(3)]
    ref = [patch._int8_linear_kernel(x, w, s, None, True) for w, s in ws]
    calls = []
    q = patch._quantize_activation
    monkeypatch.setattr(patch, "_quantize_activation", lambda *a: calls.append(1) or q(*a))
    with patch.shared_activation(x):
        out = [patch._int8_linear_kernel(x, w, s, None, True) for w, s in ws]
        patch._int8_linear_kernel(y, *ws[0], None, True)
    assert calls == [1, 1] and patch._shared is None
    assert all(torch.equal(a, b) for a, b in zip(out, ref))


def test_quant_target_is_inert_unless_the_int8_patch_is_installed(monkeypatch):
    """With the int8 seam off (ASFP8_INT8_EXT=off), producers must not pre-quantize: the
    projections then take comfy's own path, which never reads the shared quantization."""
    monkeypatch.setattr(patch, "_installed", False)
    x = torch.zeros(2, 512, dtype=torch.bfloat16)
    assert patch.quant_target(x, object()) is None


@requires_int8
def test_shared_activation_takes_the_producers_quantization(monkeypatch):
    """A producer's quantization is used as is, an inner block on the same tensor keeps it,
    and a layer with the other rotation setting re-quantizes."""
    assert patch._verify()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(70, 512, generator=g).to("mps", torch.bfloat16)
    w = torch.randint(-127, 128, (200, 512), generator=g, dtype=torch.int8).to("mps")
    ws = torch.full((200,), 0.01, device="mps")
    q, scale = patch._quantize_activation(x, True)
    want = patch._int8_linear_kernel(x, w, ws, None, True)
    calls = []
    real = patch._quantize_activation
    monkeypatch.setattr(patch, "_quantize_activation", lambda x2d, r: calls.append(r) or real(x2d, r))
    with patch.shared_activation(x, (True, q, scale)):
        with patch.shared_activation(x):
            assert torch.equal(patch._int8_linear_kernel(x, w, ws, None, True), want)
        assert calls == []
        patch._int8_linear_kernel(x, w, ws, None, False)
        assert calls == [False]
    assert patch._shared is None
