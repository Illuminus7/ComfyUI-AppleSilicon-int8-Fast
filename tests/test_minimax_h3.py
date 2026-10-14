"""Tests for the MiniMax H3 fast path (fused q/k norm+RoPE, adaLN modulate, SwiGLU) and the
F.scaled_dot_product_attention route into the flash kernel. Each compares against stock math."""
import importlib
import os
import sys

import pytest
import torch
import torch.nn.functional as F

from conftest import requires_mps

COMFY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@pytest.fixture(scope="module")
def H():
    if COMFY not in sys.path:
        sys.path.insert(0, COMFY)
    try:
        import comfy.options
        comfy.options.enable_args_parsing(False)
        hm = importlib.import_module("comfy.ldm.minimax.model")
        import comfy.ops
        import comfy.quant_ops
    except Exception as e:
        pytest.skip(f"ComfyUI not importable: {e!r}")
    from _patches import minimax_h3_mps as H
    H._orig.setdefault("rms_rope_split_half_", comfy.quant_ops.ck.rms_rope_split_half_)
    H._orig.setdefault("_mod_scale_shift", hm._mod_scale_shift)
    H._orig.setdefault("swiglu", comfy.ops.INPUT_ACT_EAGER["swiglu"])
    if H._orig["_mod_scale_shift"] is H._mod_scale_shift:
        pytest.skip("patch already installed in this process")
    return H


def _err(a, ref):
    a, ref = a.cpu().float(), ref.cpu().float()
    return ((a - ref).norm() / ref.norm()).item()


@requires_mps
def test_shaders_verify(H):
    assert H._verify()


@requires_mps
@pytest.mark.parametrize("rot_dim", [96, 128])
def test_rms_rope_in_place_on_qkv_views(H, rot_dim):
    """q/k are strided views into one fused qkv buffer; v must be left untouched."""
    assert H._verify()
    g = torch.Generator().manual_seed(1)
    S, nh, half = 50, 3, rot_dim // 2
    qkv = torch.randn(1, S, 3 * nh * 128, generator=g).to(torch.bfloat16)
    ang = torch.rand(S, half, generator=g) * 9
    f = torch.stack([ang.cos(), -ang.sin(), ang.sin(), ang.cos()], -1).reshape(1, S, 1, half, 2, 2).to(torch.bfloat16)
    qw, kw = (torch.randn(128, generator=g).to(torch.bfloat16) for _ in range(2))
    ref = qkv.clone()
    rq, rk, _ = ref.split(nh * 128, dim=-1)
    H._orig["rms_rope_split_half_"](rq.view(1, S, nh, 128), rk.view(1, S, nh, 128), f, qw, kw, epsilon=1e-5, rot_dim=rot_dim)
    out = qkv.to("mps")
    oq, ok, _ = out.split(nh * 128, dim=-1)
    H._rms_rope_split_half_(oq.view(1, S, nh, 128), ok.view(1, S, nh, 128), f.to("mps"), qw.to("mps"), kw.to("mps"),
                            epsilon=1e-5, rot_dim=rot_dim)
    assert _err(out, ref) < 0.01
    assert torch.equal(out.cpu()[..., 2 * nh * 128:], qkv[..., 2 * nh * 128:])


@requires_mps
def test_mod_scale_shift_and_masked_fallback(H):
    assert H._verify()
    g = torch.Generator().manual_seed(2)
    h = torch.randn(40, 256, generator=g).to(torch.bfloat16)
    sc, sh = torch.randn(3, 256, generator=g) * 0.2, torch.randn(3, 256, generator=g) * 0.2
    segs = [(0, 9, 2), (9, 40, 0)]
    ref = H._orig["_mod_scale_shift"](h.float().clone(), sh, sc, segs)
    assert _err(H._mod_scale_shift(h.to("mps"), sh.to("mps"), sc.to("mps"), segs), ref) < 0.01
    # per-token rows (masked conditioning) take the stock path
    rows = torch.tensor([0, 1, 2] * 13 + [0], device="mps")
    segs = [(0, 40, rows)]
    ref = H._orig["_mod_scale_shift"](h.float().clone(), sh, sc, [(0, 40, rows.cpu())])
    assert _err(H._mod_scale_shift(h.to("mps"), sh.to("mps"), sc.to("mps"), segs), ref) < 0.01


@requires_mps
def test_swiglu(H):
    assert H._verify()
    x = torch.randn(11, 2 * 96, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    assert _err(H._swiglu(x.to("mps")), H._orig["swiglu"](x.float())) < 0.01


@requires_mps
def test_sdpa_routes_unmasked_calls_to_the_kernel(H):
    from _patches import attention_mps as A
    calls = []

    def orig(q, k, v, attn_mask=None, **kw):
        calls.append(attn_mask is not None)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, **kw)

    sdpa = A._make_sdpa(orig)
    g = torch.Generator().manual_seed(4)
    q = torch.randn(1, 8, 200, 128, generator=g).to(torch.bfloat16)
    k, v = (torch.randn(1, 2, 170, 128, generator=g).to(torch.bfloat16) for _ in range(2))
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), enable_gqa=True)
    out = sdpa(q.to("mps"), k.to("mps"), v.to("mps"), enable_gqa=True)
    assert calls == [] and out.shape == (1, 8, 200, 128) and _err(out, ref) < 0.005
    sdpa(q.to("mps"), q.to("mps"), q.to("mps"), attn_mask=torch.zeros(200, 200, device="mps", dtype=torch.bfloat16))
    sdpa(q.to("mps"), q.to("mps"), q.to("mps"), is_causal=True)
    assert calls == [True, False]


@requires_mps
def test_nvfp4_dequant_shader_is_bit_exact(H):
    """The text encoder's NVFP4 weights: the shader must match comfy_kitchen's eager dequant exactly."""
    import comfy_kitchen.backends.eager.quantization as qmod
    from comfy_kitchen.float_utils import to_blocked
    from _patches import comfykitchen_fp8 as C
    orig = getattr(qmod.dequantize_nvfp4, "__wrapped__", None) or qmod.dequantize_nvfp4
    assert C._nvfp4_verify(lambda *a, **k: orig(*[t.cpu() if isinstance(t, torch.Tensor) else t for t in a], **k))()
    g = torch.Generator().manual_seed(5)
    qx = torch.randint(0, 256, (300, 80), generator=g, dtype=torch.uint8)
    bs = to_blocked((torch.rand(300, 10, generator=g) * 3 + 0.05).to(torch.float8_e4m3fn), flatten=False)
    ts = torch.tensor(1.7)
    ref = orig(qx, ts, bs, torch.bfloat16)
    assert torch.equal(C._nvfp4_mps(qx.to("mps"), ts.to("mps"), bs.to("mps"), torch.bfloat16, True).cpu(), ref)


@requires_mps
def test_lora_bypass_in_place_matches_stock(H):
    import comfy.weight_adapter.bypass as bp
    from comfy.weight_adapter.lora import LoRAAdapter
    from _patches import lora_bypass_mps as L
    stock = bp.BypassForwardHook._bypass_forward
    if not L._installed:
        L.install()
    if bp.BypassForwardHook._bypass_forward is stock:
        pytest.skip("patch not installable here")
    g = torch.Generator().manual_seed(6)
    lin = torch.nn.Linear(256, 512, bias=False).to("mps", torch.bfloat16)
    up, down = (torch.randn(512, 16, generator=g) * 0.2).to("mps", torch.bfloat16), (torch.randn(16, 256, generator=g) * 0.2).to("mps", torch.bfloat16)
    adapter = LoRAAdapter(set(), (up, down, torch.tensor(8.0), None, None, None))
    hook = bp.BypassForwardHook(lin, adapter, multiplier=0.7)
    hook.original_forward = lin.forward
    x = torch.randn(3, 40, 256, generator=g).to("mps", torch.bfloat16)
    ref = lin(x).float() + (x.float() @ down.float().t() @ up.float().t()) * (8.0 / 16 * 0.7)
    out = bp.BypassForwardHook._bypass_forward(hook, x)
    assert _err(out, ref) < 0.01


@requires_mps
def test_linear_input_act_routes_int8_layers_to_the_kernel(H, monkeypatch):
    """An int8 layer comfy did not mark full-precision (the H3 VAE) must not reach
    comfy_kitchen's int8_linear (torch._int_mm in fp32 on MPS)."""
    import comfy.ops
    from comfy_kitchen.tensor import QuantizedTensor
    from _patches import int8_linear_kernel_mps as P
    monkeypatch.setattr(P, "_verify", lambda: True)
    calls = []

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = QuantizedTensor.from_float((torch.randn(64, 256) * 0.1).to(torch.bfloat16), "TensorWiseINT8Layout").to("mps")
            self._full_precision_mm_config = False

        def forward(self, x):
            calls.append(x.shape)
            return torch.zeros(*x.shape[:-1], 64, device=x.device, dtype=x.dtype)

    x = torch.randn(5, 512, device="mps", dtype=torch.bfloat16)
    assert P._kernel_owns(Layer(), x)
    if getattr(comfy.ops.linear_input_act, "_asfp8_deq_wrapped", False):
        res = torch.ones(5, 64, device="mps", dtype=torch.bfloat16)
        out = comfy.ops.linear_input_act(Layer(), x, "swiglu", residual=res, residual_scale=torch.full((64,), 2.0, device="mps", dtype=torch.bfloat16))
        assert calls == [(5, 256)] and torch.equal(out.cpu(), res.cpu())


@requires_mps
@pytest.mark.parametrize("kw", [dict(tau=0.5, sink_blocks=[0, 3], sink_q=[2, 4]), dict(topk_ratio=0.2), dict(tau=1.3, tail=False)])
def test_sol_attn_matches_reference(H, kw):
    """Sparse attention vs comfy_kitchen's eager Sol-Attn on (B, T, H, D) strided views."""
    from comfy_kitchen.backends.eager.sol_attn import sol_attn as ref_sol_attn
    from _patches import sol_attn_mps as S
    assert S._verify()
    g = torch.Generator().manual_seed(7)
    q, k, v = (torch.randn(1, 2, 517, 128, generator=g).to(torch.bfloat16).transpose(1, 2) for _ in range(3))
    ref = ref_sol_attn(q.float(), k.float(), v.float(), **kw)
    out = S._sol_attn_mps(q.to("mps"), k.to("mps"), v.to("mps"), kw.get("tau", 1.0), None, kw.get("sink_blocks"),
                          kw.get("sink_q"), kw.get("topk_ratio", 0.0), kw.get("tail", True))
    assert out.shape == (1, 517, 2, 128) and _err(out, ref) < 0.01


def test_sparse_node_accepts_mps(H, monkeypatch):
    """The node's _ineligible() lets MPS calls through and keeps its other rules."""
    import types
    from _patches import sol_attn_mps as S
    fake = types.ModuleType("fake_nodes_sparse_attention")
    fake.make_attention_override = object()
    fake._ineligible = lambda q, k, v, d: "not on CUDA"
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    S._patch_node()
    q = torch.empty(1, 4, 2, 128, device="mps", dtype=torch.bfloat16)
    assert fake._ineligible(q, q, q, 128) is None
    assert fake._ineligible(q, q, q, 64) is not None
    assert fake._ineligible(q.cpu(), q.cpu(), q.cpu(), 128) == "not on CUDA"
