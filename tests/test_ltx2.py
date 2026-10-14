"""Tests for the LTX-2.x fast paths: the transformer attention's fused q/k RMSNorm + RoPE and
the diffusion VAE decoder's fused norm/RoPE/SwiGLU, each against the stock module forward."""
import importlib
import os
import sys

import pytest
import torch

from conftest import requires_mps

COMFY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@pytest.fixture(scope="module")
def L():
    if COMFY not in sys.path:
        sys.path.insert(0, COMFY)
    try:
        import comfy.options
        comfy.options.enable_args_parsing(False)
        lm = importlib.import_module("comfy.ldm.lightricks.model")
        nd = importlib.import_module("comfy.ldm.lightricks.vae.na_diffusion_decoder")
    except Exception as e:
        pytest.skip(f"ComfyUI not importable: {e!r}")
    from _patches import ltx2_mps as L
    if lm.CrossAttention.forward is L._attention_forward:
        pytest.skip("patch already installed in this process")
    L._orig.setdefault("attention", lm.CrossAttention.forward)
    L._orig.setdefault("na_attention", nd.NeighborhoodAttention3D.forward)
    L._orig.setdefault("swiglu", nd.SwiGLU.forward)
    L._orig.setdefault("diffusion_block", nd.DiffusionNABlock.forward)
    return L


def _err(a, ref):
    a, ref = a.cpu().float(), ref.cpu().float()
    return ((a - ref).norm() / ref.norm()).item()


def _init(module, seed):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * (0.05 if p.dim() > 1 else 1.0))
    return module.to("mps", torch.bfloat16)


def _rot(n, heads, half, g):
    ang = torch.rand(1, n, heads, half, generator=g) * 6
    return torch.stack([ang.cos(), -ang.sin(), ang.sin(), ang.cos()], -1).reshape(1, n, heads, half, 2, 2).to("mps", torch.bfloat16)


@requires_mps
def test_shaders_verify(L):
    assert L._verify()


@requires_mps
def test_self_attention_matches_stock(L):
    import comfy.ops
    from comfy.ldm.lightricks.model import CrossAttention
    attn = _init(CrossAttention(256, heads=2, dim_head=128, operations=comfy.ops.disable_weight_init), 1)
    g = torch.Generator().manual_seed(2)
    x = torch.randn(1, 75, 256, generator=g).to("mps", torch.bfloat16)
    pe = (_rot(75, 2, 64, g), True)
    with torch.inference_mode():
        ref = L._orig["attention"](attn, x, pe=pe)
        out = L._attention_forward(attn, x, pe=pe)
    assert _err(out, ref) < 0.01


@requires_mps
def test_cross_attention_with_its_own_key_rope_matches_stock(L):
    """LTX's audio-video cross attention: q and k from different streams, each with its own table."""
    import comfy.ops
    from comfy.ldm.lightricks.model import CrossAttention
    attn = _init(CrossAttention(512, context_dim=256, heads=2, dim_head=64, operations=comfy.ops.disable_weight_init), 3)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 90, 512, generator=g).to("mps", torch.bfloat16)
    ctx = torch.randn(1, 21, 256, generator=g).to("mps", torch.bfloat16)
    pe, k_pe = (_rot(90, 2, 32, g), True), (_rot(21, 2, 32, g), True)
    with torch.inference_mode():
        ref = L._orig["attention"](attn, x, context=ctx, pe=pe, k_pe=k_pe)
        out = L._attention_forward(attn, x, context=ctx, pe=pe, k_pe=k_pe)
    assert _err(out, ref) < 0.01


@requires_mps
def test_stg_and_patched_norm_take_the_stock_forward(L, monkeypatch):
    import comfy.ops
    from comfy.ldm.lightricks.model import CrossAttention
    attn = _init(CrossAttention(256, heads=2, dim_head=128, operations=comfy.ops.disable_weight_init), 5)
    calls = []
    monkeypatch.setitem(L._orig, "attention", lambda *a, **k: calls.append(1))
    x = torch.randn(1, 8, 256, device="mps", dtype=torch.bfloat16)
    pe = (_rot(8, 2, 64, torch.Generator().manual_seed(6)), True)
    L._attention_forward(attn, x, pe=pe, transformer_options={"stg_skip_self_attn": True})
    attn.q_norm.weight_function = [lambda w: w]   # a lowvram LoRA patch on the norm
    L._attention_forward(attn, x, pe=pe)
    assert calls == [1, 1]


@requires_mps
@pytest.mark.parametrize("pre,add_to", [(False, False), (True, True)])
def test_decoder_neighborhood_attention_matches_stock(L, pre, add_to):
    import comfy.ldm.lightricks.vae.na_diffusion_decoder as nd
    attn = _init(nd.NeighborhoodAttention3D(128, (3, 5, 5)), 7)
    x = torch.randn(2, 4, 6, 7, 128, generator=torch.Generator().manual_seed(8)).to("mps", torch.bfloat16)
    p = (lambda s: s * 0.5) if pre else None
    with torch.inference_mode():
        ref = L._orig["na_attention"](attn, x.clone(), pre=p, add_to=x.clone() if add_to else None)
        out = L._na_attention_forward(attn, x.clone(), pre=p, add_to=x.clone() if add_to else None)
    assert _err(out, ref) < 0.01


@requires_mps
def test_decoder_diffusion_block_and_swiglu_match_stock(L):
    import comfy.ldm.lightricks.vae.na_diffusion_decoder as nd
    blk = _init(nd.DiffusionNABlock(128, (3, 5, 5), context_channels=128), 9)
    g = torch.Generator().manual_seed(10)
    x = torch.randn(1, 3, 5, 6, 128, generator=g).to("mps", torch.bfloat16)
    ctx = torch.randn(1, 3, 5, 6, 128, generator=g).to("mps", torch.bfloat16)
    mod = tuple(torch.randn(1, 1, 1, 1, 128, generator=g).to("mps", torch.bfloat16) * 0.1 for _ in range(7))
    with torch.inference_mode():
        ref = L._orig["diffusion_block"](blk, x.clone(), ctx, mod)
        out = L._diffusion_block_forward(blk, x.clone(), ctx, mod)
        assert _err(out, ref) < 0.01
        sl = torch.randn(1, 3, 5, 6, 128, generator=g).to("mps", torch.bfloat16)
        assert _err(L._swiglu_forward(blk.mlp, sl), L._orig["swiglu"](blk.mlp, sl)) < 0.01
