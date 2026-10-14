"""Tests for the Krea2 fast path: matmul2d flash attention, fused block shaders, and the
Wan VAE single-frame conv3d / channel RMS_norm. Each compares against the stock math."""
import importlib
import os
import sys

import pytest
import torch
import torch.nn.functional as F

from conftest import requires_mps

COMFY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@pytest.fixture(scope="module")
def comfy():
    """These patches wrap ComfyUI modules, so they need ComfyUI importable."""
    if COMFY not in sys.path:
        sys.path.insert(0, COMFY)
    try:
        import comfy.options
        comfy.options.enable_args_parsing(False)
        importlib.import_module("comfy.ldm.krea2.model")
    except Exception as e:
        pytest.skip(f"ComfyUI not importable: {e!r}")


def _err(a, ref):
    return ((a.cpu().float() - ref).norm() / ref.norm()).item()


@requires_mps
@pytest.mark.parametrize("gqa", [1, 4])
@pytest.mark.parametrize("Lq,Lk", [(70, 150), (300, 77), (1, 33)])
def test_flash_attention_matches_softmax(comfy, gqa, Lq, Lk):
    from _patches import attention_mps as A
    assert A._verify(128)
    g = torch.Generator().manual_seed(0)
    q = torch.randn(2, 8, Lq, 128, generator=g).to(torch.bfloat16)
    k, v = (torch.randn(2, 8 // gqa, Lk, 128, generator=g).to(torch.bfloat16) for _ in range(2))
    kr, vr = k.float().repeat_interleave(gqa, 1), v.float().repeat_interleave(gqa, 1)
    ref = (torch.softmax(q.float() @ kr.transpose(-1, -2) * 128 ** -0.5, -1) @ vr).transpose(1, 2)
    out = A._flash(q.to("mps"), k.to("mps"), v.to("mps"), 128 ** -0.5)
    assert _err(out, ref) < 0.005


@requires_mps
def test_attention_wrapper_falls_back_for_masks(comfy):
    """Masked calls go to the attention ComfyUI picked; unmasked ones to the kernel."""
    from _patches import attention_mps as A
    calls = []

    def orig(q, k, v, heads, mask=None, **kw):
        calls.append(mask is not None)
        return "orig"

    fn = A._make_attention(orig)
    q = torch.randn(1, 40, 4 * 128, device="mps", dtype=torch.bfloat16)
    assert fn(q, q, q, 4, mask=torch.ones(40, 40, device="mps")) == "orig"
    out = fn(q, q, q, 4)
    assert calls == [True] and out.shape == (1, 40, 512)
    ref = F.scaled_dot_product_attention(*(q.float().cpu().unflatten(-1, (4, 128)).transpose(1, 2),) * 3)
    assert _err(out, ref.transpose(1, 2).flatten(2)) < 0.005


@requires_mps
def test_krea2_fused_shaders_match_stock(comfy):
    from _patches import krea2_mps as K
    import comfy.ldm.krea2.model as km
    if not K._orig:
        K._orig.update({c.__name__: c.forward for c in (km.RMSNorm, km.SwiGLU, km.Attention, km.SingleStreamBlock, km.LastLayer)})
    assert K._verify()


@requires_mps
def test_krea2_block_fast_path_matches_stock(comfy):
    """A whole SingleStreamBlock through the fused path vs its original forward."""
    from _patches import krea2_mps as K
    import comfy.ldm.krea2.model as km
    import comfy.ops
    from comfy.ldm.flux.layers import EmbedND
    if not K._orig:
        K._orig.update({c.__name__: c.forward for c in (km.RMSNorm, km.SwiGLU, km.Attention, km.SingleStreamBlock, km.LastLayer)})
    torch.manual_seed(0)
    blk = km.SingleStreamBlock(512, 4, 4, False, 2, operations=comfy.ops.disable_weight_init)
    for p in blk.parameters():
        p.data.normal_(0, 0.05)
    blk = blk.to("mps", torch.bfloat16)
    x = torch.randn(1, 96, 512, device="mps", dtype=torch.bfloat16)
    vec = torch.randn(1, 1, 6 * 512, device="mps", dtype=torch.bfloat16) * 0.1
    freqs = EmbedND(dim=128, theta=1000, axes_dim=[56, 36, 36])(torch.rand(1, 96, 3, device="mps") * 10)
    with torch.inference_mode():
        stock = K._orig["SingleStreamBlock"](blk, x, vec, freqs).float()
        fast = K._block_forward(blk, x, vec, freqs).float()
    assert ((fast - stock).norm() / stock.norm()).item() < 0.01


@requires_mps
def test_single_frame_conv3d_matches_stock(comfy):
    """Includes comfy's causal_zero slice: a strided weight view at a storage offset, with
    3 output channels, which MPS conv2d gets wrong unless made contiguous."""
    from _patches import conv_im2col_mps as C
    C._orig_conv2d, C._orig_conv3d = F.conv2d, F.conv3d
    g = torch.Generator().manual_seed(0)
    for cout, kd, T in ((3, 3, 1), (16, 3, 3), (16, 1, 1)):
        x = torch.randn(1, 24, T, 40, 56, generator=g)
        w = torch.randn(cout, 24, 3, 3, 3, generator=g) * 0.1
        w = w[:, :, -T:] if kd == 3 else w[:, :, :1]
        b = torch.randn(cout, generator=g)
        ref = F.conv3d(x, w, b, 1, (0, 1, 1))
        out = C._single_frame_conv3d(x.to("mps"), w.to("mps"), b.to("mps"), 1, (0, 1, 1), 1, 1)
        assert _err(out, ref) < 1e-4, (cout, kd, T)
    # the real causal_zero layout: slice taken on-device so the view keeps its offset
    w = (torch.randn(3, 24, 3, 3, 3, generator=g) * 0.1).to("mps")[:, :, -1:]
    x = torch.randn(1, 24, 1, 40, 56, generator=g)
    ref = F.conv3d(x, w.cpu(), None, 1, (0, 1, 1))
    assert _err(C._single_frame_conv3d(x.to("mps"), w, None, 1, (0, 1, 1), 1, 1), ref) < 1e-4


@requires_mps
def test_wan_vae_channel_norm_matches_stock(comfy):
    from _patches import wan_vae_norm_mps as W
    from comfy.ldm.wan.vae import RMS_norm
    if W._orig_forward is None:
        W._orig_forward = RMS_norm.forward
    assert W._verify()
