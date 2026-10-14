"""Tests for the fused neighborhood attention (comfy_kitchen.na3d) kernel on MPS, against
comfy_kitchen's eager implementation."""
import pytest
import torch

from conftest import requires_mps

from _patches import na3d_mps as N


@pytest.fixture(scope="module")
def ref_na3d():
    eager = pytest.importorskip("comfy_kitchen.backends.eager.na")
    N._orig = eager.na3d
    return eager.na3d


def _err(a, ref):
    a, ref = a.cpu().float(), ref.cpu().float()
    return ((a - ref).norm() / ref.norm()).item()


@requires_mps
def test_shader_verifies(ref_na3d):
    assert N._verify()


@requires_mps
@pytest.mark.parametrize("shape,ks", [
    ((1, 9, 13, 10, 3, 64), (3, 7, 7)),       # ragged bricks on every axis
    ((2, 4, 6, 5, 2, 64), (11, 11, 11)),      # kernel clamped to every axis
    ((1, 12, 20, 18, 1, 64), (11, 11, 11)),   # interior windows, union larger than one key block
])
def test_matches_eager(ref_na3d, shape, ks):
    assert N._verify()
    g = torch.Generator().manual_seed(1)
    q, k, v = (torch.randn(shape, generator=g).to(torch.bfloat16) for _ in range(3))
    ref = ref_na3d(q.float(), k.float(), v.float(), list(ks), None, 0.125)
    out = N._na3d(q.to("mps"), k.to("mps"), v.to("mps"), list(ks), None, 0.125)
    assert out.shape == shape and _err(out, ref) < 0.01


@requires_mps
def test_qkv_views_of_one_buffer(ref_na3d):
    """q/k/v as strided views of a fused qkv buffer (the LTX decoder's layout); output contiguous."""
    assert N._verify()
    g = torch.Generator().manual_seed(2)
    qkv = torch.randn(1, 5, 9, 11, 3, 2, 64, generator=g).to(torch.bfloat16)
    q, k, v = qkv.unbind(4)
    ref = ref_na3d(q.float(), k.float(), v.float(), [3, 5, 5], None, 0.125)
    out = N._na3d(*qkv.to("mps").unbind(4), [3, 5, 5], None, 0.125)
    assert out.is_contiguous() and _err(out, ref) < 0.01


@requires_mps
def test_unsupported_calls_take_the_original(ref_na3d, monkeypatch):
    calls = []
    monkeypatch.setattr(N, "_orig", lambda *a: calls.append(1) or a[0])
    x = torch.randn(1, 3, 4, 4, 2, 64, device="mps", dtype=torch.bfloat16)
    N._na3d(x, x, x, [3, 3, 3], [True, False, False], None)               # causal
    y = torch.randn(1, 3, 4, 4, 2, 32, device="mps", dtype=torch.bfloat16)
    N._na3d(y, y, y, [3, 3, 3], None, None)                                # head_dim 32
    N._na3d(x.float(), x.float(), x.float(), [3, 3, 3], None, None)        # fp32
    assert calls == [1, 1, 1]
