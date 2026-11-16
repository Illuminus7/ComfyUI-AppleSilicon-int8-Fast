import gc
import sys
import types

import pytest
import torch

from conftest import requires_mps

from _patches import mps_orphan_gc

GIB = 2**30


class _Loaded:
    def __init__(self, size):
        self.device = torch.device("mps")
        self.model = types.SimpleNamespace(loaded_size=lambda: size)


@pytest.fixture
def mm(monkeypatch):
    """A stand-in comfy.model_management with one 10 GiB model loaded."""
    mod = types.ModuleType("comfy.model_management")
    mod.current_loaded_models = [_Loaded(10 * GIB)]
    mod.calls = []
    mod.free_memory = lambda memory_required, device, *a, **k: mod.calls.append(memory_required) or []
    comfy = types.ModuleType("comfy")
    comfy.model_management = mod
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mod)
    monkeypatch.setattr(mps_orphan_gc, "_installed", False)
    monkeypatch.setattr(mps_orphan_gc, "_baseline", 0)
    monkeypatch.delenv("APPLESILICON_MPS_ORPHAN_GC", raising=False)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    return mod


def _fake_gc(monkeypatch, held, after):
    """MPS holds `held` GiB until a collection, then `after`."""
    state = {"held": held * GIB, "collections": 0}
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: state["held"])

    def collect():
        state["collections"] += 1
        state["held"] = after * GIB
        return 0
    monkeypatch.setattr(gc, "collect", collect)
    return state


@requires_mps
def test_orphaned_weights_are_collected_before_a_load(mm, monkeypatch):
    state = _fake_gc(monkeypatch, held=30, after=10)       # 20 GiB beyond the loaded model
    mps_orphan_gc.install()

    mm.free_memory(5 * GIB, torch.device("mps"))

    assert state["collections"] == 1
    assert mm.calls == [5 * GIB]                           # ComfyUI's own free_memory still runs


@requires_mps
def test_no_collection_when_memory_is_accounted_for(mm, monkeypatch):
    state = _fake_gc(monkeypatch, held=11, after=11)       # 1 GiB of activations: under the bar
    mps_orphan_gc.install()

    mm.free_memory(5 * GIB, torch.device("mps"))

    assert state["collections"] == 0


@requires_mps
def test_memory_a_collection_cannot_free_is_not_collected_for_again(mm, monkeypatch):
    state = _fake_gc(monkeypatch, held=15, after=15)       # 5 GiB held by something live
    mps_orphan_gc.install()

    mm.free_memory(GIB, torch.device("mps"))
    mm.free_memory(GIB, torch.device("mps"))

    assert state["collections"] == 1
    state["held"] = 18 * GIB                               # another 3 GiB on top: collect again
    mm.free_memory(GIB, torch.device("mps"))
    assert state["collections"] == 2


@requires_mps
def test_other_devices_and_the_off_switch(mm, monkeypatch):
    state = _fake_gc(monkeypatch, held=30, after=10)
    mps_orphan_gc.install()
    mm.free_memory(GIB, torch.device("cpu"))
    assert state["collections"] == 0

    monkeypatch.setattr(mps_orphan_gc, "_installed", False)
    monkeypatch.setenv("APPLESILICON_MPS_ORPHAN_GC", "off")
    plain = types.ModuleType("comfy.model_management")
    plain.free_memory = orig = lambda *a, **k: []
    plain.current_loaded_models = []
    monkeypatch.setitem(sys.modules, "comfy.model_management", plain)
    sys.modules["comfy"].model_management = plain
    mps_orphan_gc.install()
    assert plain.free_memory is orig
