import sys
import types

import pytest
import torch

from conftest import requires_mps

from _patches import mps_free_memory

GIB = 2**30


@pytest.fixture
def mm(monkeypatch):
    """A stand-in comfy.model_management reporting 40 GiB of free system memory."""
    mod = types.ModuleType("comfy.model_management")
    mod.get_torch_device = lambda: torch.device("mps")
    mod.get_free_memory = lambda dev=None, torch_free_too=False: (40 * GIB, 40 * GIB) if torch_free_too else 40 * GIB
    comfy = types.ModuleType("comfy")
    comfy.model_management = mod
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mod)
    monkeypatch.setattr(mps_free_memory, "_installed", False)
    monkeypatch.delenv("APPLESILICON_MPS_FREE_MEMORY", raising=False)
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 30 * GIB)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 25 * GIB)
    return mod


@requires_mps
def test_free_memory_is_the_room_left_under_the_allocator_limit(mm, monkeypatch):
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
    mps_free_memory.install()

    assert mm.get_free_memory(torch.device("mps")) == 5 * GIB
    assert mm.get_free_memory() == 5 * GIB                           # default device
    assert mm.get_free_memory(torch.device("mps"), torch_free_too=True) == (5 * GIB, 5 * GIB)


@requires_mps
def test_system_memory_still_wins_when_it_is_the_smaller(mm, monkeypatch):
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "2.0")    # limit 60 GiB, room 35
    mps_free_memory.install()

    assert mm.get_free_memory(torch.device("mps")) == 35 * GIB
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 10 * GIB)
    assert mm.get_free_memory(torch.device("mps")) == 40 * GIB


@requires_mps
def test_room_never_goes_negative_and_other_devices_are_untouched(mm, monkeypatch):
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 31 * GIB)
    mps_free_memory.install()

    assert mm.get_free_memory(torch.device("mps")) == 0
    assert mm.get_free_memory(torch.device("cpu")) == 40 * GIB


@requires_mps
@pytest.mark.parametrize("env", [{"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0"}, {"APPLESILICON_MPS_FREE_MEMORY": "off"}])
def test_not_installed_without_a_limit_or_when_switched_off(mm, monkeypatch, env):
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    orig = mm.get_free_memory

    mps_free_memory.install()

    assert mm.get_free_memory is orig
