import gc
import types

import pytest
import torch

from _patches import mps_fast_free

MPS = types.SimpleNamespace(type="mps")
CPU = types.SimpleNamespace(type="cpu")


class _Patcher:
    def __init__(self):
        self.detached = 0

    def detach(self, unpatch_all=True):
        self.detached += 1


class _Finalizer:
    def __init__(self):
        self.detached = False

    def detach(self):
        self.detached = True


class _Loaded:
    def __init__(self, patcher, device=MPS):
        self.model = patcher
        self.device = device
        self.model_finalizer = _Finalizer()


@pytest.fixture(autouse=True)
def _clean():
    mps_fast_free._dropped.clear()
    yield
    mps_fast_free._dropped.clear()


def test_drop_stops_tracking_without_unloading():
    p = _Patcher()
    lm = _Loaded(p)
    fin = lm.model_finalizer
    mm = types.SimpleNamespace(current_loaded_models=[lm])
    assert mps_fast_free._drop(mm)
    assert mm.current_loaded_models == []
    assert fin.detached and p.detached == 0
    del lm


def test_a_model_freed_by_the_cache_reset_is_not_touched():
    p = _Patcher()
    mm = types.SimpleNamespace(current_loaded_models=[_Loaded(p)])
    mps_fast_free._drop(mm)
    del p                      # the cache reset dropped the last reference
    gc.collect()
    assert mps_fast_free._finish() == 0


def test_a_model_still_held_is_unloaded_normally():
    p = _Patcher()
    mm = types.SimpleNamespace(current_loaded_models=[_Loaded(p)])
    mps_fast_free._drop(mm)
    gc.collect()
    assert mps_fast_free._finish() == 1
    assert p.detached == 1
    assert mps_fast_free._dropped == []


def test_other_devices_keep_the_normal_unload():
    lm = _Loaded(_Patcher(), device=CPU)
    mm = types.SimpleNamespace(current_loaded_models=[lm])
    assert not mps_fast_free._drop(mm)
    assert mm.current_loaded_models == [lm]
    assert not mps_fast_free._drop(types.SimpleNamespace(current_loaded_models=[]))
