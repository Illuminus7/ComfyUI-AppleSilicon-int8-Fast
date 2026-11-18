import types

import pytest
import torch

from conftest import requires_mps

from _patches import mps_te_offload

GIB = 2**30
MPS = torch.device("mps")


class _BaseModel:
    pass


model_base = types.SimpleNamespace(BaseModel=_BaseModel)


class _Patcher:
    def __init__(self, size, kind):
        self.is_clip = kind == "te"
        self.model = _BaseModel() if kind == "dit" else object()
        self.load_device = MPS
        self.size = size
        self.loaded = 0

    def model_size(self):
        return self.size

    def loaded_size(self):
        return self.loaded


class _Loaded:
    def __init__(self, patcher):
        self.model = patcher
        self.device = MPS
        self.unloads = 0

    def is_dead(self):
        return False

    def model_unload(self):
        self.unloads += 1
        self.model.loaded = 0
        return True


def _resident(size, kind):
    p = _Patcher(size, kind)
    p.loaded = size
    return p


@pytest.fixture
def mm(monkeypatch):
    monkeypatch.setattr(mps_te_offload, "_mode", "auto")
    monkeypatch.setattr(mps_te_offload, "_limit", 41 * GIB)
    monkeypatch.setattr(mps_te_offload, "_offloading", False)
    monkeypatch.setattr(mps_te_offload, "_ram", 48 * GIB)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    held = {"bytes": 18 * GIB}
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: held["bytes"])
    te, vae = _Loaded(_resident(15 * GIB, "te")), _Loaded(_resident(3 * GIB, "vae"))
    return types.SimpleNamespace(current_loaded_models=[te, vae], te=te, vae=vae, held=held)


@requires_mps
def test_video_sampling_offloads_the_encoder_and_helpers(mm):
    dit = _Patcher(20 * GIB, "dit")
    freed = mps_te_offload._before_load(mm, [dit], 133 * GIB, model_base)
    assert freed == 18 * GIB
    assert (mm.te.unloads, mm.vae.unloads) == (1, 1)
    assert mm.current_loaded_models == []


@requires_mps
def test_image_sampling_keeps_the_encoder(mm):
    dit = _Patcher(20 * GIB, "dit")             # 18 + 20 + 0.5 GiB fits under 41 GiB
    assert mps_te_offload._before_load(mm, [dit], GIB // 2, model_base) == 0
    assert mm.te.unloads == 0


@requires_mps
def test_next_encoder_load_unloads_the_diffusion_model_after_an_offload(mm):
    dit = _Patcher(20 * GIB, "dit")
    mps_te_offload._before_load(mm, [dit], 133 * GIB, model_base)
    dit.loaded = dit.size
    dit_lm = _Loaded(dit)
    mm.current_loaded_models[:] = [dit_lm]
    mm.held["bytes"] = 12 * GIB
    te = _Patcher(15 * GIB, "te")                # 12 + 15 GiB fits in 60% of 48: leave it
    assert mps_te_offload._before_load(mm, [te], 0, model_base) == 0
    mm.held["bytes"] = 22 * GIB                  # 22 + 15 GiB does not
    assert mps_te_offload._before_load(mm, [te], 0, model_base) == 20 * GIB
    assert dit_lm.unloads == 1


@requires_mps
def test_encoder_loads_are_left_alone_without_an_offload(mm):
    dit_lm = _Loaded(_resident(20 * GIB, "dit"))
    mm.current_loaded_models[:] = [dit_lm]
    mm.held["bytes"] = 30 * GIB
    assert mps_te_offload._before_load(mm, [_Patcher(15 * GIB, "te")], 0, model_base) == 0
    assert dit_lm.unloads == 0


@requires_mps
def test_nothing_to_offload_without_an_encoder(mm):
    mm.current_loaded_models[:] = [mm.vae]
    assert mps_te_offload._before_load(mm, [_Patcher(20 * GIB, "dit")], 133 * GIB, model_base) == 0
    assert mm.vae.unloads == 0
