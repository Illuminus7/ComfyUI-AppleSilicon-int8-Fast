import ctypes

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from conftest import requires_mps

from _patches import mmap_load_mps

MINCORE_ANONYMOUS = 0x80


@pytest.fixture(autouse=True)
def _installed(monkeypatch):
    """Install the patch onto a pristine torch.Tensor, restored after each test."""
    monkeypatch.setattr(mmap_load_mps, "_installed", False)
    monkeypatch.setattr(torch.Tensor, "to", torch.Tensor.to)
    monkeypatch.setattr(mmap_load_mps, "stats", {"fast_bytes": 0, "fast_tensors": 0, "fallback_tensors": 0, "returned_tensors": 0})
    monkeypatch.setattr(mmap_load_mps, "_sources", {})
    monkeypatch.setattr(mmap_load_mps, "_decisions", {})
    monkeypatch.setenv("APPLESILICON_MMAP_LOAD", "on")       # the tests' files are tiny
    mmap_load_mps.install()


def _mapped(tmp_path, tensors):
    """The tensors as safetensors hands them to ComfyUI: views into a private file mapping."""
    path = tmp_path / "model.safetensors"
    save_file(tensors, str(path))
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return {k: f.get_tensor(k) for k in f.keys()}


def _anonymous_pages(t):
    page = mmap_load_mps._PAGE
    ptr = t.data_ptr()
    start = ptr - ptr % page
    length = -(-(ptr + t.numel() * t.element_size() - start) // page) * page
    vec = ctypes.create_string_buffer(length // page)
    assert mmap_load_mps._libc.mincore(start, length, vec) == 0
    return sum(1 for b in vec.raw if b & MINCORE_ANONYMOUS)


@requires_mps
@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.float16, torch.bfloat16, torch.float32, torch.float8_e4m3fn])
def test_mapped_tensor_moves_with_identical_bytes(tmp_path, dtype):
    src = torch.randint(0, 255, (300, 1000), dtype=torch.uint8).view(dtype)
    sd = _mapped(tmp_path, {"pad": torch.zeros(7, dtype=torch.uint8), "w": src})

    got = sd["w"].to("mps")

    assert mmap_load_mps.stats["fast_tensors"] == 1
    assert got.device.type == "mps" and got.dtype == dtype and got.shape == src.shape
    assert torch.equal(got.cpu().view(torch.uint8), src.view(torch.uint8))


@requires_mps
def test_mapped_pages_are_not_duplicated(tmp_path):
    """torch's own move turns every page of the source into anonymous memory; ours none."""
    sd = _mapped(tmp_path, {"w": torch.randint(0, 255, (4 << 20,), dtype=torch.uint8)})

    sd["w"].to("mps")
    torch.mps.synchronize()

    assert _anonymous_pages(sd["w"]) == 0


@requires_mps
def test_tensor_larger_than_scratch_moves_in_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(mmap_load_mps, "_SCRATCH_MAX", 100_000)
    src = torch.randn(50_000, 7)
    sd = _mapped(tmp_path, {"w": src})

    got = sd["w"].to(device="mps")

    assert mmap_load_mps.stats["fast_tensors"] == 1
    assert torch.equal(got.cpu(), src)


@requires_mps
def test_parameter_and_keyword_forms_take_the_fast_path(tmp_path):
    src = torch.randn(64, 64)
    sd = _mapped(tmp_path, {"w": src})

    a = torch.nn.Parameter(sd["w"], requires_grad=False).to(torch.device("mps"), non_blocking=True)
    b = sd["w"].to(device="mps", dtype=torch.float32, copy=True)

    assert mmap_load_mps.stats["fast_tensors"] == 2
    assert torch.equal(a.cpu(), src) and torch.equal(b.cpu(), src)


@requires_mps
def test_tensor_modified_in_memory_keeps_its_values(tmp_path):
    """A mapped tensor written to in place no longer equals the file, so torch moves it."""
    sd = _mapped(tmp_path, {"w": torch.zeros(100_000)})
    sd["w"].add_(3.0)

    got = sd["w"].to("mps")

    assert mmap_load_mps.stats["fast_tensors"] == 0
    assert torch.equal(got.cpu(), torch.full((100_000,), 3.0))


@requires_mps
def test_ordinary_and_converting_moves_are_left_to_torch(tmp_path):
    sd = _mapped(tmp_path, {"w": torch.randn(64, 64)})
    plain = torch.randn(64, 64)

    assert torch.equal(plain.to("mps").cpu(), plain)                    # not file-backed
    assert sd["w"].to("mps", torch.float16).dtype == torch.float16      # dtype change
    assert sd["w"][:, ::2].to("mps").shape == (64, 32)                  # not contiguous
    assert sd["w"].to(torch.float64).device.type == "cpu"               # no device move

    assert mmap_load_mps.stats["fast_tensors"] == 0


@requires_mps
def test_unchanged_weight_returns_to_cpu_as_the_mapped_tensor(tmp_path):
    src = torch.randn(300, 1000)
    sd = _mapped(tmp_path, {"w": src})
    on_mps = torch.nn.Parameter(sd["w"].to("mps"), requires_grad=False)

    back = on_mps.to(torch.device("cpu"), None, False)      # the form nn.Module.to uses

    assert mmap_load_mps.stats["returned_tensors"] == 1
    assert back.data_ptr() == sd["w"].data_ptr()            # no copy was made
    assert torch.equal(back, src)
    assert _anonymous_pages(back) == 0


@requires_mps
def test_weight_changed_on_mps_is_copied_back_with_its_new_values(tmp_path):
    src = torch.randn(300, 1000)
    sd = _mapped(tmp_path, {"w": src})
    on_mps = sd["w"].to("mps")
    on_mps.data.mul_(2.0)

    back = on_mps.to("cpu")

    assert mmap_load_mps.stats["returned_tensors"] == 0
    assert torch.equal(back, src * 2.0)


@requires_mps
def test_mapped_tensor_is_handed_back_only_once(tmp_path):
    """The caller may write to what it got, so a second request gets its own copy."""
    src = torch.randn(300, 1000)
    sd = _mapped(tmp_path, {"w": src})
    on_mps = sd["w"].to("mps")

    first = on_mps.to("cpu")
    first.zero_()
    second = on_mps.to("cpu")

    assert mmap_load_mps.stats["returned_tensors"] == 1
    assert torch.equal(second, src)


@requires_mps
def test_views_copies_and_conversions_are_copied_back_by_torch(tmp_path):
    src = torch.randn(300, 1000)
    sd = _mapped(tmp_path, {"w": src})
    on_mps = sd["w"].to("mps")

    assert torch.equal(on_mps[:10].to("cpu"), src[:10])                         # part of it
    assert on_mps.to("cpu", torch.float16).dtype == torch.float16               # dtype change
    assert torch.equal(on_mps.to("cpu", copy=True), src)                        # explicit copy

    assert mmap_load_mps.stats["returned_tensors"] == 0


@requires_mps
def test_returned_weight_reloads_by_the_fast_path(tmp_path):
    src = torch.randn(300, 1000)
    sd = _mapped(tmp_path, {"w": src})

    again = sd["w"].to("mps").to("cpu").to("mps")

    assert mmap_load_mps.stats["fast_tensors"] == 2
    assert torch.equal(again.cpu(), src)


@requires_mps
def test_small_model_on_a_roomy_machine_is_left_to_torch(tmp_path, monkeypatch):
    monkeypatch.setattr(mmap_load_mps, "_mode", "auto")
    src = torch.randn(300, 1000)
    sd = _mapped(tmp_path, {"w": src})

    got = sd["w"].to("mps")

    assert mmap_load_mps.stats["fast_tensors"] == 0
    assert torch.equal(got.cpu(), src)


@requires_mps
@pytest.mark.parametrize("ram_gb,held_gb,file_gb,want", [
    (48, 0, 15, True),      # 30 > 28.8: the text encoder alone on a 48 GB Mac
    (48, 15, 20, True),     # the diffusion model after it
    (48, 0, 12, False),     # 24 < 28.8
    (128, 15, 20, False),   # 55 < 76.8: the same pair on a 128 GB Mac
    (64, 15, 20, True),     # 55 > 38.4
])
def test_tight_compares_torchs_peak_with_sixty_percent_of_ram(monkeypatch, ram_gb, held_gb, file_gb, want):
    monkeypatch.setattr(mmap_load_mps, "_mode", "auto")
    monkeypatch.setattr(mmap_load_mps, "_ram", ram_gb << 30)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: held_gb << 30)

    assert mmap_load_mps._tight((1, 1), file_gb << 30) is want


@requires_mps
def test_decision_holds_for_the_rest_of_a_files_load(monkeypatch):
    """MPS fills as a file's weights arrive; the file must not change paths halfway."""
    monkeypatch.setattr(mmap_load_mps, "_mode", "auto")
    monkeypatch.setattr(mmap_load_mps, "_ram", 128 << 30)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 0)
    assert mmap_load_mps._tight((1, 1), 20 << 30) is False

    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 100 << 30)

    assert mmap_load_mps._tight((1, 1), 20 << 30) is False       # same file, same load
    assert mmap_load_mps._tight((1, 2), 20 << 30) is True        # another file


@requires_mps
def test_env_switch_disables_the_patch(monkeypatch):
    monkeypatch.setattr(mmap_load_mps, "_installed", False)
    monkeypatch.setattr(torch.Tensor, "to", mmap_load_mps._orig_to)
    monkeypatch.setenv("APPLESILICON_MMAP_LOAD", "off")

    mmap_load_mps.install()

    assert torch.Tensor.to is mmap_load_mps._orig_to
