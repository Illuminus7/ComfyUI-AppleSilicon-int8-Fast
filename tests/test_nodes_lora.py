"""The optional bypass LoRA loader: "None" is listed first and passes the model through."""
import os
import sys

import pytest

COMFY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@pytest.fixture(scope="module")
def node():
    if COMFY not in sys.path:
        sys.path.insert(0, COMFY)
    try:
        import comfy.options
        comfy.options.enable_args_parsing(False)
        import nodes_lora
    except Exception as e:
        pytest.skip(f"ComfyUI not importable: {e!r}")
    return nodes_lora.LoraLoaderBypassModelOnlyOptional


def test_none_is_the_first_choice(node):
    assert node.INPUT_TYPES()["required"]["lora_name"][0][0] == "None"


def test_none_passes_the_model_through(node, monkeypatch):
    calls = []
    monkeypatch.setattr(node.__mro__[1], "load_lora_model_only", lambda self, *a: calls.append(a) or ("patched",))
    model = object()
    assert node().load_lora_model_only(model, "None", 1.0) == (model,)
    assert node().load_lora_model_only(model, "some.safetensors", 1.0) == ("patched",)
    assert calls == [(model, "some.safetensors", 1.0)]
