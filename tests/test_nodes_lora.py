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


class _Patcher:
    """The injection API of ModelPatcher; clone() copies the lists, as ModelPatcher does."""
    def __init__(self, module, injections=None):
        self.module, self.injections = module, dict(injections or {})

    def clone(self):
        return _Patcher(self.module, {k: v.copy() for k, v in self.injections.items()})

    def get_injections(self, key):
        return self.injections.get(key)

    def set_injections(self, key, injections):
        self.injections[key] = injections

    def inject(self):
        for injs in self.injections.values():
            for i in injs:
                i.inject(self)

    def eject(self):   # ModelPatcher.eject_model walks the same order as inject
        for injs in self.injections.values():
            for i in injs:
                i.eject(self)


def test_chained_slots_stack_both_loras(node, monkeypatch):
    import torch
    from comfy.weight_adapter import BypassInjectionManager
    from comfy.model_management import get_torch_device
    from comfy.weight_adapter.lora import LoRAAdapter

    torch.manual_seed(0)
    dev = get_torch_device()   # inject() moves the adapter weights there
    lin = torch.nn.Linear(16, 8).to(dev)
    deltas = {}

    def stock(self, model, lora_name, strength):   # what the stock loader does to the patcher
        up, down = torch.randn(8, 4), torch.randn(4, 16)
        deltas[lora_name] = strength * (up @ down).to(dev)
        m = BypassInjectionManager()
        m.add_adapter("weight", LoRAAdapter(set(), (up, down, None, None, None, None)), strength)
        m._get_module_by_key = lambda model, key: lin
        out = model.clone()
        out.set_injections("bypass_lora", m.create_injections(lin))
        return (out,)

    monkeypatch.setattr(node.__mro__[1], "load_lora_model_only", stock)
    x = torch.randn(3, 16, device=dev)
    base = lin(x)
    p = node().load_lora_model_only(_Patcher(lin), "a", 1.0)[0]
    p = node().load_lora_model_only(p, "None", 1.0)[0]
    p = node().load_lora_model_only(p, "b", 0.5)[0]
    for _ in range(2):   # inject/eject twice: a wrong eject order leaves a dead hook behind
        p.inject()
        want = base + x @ (deltas["a"] + deltas["b"]).t()
        assert torch.allclose(lin(x), want, atol=1e-4)
        p.eject()
        assert torch.allclose(lin(x), base)
