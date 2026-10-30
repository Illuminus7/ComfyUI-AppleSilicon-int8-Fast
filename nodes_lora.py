"""An optional LoRA slot: ComfyUI's bypass LoRA loader with "None" at the top of the list,
so a workflow can keep the slot in place and still queue with no LoRA selected.

Slots can be chained. The stock loader stores its hooks under one injection key, so a
second bypass LoRA would replace the first; here a second one is stacked on the first
(injected after it, ejected before it, since each hook wraps the forward it found)."""

import folder_paths
from comfy.patcher_extension import PatcherInjection
from comfy_extras.nodes_lora_debug import LoraLoaderBypassModelOnly

_KEY = "bypass_lora"


def _stacked(inner, outer):
    def inject(model_patcher):
        for i in inner:
            i.inject(model_patcher)
        for i in outer:
            i.inject(model_patcher)

    def eject(model_patcher):
        for i in reversed(outer):
            i.eject(model_patcher)
        for i in reversed(inner):
            i.eject(model_patcher)

    return PatcherInjection(inject=inject, eject=eject)


class LoraLoaderBypassModelOnlyOptional(LoraLoaderBypassModelOnly):
    DESCRIPTION = ("Apply a LoRA in bypass mode, or pass the model through unchanged when lora_name is None. "
                   "Chained loaders stack their LoRAs.")
    EXPERIMENTAL = False

    @classmethod
    def INPUT_TYPES(s):
        types = super().INPUT_TYPES()
        types["required"]["lora_name"] = (["None"] + folder_paths.get_filename_list("loras"), )
        return types

    def load_lora_model_only(self, model, lora_name, strength_model):
        if lora_name == "None":
            return (model,)
        prev = model.get_injections(_KEY) if hasattr(model, "get_injections") else None
        out = super().load_lora_model_only(model, lora_name, strength_model)
        new = out[0].get_injections(_KEY) if prev else None
        if new and [id(i) for i in new] != [id(i) for i in prev]:
            out[0].set_injections(_KEY, [_stacked(prev, new)])
        return out


NODE_CLASS_MAPPINGS = {"LoraLoaderBypassModelOnlyOptional": LoraLoaderBypassModelOnlyOptional}
NODE_DISPLAY_NAME_MAPPINGS = {"LoraLoaderBypassModelOnlyOptional": "Load LoRA (Bypass, Optional)"}
