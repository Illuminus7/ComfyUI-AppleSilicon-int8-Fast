"""An optional LoRA slot: ComfyUI's bypass LoRA loader with "None" at the top of the list,
so a workflow can keep the slot in place and still queue with no LoRA selected."""

import folder_paths
from comfy_extras.nodes_lora_debug import LoraLoaderBypassModelOnly


class LoraLoaderBypassModelOnlyOptional(LoraLoaderBypassModelOnly):
    DESCRIPTION = "Apply a LoRA in bypass mode, or pass the model through unchanged when lora_name is None."
    EXPERIMENTAL = False

    @classmethod
    def INPUT_TYPES(s):
        types = super().INPUT_TYPES()
        types["required"]["lora_name"] = (["None"] + folder_paths.get_filename_list("loras"), )
        return types

    def load_lora_model_only(self, model, lora_name, strength_model):
        if lora_name == "None":
            return (model,)
        return super().load_lora_model_only(model, lora_name, strength_model)


NODE_CLASS_MAPPINGS = {"LoraLoaderBypassModelOnlyOptional": LoraLoaderBypassModelOnlyOptional}
NODE_DISPLAY_NAME_MAPPINGS = {"LoraLoaderBypassModelOnlyOptional": "Load LoRA (Bypass, Optional)"}
