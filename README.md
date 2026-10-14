This custom node is based on the original work of Paweł Mazurkiewicz ([ComfyUI-AppleSilicon-FP8](https://github.com/pawel-mazurkiewicz/ComfyUI-AppleSilicon-FP8)).

Proudly vibe-coded using Claude.

# ComfyUI-AppleSilicon-int8-Fast

Fast int8 support in ComfyUI on Apple Silicon.

Copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI.

Alternative: Open the terminal, cd to your custom nodes folder, type git clone https://github.com/Illuminus7/ComfyUI-AppleSilicon-int8-Fast

The kernels are tuned on an M5 Max. Smaller M5 chips run the same kernels with fewer GPU cores; they should get the same relative speedups, but the large models (MiniMax H3 needs about 40 GB) won't fit in their memory.

This was tested with ComfyUI v0.40 and PyTorch 2.13. Future updates may break compatibility.

No startup arguments are needed (Comfy Desktop's default `--enable-manager` is fine). Memory-related arguments are not recommended.

Sample workflows for Krea 2, MiniMax H3 and LTX 2.5 are on Hugging Face: [Illuminus7/Comfy_Int8-Fast_Workflows](https://huggingface.co/Illuminus7/Comfy_Int8-Fast_Workflows/tree/main). Using the exact models in the workflows is recommended. Put each file in the listed folder under `ComfyUI/models/`. The image-to-video workflows need a start image of your own in the Load Image node.

### Krea 2

`krea2_turbo_t2i.json` needs [rgthree-comfy](https://github.com/rgthree/rgthree-comfy), installable from ComfyUI Manager. Its LoRA stack is empty by default.

| File | Folder | Size | Download |
|---|---|---|---|
| `Krea2_Turbo_convrot_int8mixed.safetensors` | `diffusion_models` | 12.9 GB | [Winnougan/Krea-2-Base-Turbo-NVFP4-FP8-INT8](https://huggingface.co/Winnougan/Krea-2-Base-Turbo-NVFP4-FP8-INT8/resolve/main/Krea2_Turbo_convrot_int8mixed.safetensors) |
| `qwen3vl_4b_int8.safetensors` | `text_encoders` | 4.8 GB | [Winnougan/Krea-2-Base-Turbo-NVFP4-FP8-INT8](https://huggingface.co/Winnougan/Krea-2-Base-Turbo-NVFP4-FP8-INT8/resolve/main/Text%20Encoder/qwen3vl_4b_int8.safetensors) |
| `wan_2.1_vae.safetensors` | `vae` | 0.25 GB | [Comfy-Org/Wan_2.1_ComfyUI_repackaged](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors) |

### MiniMax H3

`minimax_h3_t2v_mac.json` (text-to-video) and `minimax_h3_i2v_mac.json` (image-to-video) share the models and the turbo LoRA. MiniMax H3 needs about 40 GB of memory.

| File | Folder | Size | Download |
|---|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `diffusion_models` | 21.0 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) |
| `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `text_encoders` | 15.7 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) |
| `minimax_h3_video_vae_int8_convrot.safetensors` | `vae` | 2.8 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors) |
| `minimax_h3_audio_vae_fp32.safetensors` | `vae` | 0.6 GB | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) |
| `minimax_h3_turbo_v4_step600_pruned_comfyui.safetensors` (turbo LoRA) | `loras` | 0.6 GB | [drbaph/MiniMax-H3-Turbo-Lora-ComfyUI](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/resolve/main/minimax_h3_turbo_v4_step600_pruned_comfyui.safetensors) |

### LTX 2.5

`ltx2_5_t2v_mac.json` (text-to-video) and `ltx2_5_i2v_mac.json` (image-to-video) share the models. Their LoRA slot starts bypassed so they queue without one: pick a LoRA on the main node, then enable the LoRA node inside the subgraph (select it, Ctrl+B). The Lightricks repo is gated: accept its license on Hugging Face before downloading.

| File | Folder | Size | Download |
|---|---|---|---|
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | `diffusion_models` | 21.5 GB | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors) |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | `text_encoders` | 15.4 GB | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors) |
| `gemma4_e2b_it_int8_convrot.safetensors` (prompt enhancer; the workflows won't queue without it, even with it off) | `text_encoders` | 5.2 GB | [Comfy-Org/gemma-4](https://huggingface.co/Comfy-Org/gemma-4/resolve/main/text_encoders/gemma4_e2b_it_int8_convrot.safetensors) |
| `ltx-2.5-video-vae-bf16.safetensors` | `vae` | 1.5 GB | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-bf16.safetensors) |
| `ltx-2.5-audio-vae-bf16.safetensors` | `vae` | 0.4 GB | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors) |
| `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | `latent_upscale_models` | 1.0 GB | [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5/resolve/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors) |

## License

MIT; see [LICENSE](LICENSE).
