"""ComfyUI prestartup hook: cap the MPS allocator watermarks before torch is imported.

torch's defaults (low=1.4, high=1.7 x recommended_max) exceed physical RAM on unified
memory, so the reserved pool spills into swap and the OS jetsam-kills the process.

high=1.1 is measured on a 48 GB M5 Max with MiniMax H3 (15 GB text encoder + 20 GB model):
at 1.0 a 4096x4096 image slows down or fails after the first run because the allocator
holds 1-2 GiB more than tensors use; at 1.2 sampling is slower; with no limit Metal itself
runs out.
"""
import os

if os.environ.get("APPLESILICON_FP8_MPS_WATERMARK", "auto").lower() not in ("off", "0", "false"):
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.9")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.1")
    print("[AppleSilicon-int8-Fast/prestartup] MPS allocator watermark capped "
          "(low=%s, high=%s) to keep the reserved pool resident instead of "
          "spilling into swap; set APPLESILICON_FP8_MPS_WATERMARK=off to disable."
          % (os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"],
             os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"]), flush=True)
