from .nodes import (
    MfluxModelLoader,
    MfluxModelSampler,
    MfluxLora,
    MfluxImage,
    MfluxUpscale,
)
from .segmentation import MfluxAutoMask
from .depth_map import MfluxDepthMap
from .vlm import MfluxVLM

NODE_CLASS_MAPPINGS = {
    "MfluxModelLoader": MfluxModelLoader,
    "MfluxModelSampler": MfluxModelSampler,
    "MfluxLora": MfluxLora,
    "MfluxImage": MfluxImage,
    "MfluxUpscale": MfluxUpscale,
    "MfluxAutoMask": MfluxAutoMask,
    "MfluxDepthMap": MfluxDepthMap,
    "MfluxVLM": MfluxVLM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MfluxModelLoader": "mflux Model Loader (MLX)",
    "MfluxModelSampler": "mflux Sampler (MLX)",
    "MfluxLora": "mflux LoRA",
    "MfluxImage": "mflux Image",
    "MfluxUpscale": "mflux Upscale (SeedVR2)",
    "MfluxAutoMask": "mflux Auto Mask (interior segmentation)",
    "MfluxDepthMap": "mflux Depth Map (DepthPro)",
    "MfluxVLM": "mflux VLM (FIBO-vlm, room brief)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
