from .nodes import (
    MfluxModelLoader,
    MfluxModelSampler,
    MfluxLora,
    MfluxImage,
    MfluxUpscale,
)
from .segmentation import MfluxAutoMask

NODE_CLASS_MAPPINGS = {
    "MfluxModelLoader": MfluxModelLoader,
    "MfluxModelSampler": MfluxModelSampler,
    "MfluxLora": MfluxLora,
    "MfluxImage": MfluxImage,
    "MfluxUpscale": MfluxUpscale,
    "MfluxAutoMask": MfluxAutoMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MfluxModelLoader": "mflux Model Loader (MLX)",
    "MfluxModelSampler": "mflux Sampler (MLX)",
    "MfluxLora": "mflux LoRA",
    "MfluxImage": "mflux Image",
    "MfluxUpscale": "mflux Upscale (SeedVR2)",
    "MfluxAutoMask": "mflux Auto Mask (interior segmentation)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
