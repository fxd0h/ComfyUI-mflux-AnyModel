from .nodes import (
    MfluxModelLoader,
    MfluxModelSampler,
    MfluxLora,
    MfluxImage,
    MfluxUpscale,
)

NODE_CLASS_MAPPINGS = {
    "MfluxModelLoader": MfluxModelLoader,
    "MfluxModelSampler": MfluxModelSampler,
    "MfluxLora": MfluxLora,
    "MfluxImage": MfluxImage,
    "MfluxUpscale": MfluxUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MfluxModelLoader": "mflux Model Loader (MLX)",
    "MfluxModelSampler": "mflux Sampler (MLX)",
    "MfluxLora": "mflux LoRA",
    "MfluxImage": "mflux Image",
    "MfluxUpscale": "mflux Upscale (SeedVR2)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
