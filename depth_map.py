"""DepthPro preprocessor node for ComfyUI-mflux-AnyModel.

MfluxDepthMap turns a photo into a depth map with Apple's DepthPro, natively in MLX, so a
depth-guided workflow is self-contained: no need to produce the depth map outside ComfyUI.

Feed the result to a depth ControlNet (e.g. Shakker-Labs Union-Pro-2.0 via the loader's
controlnet_path) or to `mflux Image` -> map_image for `krea-2-depth` / `dev-depth`.

DepthPro emits near=white, which is the convention the depth models here are trained on, so the
map is passed through as produced (inverting it is a classic and silent way to get wrong output).
"""
import numpy as np
import torch

_DEPTH_CACHE = {"quantize": None, "model": None}


def _load_depth_pro(quantize):
    if _DEPTH_CACHE["model"] is not None and _DEPTH_CACHE["quantize"] == quantize:
        return _DEPTH_CACHE["model"]
    from mflux.models.depth_pro.model.depth_pro import DepthPro
    model = DepthPro(quantize=quantize)
    _DEPTH_CACHE.update(quantize=quantize, model=model)
    return model


class MfluxDepthMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "quantize": (["none", "4", "8"], {"default": "none", "tooltip": "Quantize DepthPro itself. 'none' is the usual choice: the model is small next to the image model it feeds."}),
                "keep_loaded": ("BOOLEAN", {"default": True, "tooltip": "Keep DepthPro resident between runs."}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Flip near/far. Leave OFF: DepthPro emits near=white, which is what the depth models here expect. Only turn it on for a control model trained the other way round."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_map",)
    FUNCTION = "estimate"
    CATEGORY = "MLX/mflux/inputs"

    def estimate(self, image, quantize="none", keep_loaded=True, invert=False):
        import os
        import tempfile

        from PIL import Image

        q = None if quantize == "none" else int(quantize)
        arr = (image[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        fd, path = tempfile.mkstemp(suffix=".png", prefix="mflux_depth_in_")
        os.close(fd)
        try:
            Image.fromarray(arr).save(path)
            model = _load_depth_pro(q)
            result = model.create_depth_map(image_path=path)
        finally:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        if not keep_loaded:
            _DEPTH_CACHE.update(quantize=None, model=None)

        depth = np.asarray(result.depth_image.convert("RGB"), dtype=np.float32) / 255.0
        if invert:
            depth = 1.0 - depth
        print(f"[mflux] DepthMap: {depth.shape[1]}x{depth.shape[0]} "
              f"min={result.min_depth:.2f}m max={result.max_depth:.2f}m invert={invert}")
        return (torch.from_numpy(np.clip(depth, 0.0, 1.0)).unsqueeze(0),)
