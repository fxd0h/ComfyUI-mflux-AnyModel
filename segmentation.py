"""Interior auto-mask node for ComfyUI-mflux-AnyModel.

MfluxAutoMask segments a room photo with an ADE20K semantic model (SegFormer) and turns a
named region (floor / walls / ceiling / windows / doors / furniture / custom) into a mask.
That mask feeds the sampler's mask-preserve composite (connect it to `mflux Image` -> mask):
- mask_mode=inpaint on the sampler -> only the segmented region takes the edit ("restyle only the floor")
- mask_mode=preserve -> the segmented region is locked to the original ("keep the windows")

ADE20K natively has the interior surface/opening classes, so no hand-painting and no
Grounding-DINO/SAM prompt is needed. Runs locally on Apple Silicon (MPS) with a CPU fallback.
Segmentation is torch/transformers (already an mflux dependency), not an mflux model.
"""
import numpy as np
import torch

# Named regions -> label keywords matched against the model's ADE20K id2label at run time
# (case-insensitive substring). Resolving from the live label map keeps this robust to the
# exact class-name spelling a given checkpoint uses.
REGION_PRESETS = {
    "floor": ["floor", "flooring"],
    "walls": ["wall"],
    "ceiling": ["ceiling"],
    "windows": ["windowpane", "window"],
    "doors": ["door"],
    "openings (windows + doors)": ["windowpane", "window", "door"],
    "surfaces (floor + walls + ceiling)": ["floor", "flooring", "wall", "ceiling"],
    "furniture": ["sofa", "chair", "armchair", "table", "bed", "cabinet", "shelf",
                  "desk", "wardrobe", "chest", "stool", "bench", "ottoman"],
    "floor + rug": ["floor", "flooring", "rug", "carpet"],
    "custom (use custom_classes only)": [],
}

SEG_MODELS = [
    "nvidia/segformer-b2-finetuned-ade-512-512",
    "nvidia/segformer-b0-finetuned-ade-512-512",
    "nvidia/segformer-b4-finetuned-ade-512-512",
    "nvidia/segformer-b5-finetuned-ade-640-640",
]

_SEG_CACHE = {"repo": None, "model": None, "proc": None, "device": None}


def _device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load_segmenter(repo):
    if _SEG_CACHE["repo"] == repo and _SEG_CACHE["model"] is not None:
        return _SEG_CACHE["model"], _SEG_CACHE["proc"], _SEG_CACHE["device"]
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    proc = SegformerImageProcessor.from_pretrained(repo)
    model = SegformerForSemanticSegmentation.from_pretrained(repo).eval()
    dev = _device()
    try:
        model = model.to(dev)
    except Exception:
        dev = "cpu"
        model = model.to("cpu")
    _SEG_CACHE.update(repo=repo, model=model, proc=proc, device=dev)
    return model, proc, dev


def _resolve_ids(id2label, keywords):
    """ADE20K class ids whose label contains any keyword (case-insensitive)."""
    kws = [k.strip().lower() for k in keywords if k.strip()]
    ids = []
    for i, label in id2label.items():
        low = str(label).lower()
        if any(kw in low for kw in kws):
            ids.append(int(i))
    return ids


def _segment_mask(pil, repo, class_ids):
    """Return a float HxW mask in [0,1] (1 where any target class), at the image resolution."""
    import torch.nn.functional as F
    model, proc, dev = _load_segmenter(repo)
    inputs = proc(images=pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(dev)
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits          # (1, C, h/4, w/4)
    w, h = pil.size
    up = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
    seg = up.argmax(dim=1)[0].to("cpu").numpy()                    # (H, W) class ids
    if not class_ids:
        return np.zeros((h, w), dtype=np.float32), seg
    mask = np.isin(seg, np.asarray(class_ids)).astype(np.float32)
    return mask, seg


def _postprocess(mask, invert, dilate, feather):
    from PIL import Image, ImageFilter
    import cv2
    if dilate and dilate > 0:
        k = np.ones((int(dilate) * 2 + 1, int(dilate) * 2 + 1), np.uint8)
        mask = cv2.dilate((mask * 255).astype(np.uint8), k, iterations=1).astype(np.float32) / 255.0
    if invert:
        mask = 1.0 - mask
    if feather and feather > 0:
        pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L").filter(ImageFilter.GaussianBlur(float(feather)))
        mask = np.asarray(pil, dtype=np.float32) / 255.0
    return np.clip(mask, 0.0, 1.0)


class MfluxAutoMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "region": (list(REGION_PRESETS.keys()), {"default": "floor", "tooltip": "Which room region to select via ADE20K semantic segmentation. Feed the mask to 'mflux Image' -> mask; on the sampler pick mask_mode=inpaint to restyle only this region, or preserve to lock it."}),
            },
            "optional": {
                "custom_classes": ("STRING", {"default": "", "tooltip": "Comma-separated ADE20K label keywords to add (e.g. 'lamp, curtain, plant'). Matched case-insensitively against the model's class names."}),
                "invert": ("BOOLEAN", {"default": False, "tooltip": "Flip the mask (select everything EXCEPT the region)."}),
                "dilate": ("INT", {"default": 6, "min": 0, "max": 128, "tooltip": "Grow the mask outward (px) to cover class-boundary seams."}),
                "feather": ("INT", {"default": 4, "min": 0, "max": 128, "tooltip": "Soften the mask edge (px). The sampler's mask_feather can soften further."}),
                "model": (SEG_MODELS, {"default": SEG_MODELS[0]}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE", "STRING")
    RETURN_NAMES = ("mask", "mask_image", "info")
    FUNCTION = "segment"
    CATEGORY = "MLX/mflux/inputs"

    def segment(self, image, region, custom_classes="", invert=False, dilate=6, feather=4,
                model=SEG_MODELS[0]):
        from PIL import Image
        arr = (image[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr).convert("RGB")

        seg_model, _, _ = _load_segmenter(model)
        id2label = seg_model.config.id2label
        keywords = list(REGION_PRESETS.get(region, []))
        keywords += [c for c in custom_classes.split(",")]
        class_ids = _resolve_ids(id2label, keywords)
        matched = sorted({str(id2label[i]) for i in class_ids})

        mask, _seg = _segment_mask(pil, model, class_ids)
        coverage = float(mask.mean())
        mask = _postprocess(mask, invert, dilate, feather)

        mask_t = torch.from_numpy(mask).unsqueeze(0)                       # (1,H,W) MASK
        mask_img = torch.from_numpy(np.stack([mask] * 3, axis=-1)).unsqueeze(0)  # (1,H,W,3) IMAGE
        info = (f"region={region} classes={matched or 'NONE MATCHED'} "
                f"raw_coverage={coverage:.1%} invert={invert} dilate={dilate} feather={feather}")
        if not class_ids:
            info += "  [warning] no ADE20K class matched — mask is empty; check custom_classes."
        print("[mflux] AutoMask:", info)
        return (mask_t, mask_img, info)
