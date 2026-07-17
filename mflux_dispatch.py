"""
Dispatch layer for the flexible mflux ComfyUI node.

Maps a model alias / HF repo / local path to (1) the correct mflux *variant* class,
(2) a resolved ModelConfig, (3) the load path, and constructs the model adapting to
each class's real constructor.

Ground-truthed by introspecting the installed mflux fork (fxd0h/mflux @ vfx1,
base 0.18.0) on 2026-06-08 — NOT derived from save.py (which only picks base
families and would silently route every flux variant to plain Flux1 txt2img).

Verified constructor shapes:
  - most:            (quantize, model_path, lora_paths, lora_scales, bake_lora, model_config)
  - Flux1Controlnet: + controlnet_path
  - Ideogram4:       (quantize, model_path, model_config, lora_paths, lora_scales)   # no bake_lora
  - FIBO / Ernie:    (quantize, model_path, lora_paths, lora_scales, model_config)   # no bake_lora
  - SeedVR2:         (quantize, model_path, model_config)                            # no lora (own node)
"""
import inspect

from mflux.models.common.config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
from mflux.models.flux.variants.fill.flux_fill import Flux1Fill
from mflux.models.flux.variants.depth.flux_depth import Flux1Depth
from mflux.models.flux.variants.redux.flux_redux import Flux1Redux
from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
from mflux.models.flux.variants.in_context.flux_in_context_fill import Flux1InContextFill
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.models.z_image.variants.z_image import ZImage
from mflux.models.ideogram4.variants.txt2img.ideogram4 import Ideogram4
from mflux.models.ideogram4.weights.ideogram4_weight_definition import Ideogram4WeightDefinition
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.fibo.variants.edit.fibo_edit import FIBOEdit
from mflux.models.ernie_image.variants.txt2img.ernie_image import ErnieImage
from mflux.models.krea2.variants.txt2img.krea2 import Krea2
from mflux.models.krea2.variants.controlnet.krea2_depth import Krea2Depth
from mflux.models.boogu.variants.txt2img.boogu_image import BooguImage

# Variant aliases that must dispatch to a dedicated class (a plain base-family
# ladder would route all of these to Flux1 and silently ignore their image inputs).
# Each entry verified against the CLI entrypoint that instantiates that class.
ALIAS_DISPATCH = {
    "dev-kontext":              (Flux1Kontext,       "flux-kontext"),    # flux_generate_kontext.py
    "dev-fill":                 (Flux1Fill,          "flux-fill"),       # flux_generate_fill.py
    "dev-fill-catvton":         (Flux1InContextFill, "flux-catvton"),    # flux_generate_in_context_catvton.py
    "dev-depth":                (Flux1Depth,         "flux-depth"),      # flux_generate_depth.py
    "dev-redux":                (Flux1Redux,         "flux-redux"),      # flux_generate_redux.py
    "dev-controlnet-canny":     (Flux1Controlnet,    "flux-controlnet"), # flux_generate_controlnet.py
    "schnell-controlnet-canny": (Flux1Controlnet,    "flux-controlnet"), # flux_generate_controlnet.py
    "dev-controlnet-upscaler":  (Flux1Controlnet,    "flux-upscaler"),   # flux_upscale.py (is controlnet)
    "fibo-edit":                (FIBOEdit,           "fibo-edit"),       # fibo_edit.py (not txt2img FIBO)
    "fibo-edit-rmbg":           (FIBOEdit,           "fibo-edit"),       # fibo_edit.py
}

# Aliases handled by dedicated nodes, not the txt2img/edit sampler.
SEEDVR2_ALIASES = {"seedvr2", "seedvr2-3b", "seedvr2-7b", "seedvr2-7B"}

# Aliases that resolve but have no standard generate_image sampler API (special output
# or a different entrypoint) — excluded from the sampler dropdown so they never crash it.
# Qwen-Image-Layered decomposes an image into RGBA layers (i2l), not a txt2img/edit sampler.
NON_SAMPLER_ALIASES = {"qwen-image-layered", "qwen-layered"}

# Base txt2img families the sampler drives end-to-end (seed/prompt, optional img2img).
BASE_FAMILIES = {"flux", "qwen", "z-image", "ideogram4", "fibo", "ernie", "flux2", "krea2", "boogu"}

# Variant/edit families the sampler drives via the typed MfluxImage feeder (primary +
# optional mask + optional depth/control map + multi-image). catvton and the in-context
# diptych variants are left out (multi-image diptych roles the feeder does not model).
# NOTE: krea2-depth (Krea2Depth) resolves but needs a required controlnet_path (the depth-control
# checkpoint) that the loader does not yet expose as an input — TODO: add a controlnet_path UI. Until
# then it is not offered (no dropdown alias routes to it) and stays out of the wired set.
WIRED_VARIANT_FAMILIES = {
    "flux-kontext", "flux-fill", "flux-depth", "flux-redux", "flux-controlnet",
    "qwen-edit", "fibo-edit", "flux2-edit",
}
# Edit/variant aliases that are valid models but not present in ui.MODEL_CHOICES.
DROPDOWN_EXTRA = ["qwen-image-edit", "flux2-klein-edit", "krea-2", "krea-2-raw"]


def pick_base_class(model: str, base_model: str = ""):
    """Base-family ladder. Order matters (qwen-edit before qwen, turbo via z-image)."""
    m, b = model.lower(), (base_model or "").lower()
    if "ernie" in m:
        return ErnieImage, "ernie"
    # Krea 2 (krea-2 / krea2), NOT Flux.1 Krea ("krea-dev" / "dev-krea", handled by the Flux1 fallback).
    if ("krea-2" in m or "krea2" in m) and "dev" not in m:
        return Krea2, "krea2"
    if "boogu" in m:
        return BooguImage, "boogu"
    if "qwen" in m and "edit" in m:
        return QwenImageEdit, "qwen-edit"
    if "qwen" in m:
        return QwenImage, "qwen"
    if "fibo" in m:
        return FIBO, "fibo"
    if "z-image" in m or "zimage" in m:        # z-image-turbo also routes to ZImage
        return ZImage, "z-image"
    if "flux2" in m or "flux.2" in m or "klein" in m:   # klein-* are FLUX.2, not Flux1
        return (Flux2KleinEdit, "flux2-edit") if "edit" in m else (Flux2Klein, "flux2")
    if "ideogram" in m or "ideogram" in b:
        return Ideogram4, "ideogram4"
    return Flux1, "flux"


def pick_model_class(model: str, base_model: str = ""):
    """Variant dispatch first (exact alias), then base ladder."""
    key = model.lower()
    if key in ALIAS_DISPATCH:
        return ALIAS_DISPATCH[key]
    return pick_base_class(model, base_model)


def is_builtin_name(model: str) -> bool:
    """True if ModelConfig.from_name resolves the alias without a model_path."""
    try:
        ModelConfig.from_name(model)
        return True
    except Exception:
        return False


def resolve_config_and_path(model: str, base_model: str = "", model_path: str = ""):
    """Return (cls, family, model_config, load_path) for any model spec.

    Ideogram-4 uses its own FP8 weight-definition resolver; everything else uses
    ModelConfig.from_name. Builtin alias -> path None; HF repo / local path -> path.
    """
    cls, family = pick_model_class(model, base_model)
    if cls is Ideogram4:
        cfg = Ideogram4WeightDefinition.resolve_inference_config(model, base_model or "", model_path or None)
        if model_path:                       # honor an explicit path over the builtin alias
            path = model_path
        else:
            path = None if Ideogram4WeightDefinition.is_builtin_name(model) else None
    else:
        cfg = ModelConfig.from_name(model, base_model=base_model or None)
        mn = (getattr(cfg, "model_name", "") or "").lower()
        if "seedvr2" in mn:                  # upscaler, not a sampler model — guard by resolved name
            raise ValueError(
                f"'{model}' resolves to SeedVR2 ({cfg.model_name}), an upscaler handled by "
                f"the MfluxUpscale node (Phase 3) — not a txt2img/edit sampler model."
            )
        if model_path:
            path = model_path
        else:
            path = None if is_builtin_name(model) else model
    return cls, family, cfg, path


def build_kwargs(cls, *, quantize=None, model_config=None, model_path=None,
                 lora_paths=None, lora_scales=None, bake_lora=True, controlnet_path=None):
    """Filter constructor kwargs to those the class's __init__ actually accepts.

    This is what makes the loader uniform across heterogeneous constructors
    (bake_lora absent on Ideogram/FIBO/Ernie, controlnet_path only on controlnet).
    Pure / no side effects, so it is unit-testable without loading weights.
    """
    params = set(inspect.signature(cls.__init__).parameters)
    kw = {}
    if "quantize" in params:
        kw["quantize"] = quantize
    if "model_config" in params:
        kw["model_config"] = model_config
    if "model_path" in params:
        kw["model_path"] = model_path
    if "lora_paths" in params:
        kw["lora_paths"] = lora_paths
    if "lora_scales" in params:
        kw["lora_scales"] = lora_scales
    if "bake_lora" in params and bake_lora is not None:
        kw["bake_lora"] = bake_lora
    if "controlnet_path" in params:
        kw["controlnet_path"] = controlnet_path
    return kw


def build_model(cls, **kwargs):
    """Construct a model instance, passing only the kwargs its __init__ accepts."""
    return cls(**build_kwargs(cls, **kwargs))
