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
import importlib
import inspect

from mflux.models.common.config import ModelConfig


def _imp(path, name):
    """Import name from module path, returning None if the installed mflux lacks it.

    The node is fork-agnostic: it drives whatever mflux is installed. Our fork (mflux-CV)
    ships extra families (Krea 2, Boogu, FLUX.2-Klein edit, ...) that stock upstream mflux
    does not have. A hard import of those would crash the whole node at load for every
    registry user on upstream mflux, so each model class is imported defensively and simply
    absent (None) when the installed mflux does not provide it. Dispatch and the dropdown
    skip the absent ones."""
    try:
        return getattr(importlib.import_module(path), name)
    except Exception:
        return None


Flux1 = _imp("mflux.models.flux.variants.txt2img.flux", "Flux1")
Flux1Kontext = _imp("mflux.models.flux.variants.kontext.flux_kontext", "Flux1Kontext")
Flux1Fill = _imp("mflux.models.flux.variants.fill.flux_fill", "Flux1Fill")
Flux1Depth = _imp("mflux.models.flux.variants.depth.flux_depth", "Flux1Depth")
Flux1Redux = _imp("mflux.models.flux.variants.redux.flux_redux", "Flux1Redux")
Flux1Controlnet = _imp("mflux.models.flux.variants.controlnet.flux_controlnet", "Flux1Controlnet")
Flux1InContextFill = _imp("mflux.models.flux.variants.in_context.flux_in_context_fill", "Flux1InContextFill")
Flux2Klein = _imp("mflux.models.flux2.variants.txt2img.flux2_klein", "Flux2Klein")
Flux2KleinEdit = _imp("mflux.models.flux2.variants.edit.flux2_klein_edit", "Flux2KleinEdit")
QwenImage = _imp("mflux.models.qwen.variants.txt2img.qwen_image", "QwenImage")
QwenImageEdit = _imp("mflux.models.qwen.variants.edit.qwen_image_edit", "QwenImageEdit")
ZImage = _imp("mflux.models.z_image.variants.z_image", "ZImage")
Ideogram4 = _imp("mflux.models.ideogram4.variants.txt2img.ideogram4", "Ideogram4")
Ideogram4WeightDefinition = _imp("mflux.models.ideogram4.weights.ideogram4_weight_definition", "Ideogram4WeightDefinition")
FIBO = _imp("mflux.models.fibo.variants.txt2img.fibo", "FIBO")
FIBOEdit = _imp("mflux.models.fibo.variants.edit.fibo_edit", "FIBOEdit")
ErnieImage = _imp("mflux.models.ernie_image.variants.txt2img.ernie_image", "ErnieImage")
Krea2 = _imp("mflux.models.krea2.variants.txt2img.krea2", "Krea2")
Krea2Depth = _imp("mflux.models.krea2.variants.controlnet.krea2_depth", "Krea2Depth")
BooguImage = _imp("mflux.models.boogu.variants.txt2img.boogu_image", "BooguImage")

# Variant aliases that must dispatch to a dedicated class (a plain base-family
# ladder would route all of these to Flux1 and silently ignore their image inputs).
# Each entry verified against the CLI entrypoint that instantiates that class.
_ALIAS_DISPATCH_RAW = {
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
    # Krea 2 depth-ControlNet (krea2_depth_generate.py). Needs a controlnet_path (the depth-control
    # checkpoint) supplied on the loader; routes here only via these explicit aliases so plain
    # "krea-2" still lands on txt2img Krea2.
    "krea-2-depth":             (Krea2Depth,         "krea2-depth"),     # krea2_depth_generate.py
    "krea2-depth":              (Krea2Depth,         "krea2-depth"),     # krea2_depth_generate.py (spelling alias)
}
# Drop entries whose class is absent from the installed mflux (upstream lacks fork-only variants).
ALIAS_DISPATCH = {k: v for k, v in _ALIAS_DISPATCH_RAW.items() if v[0] is not None}

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
# krea2-depth is wired now that the loader exposes controlnet_path (+ controlnet_strength):
# feed a room photo on 'image' (DepthPro auto-derives the depth) or a precomputed depth on 'map_image'.
WIRED_VARIANT_FAMILIES = {
    "flux-kontext", "flux-fill", "flux-depth", "flux-redux", "flux-controlnet",
    "qwen-edit", "fibo-edit", "flux2-edit", "krea2-depth",
}
# Edit/variant aliases that are valid models but not present in ui.MODEL_CHOICES.
# (krea-2-depth / krea2-depth come from ALIAS_DISPATCH and already surface in the dropdown.)
# Only offered when the installed mflux actually provides the backing class.
DROPDOWN_EXTRA = []
if QwenImageEdit is not None:
    DROPDOWN_EXTRA.append("qwen-image-edit")
if Flux2KleinEdit is not None:
    DROPDOWN_EXTRA.append("flux2-klein-edit")
if Krea2 is not None:
    DROPDOWN_EXTRA += ["krea-2", "krea-2-raw"]

# Families whose image roles are all optional in generate_image but that still need at least
# one image source at run time (krea2-depth: DepthPro needs a photo, unless a depth map is given).
NEEDS_ANY_IMAGE = {"krea2-depth"}


def pick_base_class(model: str, base_model: str = ""):
    """Base-family ladder. Order matters (qwen-edit before qwen, turbo via z-image).

    Each fork-only family is guarded on its class being available in the installed mflux;
    when absent, the model falls through (ultimately to Flux1) rather than routing to a None."""
    m, b = model.lower(), (base_model or "").lower()
    if ErnieImage is not None and "ernie" in m:
        return ErnieImage, "ernie"
    # Krea 2 (krea-2 / krea2), NOT Flux.1 Krea ("krea-dev" / "dev-krea", handled by the Flux1 fallback).
    if Krea2 is not None and ("krea-2" in m or "krea2" in m) and "dev" not in m:
        return Krea2, "krea2"
    if BooguImage is not None and "boogu" in m:
        return BooguImage, "boogu"
    if QwenImageEdit is not None and "qwen" in m and "edit" in m:
        return QwenImageEdit, "qwen-edit"
    if QwenImage is not None and "qwen" in m:
        return QwenImage, "qwen"
    if FIBO is not None and "fibo" in m:
        return FIBO, "fibo"
    if ZImage is not None and ("z-image" in m or "zimage" in m):   # z-image-turbo also routes to ZImage
        return ZImage, "z-image"
    if (Flux2Klein is not None or Flux2KleinEdit is not None) and ("flux2" in m or "flux.2" in m or "klein" in m):
        if "edit" in m and Flux2KleinEdit is not None:
            return Flux2KleinEdit, "flux2-edit"
        if Flux2Klein is not None:
            return Flux2Klein, "flux2"
        return Flux2KleinEdit, "flux2-edit"
    if Ideogram4 is not None and ("ideogram" in m or "ideogram" in b):
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
    if Flux1Redux is not None and cls is Flux1Redux and not model_path:
        # The FLUX.1-Redux-dev repo is adapter-only (SigLIP + embedder, no base weights). mflux
        # loads the redux adapter from there automatically but needs the base FLUX transformer/vae/
        # text-encoders from model_path; without one it wrongly looks for a vae in the adapter repo.
        # Default the base to FLUX.1-dev (an explicit model_path still wins via the guard above).
        cfg = ModelConfig.from_name(model, base_model=base_model or None)
        return cls, family, cfg, "black-forest-labs/FLUX.1-dev"
    if cls is Krea2Depth:
        # krea2-depth is not a ModelConfig alias (from_name would fail); it always runs on the
        # base Krea 2 config, same as the CLI. The depth-control checkpoint arrives via controlnet_path.
        cfg = ModelConfig.krea2()
        path = model_path or None
        return cls, family, cfg, path
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


def supports_controlnet_stack(cls) -> bool:
    """True if the class can load several controlnets at once (its __init__ takes controlnet_paths).
    Flux1Controlnet sums their residuals; Krea2Depth is input-concat and takes a single checkpoint."""
    return "controlnet_paths" in inspect.signature(cls.__init__).parameters


def requires_controlnet_path(cls) -> bool:
    """True if the class's __init__ makes controlnet_path a required arg (no default).
    Krea2Depth requires it (the depth-control checkpoint); Flux1Controlnet defaults it to None."""
    p = inspect.signature(cls.__init__).parameters.get("controlnet_path")
    return p is not None and p.default is inspect.Parameter.empty


def build_kwargs(cls, *, quantize=None, model_config=None, model_path=None,
                 lora_paths=None, lora_scales=None, bake_lora=True,
                 controlnet_path=None, controlnet_paths=None, controlnet_strength=None):
    """Filter constructor kwargs to those the class's __init__ actually accepts.

    This is what makes the loader uniform across heterogeneous constructors
    (bake_lora absent on Ideogram/FIBO/Ernie, controlnet_path only on controlnet models,
    controlnet_strength a constructor arg on Krea2Depth but a generate_image arg on Flux1Controlnet).
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
    if "controlnet_paths" in params and controlnet_paths:
        kw["controlnet_paths"] = list(controlnet_paths)
    elif "controlnet_path" in params:
        # Single-checkpoint classes (Krea2Depth) only take the singular form.
        kw["controlnet_path"] = controlnet_path
    if "controlnet_strength" in params and controlnet_strength is not None:
        kw["controlnet_strength"] = controlnet_strength
    return kw


def build_model(cls, **kwargs):
    """Construct a model instance, passing only the kwargs its __init__ accepts."""
    return cls(**build_kwargs(cls, **kwargs))
