"""
Flexible mflux node for ComfyUI (Apple Silicon / MLX).

Nodes:
  MfluxModelLoader  — resolve any mflux model (alias / HF repo / local path +
                      quantize + LoRA chain) once -> typed MFLUX_MODEL handle.
  MfluxModelSampler — read the handle's CapabilityProfile and honor / ignore-with-
                      warning / hard-block each param. Kills the silent steps/
                      guidance trap; wires image roles per family (img2img, kontext,
                      fill, depth, redux, controlnet, edit) via the MfluxImage feeder.
  MfluxLora         — chainable multi-LoRA feeder -> MFLUX_LORA.
  MfluxImage        — typed image feeder (primary + optional mask + optional depth/
                      control map + strength) -> MFLUX_IMAGE.
  MfluxUpscale      — SeedVR2 one-step upscaler (its own model).
"""
import json
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np
import torch

from mflux.cli.defaults import defaults as ui
from mflux.models.common.config import ModelConfig

try:
    from . import mflux_dispatch as D
    from . import capability as C
except ImportError:
    import mflux_dispatch as D
    import capability as C

try:
    import comfy.model_management as mm
except Exception:
    mm = None


# image-role arg names, by how the MfluxImage payload fills them
PRIMARY_SCALAR = {"image_path", "controlnet_image_path", "left_image_path"}
PRIMARY_LIST = {"redux_image_paths", "image_paths"}
MASK_ARGS = {"masked_image_path", "mask_path"}
AUX_ARGS = {"depth_image_path"}
STRENGTH_SCALAR = {"image_strength", "controlnet_strength"}
STRENGTH_LIST = {"redux_image_strengths"}

QUANTIZE_CHOICES = ["none"] + [str(q) for q in sorted(ui.QUANTIZE_CHOICES)]


def model_choices():
    """Curated dropdown: base txt2img families + the variant/edit families the
    typed MfluxImage feeder can drive. seedvr2 (upscaler) and multi-image diptych
    variants (catvton/in-context) are excluded; typed aliases still hard-block
    cleanly via the sampler's required-arg guard."""
    allowed = D.BASE_FAMILIES | D.WIRED_VARIANT_FAMILIES
    seen, out = set(), []
    for k in list(ui.MODEL_CHOICES) + list(D.ALIAS_DISPATCH) + list(D.DROPDOWN_EXTRA):
        if k in seen or k in D.SEEDVR2_ALIASES:
            continue
        seen.add(k)
        try:
            _, fam = D.pick_model_class(k)
            if fam in allowed:
                out.append(k)
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- #
# payloads carried on typed sockets
# --------------------------------------------------------------------------- #
@dataclass
class MfluxModelHandle:
    instance: object
    family: str
    model_config: object
    alias: str
    profile: C.CapabilityProfile
    cache_key: tuple
    free_comfy_first: bool = True


@dataclass
class MfluxImagePayload:
    primary: object               # IMAGE tensor (required)
    mask: object = None           # IMAGE tensor (inpaint / fill)
    aux: object = None            # IMAGE tensor (depth map / control image)
    strength: float = None


_CACHE = {"key": None, "model": None}          # one big sampler model resident
_UPSCALE_CACHE = {"key": None, "model": None}   # SeedVR2 lives in its own slot


def _free_comfy():
    if mm is not None:
        try:
            mm.unload_all_models()
            mm.soft_empty_cache(force=True)
        except Exception:
            pass


def _pil_to_image(pil):
    arr = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _image_to_temp_png(image_tensor):
    from PIL import Image
    arr = (image_tensor[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    fd, path = tempfile.mkstemp(suffix=".png", prefix="mflux_in_")
    os.close(fd)
    Image.fromarray(arr).save(path)
    return path


def _clean_caption_prompt(prompt):
    """Pull a clean JSON caption out of an LLM's output. mflux's Ideogram parser
    only uses the structured caption when the prompt starts with '{' and parses to
    a dict; otherwise it degrades to plain text, which raises the model's
    safety-filter false positives. An LLM (e.g. Gemma) may wrap the JSON in
    ```json fences or a preamble, so extract the outermost JSON object."""
    if not isinstance(prompt, str):
        return prompt
    s = prompt.strip()
    if s.startswith("{"):
        return s
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        cand = s[i:j + 1]
        try:
            if isinstance(json.loads(cand), dict):
                return cand
        except Exception:
            pass
    return prompt


# --------------------------------------------------------------------------- #
# image-role injection (Phase 3)
# --------------------------------------------------------------------------- #
def inject_image(profile, out, paths, gen):
    """Fill the model's real image-role args from a materialized payload of paths
    {primary, mask, aux, strength}. Driven by profile.image_role_args, so each
    family gets exactly the args its generate_image declares."""
    for arg, req in profile.image_role_args.items():
        if arg in PRIMARY_SCALAR:
            out[arg] = paths["primary"]
        elif arg in PRIMARY_LIST:
            out[arg] = [paths["primary"]]
        elif arg in MASK_ARGS:
            if paths.get("mask") is None:
                raise ValueError(f"'{profile.family}' needs a mask — connect 'mask' on MfluxImage.")
            out[arg] = paths["mask"]
        elif arg in AUX_ARGS:
            if paths.get("aux") is not None:
                out[arg] = paths["aux"]
            elif req == "req":
                raise ValueError(f"'{profile.family}' needs a depth/control map — connect 'map_image' on MfluxImage.")
    strength = paths.get("strength")
    if strength is not None:
        for arg in gen & STRENGTH_SCALAR:
            out[arg] = float(strength)
        for arg in gen & STRENGTH_LIST:
            out[arg] = [float(strength)]


def normalize_and_validate(profile, params_mode, requested, image_paths):
    """Return (forwarded_kwargs, notes). image_paths is None or a dict of
    materialized paths {primary, mask, aux, strength}."""
    notes = []
    gen = profile.gen_kwargs
    out = {}
    has_image = image_paths is not None

    if "seed" in gen:
        out["seed"] = requested["seed"]
    if "prompt" in gen:
        out["prompt"] = requested["prompt"]
    if "width" in gen:
        out["width"] = requested["width"]
    if "height" in gen:
        out["height"] = requested["height"]

    # TIER 1 — image gating
    if has_image and not (profile.supports_img2img or profile.image_role_args):
        raise ValueError(f"'{profile.family}' does not accept an input image.")
    if profile.needs_image and not has_image:
        roles = ", ".join(sorted(profile.required_image_args))
        raise ValueError(f"'{profile.family}' requires image input(s) [{roles}] — connect a MfluxImage.")

    # TIER 2 — the steps/guidance trap on preset-only models
    user_steps = requested.get("steps")
    user_guid = requested.get("guidance")
    if profile.is_preset_only:
        if params_mode == "auto":
            if user_steps or user_guid:
                notes.append(f"[mflux] '{profile.family}': the preset controls steps/guidance; your values are IGNORED (params_mode=override to force).")
            user_steps = user_guid = None
        elif user_steps or user_guid:
            notes.append(f"[mflux] OVERRIDE on '{profile.family}': flat guidance + mismatched noise schedule; quality may degrade.")
        if "preset" in gen and requested.get("preset") not in (None, "", "(model default)"):
            out["preset"] = requested["preset"]
        if "strict_caption_validation" in gen:
            out["strict_caption_validation"] = bool(requested.get("strict_caption_validation", False))

    if user_steps and int(user_steps) > 0 and "num_inference_steps" in gen:
        out["num_inference_steps"] = int(user_steps)
    if user_guid and float(user_guid) > 0 and "guidance" in gen and profile.supports_guidance:
        out["guidance"] = float(user_guid)

    # TIER 3 — drop-with-warning
    neg = requested.get("negative_prompt", "")
    if neg:
        if profile.supports_negative:
            out["negative_prompt"] = neg
        else:
            notes.append(f"[mflux] negative_prompt has no effect on '{profile.family}'; dropped.")
    sch = requested.get("scheduler", "auto")
    if sch and sch != "auto" and "scheduler" in gen:
        out["scheduler"] = sch

    # image roles (img2img + variant roles)
    if has_image:
        inject_image(profile, out, image_paths, gen)

    # invariant guard: every required generate_image arg must be satisfied
    missing = profile.required_gen_params - set(out)
    if missing:
        raise ValueError(
            f"'{profile.family}' needs generate_image arg(s) {sorted(missing)} that the sampler "
            f"could not supply — check the MfluxImage roles (mask / map_image) for this model."
        )
    return out, notes


# --------------------------------------------------------------------------- #
# typed feeders
# --------------------------------------------------------------------------- #
class MfluxLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_path": ("STRING", {"default": "", "tooltip": "Local file, HF repo, or repo:filename.safetensors"}),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            },
            "optional": {"lora_in": ("MFLUX_LORA",)},
        }

    RETURN_TYPES = ("MFLUX_LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "build"
    CATEGORY = "MLX/mflux/inputs"

    def build(self, lora_path, lora_scale, lora_in=None):
        chain = list(lora_in) if lora_in else []
        if lora_path.strip():
            chain.append((lora_path.strip(), float(lora_scale)))
        return (chain,)


class MfluxImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "mask": ("IMAGE", {"tooltip": "Inpaint mask for fill models (masked_image_path)."}),
                "map_image": ("IMAGE", {"tooltip": "Depth map / control image for depth & controlnet models."}),
                "strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MFLUX_IMAGE",)
    RETURN_NAMES = ("mflux_image",)
    FUNCTION = "build"
    CATEGORY = "MLX/mflux/inputs"

    def build(self, image, mask=None, map_image=None, strength=0.6):
        return (MfluxImagePayload(primary=image, mask=mask, aux=map_image, strength=float(strength)),)


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #
class MfluxModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (model_choices(), {"default": "schnell"}),
                "quantize": (QUANTIZE_CHOICES, {"default": "none"}),
            },
            "optional": {
                "base_model": (["(none)"] + list(ui.MODEL_CHOICES), {"default": "(none)"}),
                "model_path": ("STRING", {"default": "", "tooltip": "HF org/model or local path; overrides 'model' as the load target."}),
                "lora": ("MFLUX_LORA",),
                "keep_loaded": ("BOOLEAN", {"default": True}),
                "free_comfy_first": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "MLX/mflux/loaders"

    def load(self, model, quantize, base_model="(none)", model_path="",
             lora=None, keep_loaded=True, free_comfy_first=True):
        base = "" if base_model in ("(none)", "") else base_model
        q = None if quantize == "none" else int(quantize)
        lp = [p for p, s in lora] if lora else None
        ls = [s for p, s in lora] if lora else None

        cls, family, cfg, path = D.resolve_config_and_path(model, base, model_path.strip())
        cache_key = (family, path or model, q, tuple(lp or ()), tuple(ls or ()), base)

        if keep_loaded and _CACHE["key"] == cache_key:
            instance = _CACHE["model"]
        else:
            if free_comfy_first:
                _free_comfy()
            instance = D.build_model(
                cls, quantize=q, model_config=cfg, model_path=path,
                lora_paths=lp, lora_scales=ls, bake_lora=True,
            )
            _CACHE["key"] = cache_key if keep_loaded else None
            _CACHE["model"] = instance if keep_loaded else None

        profile = C.build_profile(cls, family, cfg)
        handle = MfluxModelHandle(
            instance=instance, family=family, model_config=cfg, alias=model,
            profile=profile, cache_key=cache_key, free_comfy_first=free_comfy_first,
        )
        return (handle,)


# --------------------------------------------------------------------------- #
# sampler
# --------------------------------------------------------------------------- #
class MfluxModelSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MFLUX_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "params_mode": (["auto", "override"], {"default": "auto", "tooltip": "auto: preset-driven models ignore steps/guidance. override: force them."}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "0 = model default. Ignored on preset-only models in auto."}),
                "guidance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 40.0, "step": 0.1, "tooltip": "0 = model default."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "scheduler": ("STRING", {"default": "auto"}),
                "preset": (["(model default)", "V4_DEFAULT_20", "V4_QUALITY_48", "V4_TURBO_12"], {"default": "(model default)"}),
                "strict_caption_validation": ("BOOLEAN", {"default": False}),
                "image": ("IMAGE", {"tooltip": "Quick img2img on base models. For variants (fill/depth/controlnet) use MfluxImage."}),
                "image_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mflux_image": ("MFLUX_IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    CATEGORY = "MLX/mflux"

    def generate(self, model, prompt, params_mode, negative_prompt="", seed=0,
                 steps=0, guidance=0.0, width=1024, height=1024, scheduler="auto",
                 preset="(model default)", strict_caption_validation=False,
                 image=None, image_strength=0.6, mflux_image=None):
        handle = model
        prompt = _clean_caption_prompt(prompt)
        if handle.free_comfy_first:
            _free_comfy()

        # resolve the image payload (typed MfluxImage wins over the simple image)
        payload = mflux_image
        if payload is None and image is not None:
            payload = MfluxImagePayload(primary=image, strength=image_strength)

        temps = []

        def materialize(t):
            p = _image_to_temp_png(t)
            temps.append(p)
            return p

        try:
            image_paths = None
            if payload is not None:
                image_paths = {
                    "primary": materialize(payload.primary),
                    "mask": materialize(payload.mask) if payload.mask is not None else None,
                    "aux": materialize(payload.aux) if payload.aux is not None else None,
                    "strength": payload.strength if payload.strength is not None else image_strength,
                }
            requested = {
                "seed": int(seed), "prompt": prompt, "negative_prompt": negative_prompt,
                "steps": steps, "guidance": guidance, "width": int(width), "height": int(height),
                "scheduler": scheduler, "preset": preset,
                "strict_caption_validation": strict_caption_validation,
            }
            forwarded, notes = normalize_and_validate(handle.profile, params_mode, requested, image_paths)
            gen = handle.instance.generate_image(**forwarded)
        finally:
            for p in temps:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

        info = f"model={handle.alias} family={handle.family} forwarded={sorted(forwarded)}"
        if notes:
            info += "\n" + "\n".join(notes)
            for n in notes:
                print(n)
        return (_pil_to_image(gen.image), info)


# --------------------------------------------------------------------------- #
# SeedVR2 upscaler (own model)
# --------------------------------------------------------------------------- #
class MfluxUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["seedvr2-3b", "seedvr2-7b"], {"default": "seedvr2-3b"}),
                "resolution": ("INT", {"default": 1080, "min": 256, "max": 4096, "step": 16, "tooltip": "Target shortest-edge resolution (pixels)."}),
            },
            "optional": {
                "softness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "quantize": (QUANTIZE_CHOICES, {"default": "none"}),
                "keep_loaded": ("BOOLEAN", {"default": True}),
                "free_comfy_first": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "MLX/mflux"

    def upscale(self, image, model, resolution, softness=0.0, seed=42,
                quantize="none", keep_loaded=True, free_comfy_first=True):
        from mflux.models.seedvr2.variants.upscale.seedvr2 import SeedVR2
        q = None if quantize == "none" else int(quantize)
        key = (model, q)
        if keep_loaded and _UPSCALE_CACHE["key"] == key:
            inst = _UPSCALE_CACHE["model"]
        else:
            if free_comfy_first:
                _free_comfy()
            cfg = ModelConfig.from_name(model)
            inst = SeedVR2(quantize=q, model_path=None, model_config=cfg)
            _UPSCALE_CACHE["key"] = key if keep_loaded else None
            _UPSCALE_CACHE["model"] = inst if keep_loaded else None

        path = _image_to_temp_png(image)
        try:
            gen = inst.generate_image(seed=int(seed), image_path=path, resolution=int(resolution), softness=float(softness))
        finally:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        return (_pil_to_image(gen.image),)
