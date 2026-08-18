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
import glob
import json
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np
import torch

from mflux.cli.defaults import defaults as ui
from mflux.models.common.config.model_config import AVAILABLE_MODELS as _REGISTRY

# Upstream #592 removed the hand-maintained ui.MODEL_CHOICES and derives CLI choices
# from the registry instead; mflux-cv 0.18.x still ships it. Same derivation here,
# with the old constant as the fallback so both runtimes keep working.
MODEL_CHOICES = getattr(ui, "MODEL_CHOICES", None) or [
    alias for config in _REGISTRY.values() for alias in config.aliases
]
from mflux.models.common.config import ModelConfig
from mflux.utils.scale_factor import ScaleFactor

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

try:
    import folder_paths
except Exception:
    folder_paths = None


def _lora_files():
    """LoRA filenames ComfyUI knows about (from the configured loras dirs)."""
    if folder_paths is None:
        return []
    try:
        return list(folder_paths.get_filename_list("loras"))
    except Exception:
        return []


# image-role arg names, by how the MfluxImage payload fills them
PRIMARY_SCALAR = {"image_path", "controlnet_image_path", "left_image_path"}
PRIMARY_LIST = {"redux_image_paths", "image_paths"}
MASK_ARGS = {"masked_image_path", "mask_path"}
AUX_ARGS = {"depth_image_path"}
STRENGTH_SCALAR = {"image_strength", "controlnet_strength"}
STRENGTH_LIST = {"redux_image_strengths"}

QUANTIZE_CHOICES = ["none"] + [str(q) for q in sorted(ui.QUANTIZE_CHOICES)]


NEEDS_DL = " ⬇"           # marks a dropdown model not present in the local HF cache
_HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")


def _repo_cached(repo):
    """True if the model's HuggingFace repo is FULLY cached locally — has weights
    and no pending .incomplete blobs (a partial/interrupted download still triggers
    a fetch when selected)."""
    if not repo or "/" not in str(repo):
        return False
    base = os.path.join(_HF_CACHE, "models--" + str(repo).replace("/", "--"))
    snaps = os.path.join(base, "snapshots")
    if not os.path.isdir(snaps):
        return False
    has_weights = bool(glob.glob(os.path.join(snaps, "**", "*.safetensors"), recursive=True)
                       or glob.glob(os.path.join(snaps, "**", "*.gguf"), recursive=True))
    if not has_weights:
        return False
    if glob.glob(os.path.join(base, "blobs", "*.incomplete")):  # partial download
        return False
    return True


def _is_downloaded(alias):
    try:
        _, _, cfg, _ = D.resolve_config_and_path(alias)
        return _repo_cached(getattr(cfg, "model_name", ""))
    except Exception:
        return False


def strip_mark(name):
    """Recover the clean alias from a dropdown label that may carry the NEEDS_DL mark."""
    return name[:-len(NEEDS_DL)] if name.endswith(NEEDS_DL) else name


def model_choices():
    """Curated dropdown of the models the sampler can drive (base txt2img + the
    wired image-conditioned variants; seedvr2/diptych excluded). Locally-downloaded
    models are listed first with clean names; models not in the HF cache come after,
    marked with NEEDS_DL so you can see they would trigger a download (gated repos a
    403). The mark is a best-effort hint and is stripped before loading."""
    allowed = D.BASE_FAMILIES | D.WIRED_VARIANT_FAMILIES
    seen, have, need = set(), [], []
    for k in list(MODEL_CHOICES) + list(D.ALIAS_DISPATCH) + list(D.DROPDOWN_EXTRA):
        if k in seen or k in D.SEEDVR2_ALIASES or k in D.NON_SAMPLER_ALIASES:
            continue
        seen.add(k)
        try:
            _, fam = D.pick_model_class(k)
            if fam not in allowed:
                continue
            (have if _is_downloaded(k) else need).append(k)
        except Exception:
            pass
    return have + [k + NEEDS_DL for k in need]


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
    lora_note: str = ""

    def __repr__(self) -> str:
        # Deliberately shallow. When a node raises, ComfyUI formats every input with str() to build
        # the error report; the dataclass default __repr__ would print `instance`, recursing into the
        # whole MLX module tree. Some MLX layers have a broken __repr__ (e.g. Conv3d reads a missing
        # `groups`), so that recursion raises INSIDE the error handler and kills ComfyUI's prompt
        # worker thread — turning one failed generation into a dead queue that needs a restart.
        return f"MfluxModelHandle(alias={self.alias!r}, family={self.family!r})"


@dataclass
class MfluxImagePayload:
    images: list                  # ordered IMAGE tensors; [0] is the primary/viewpoint ref (>=1)
    mask: object = None           # IMAGE tensor (inpaint / fill)
    aux: object = None            # IMAGE tensor (depth map / control image)
    strength: float = None        # scalar roles (img2img / controlnet) — the primary's strength
    strengths: list = None        # per-image, aligned with images (redux multi-reference blend)


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


def _mask_to_gray01(mask_tensor, out_w, out_h, feather):
    """A ComfyUI mask/image tensor -> float32 HxW in [0,1] at (out_h,out_w), edge-feathered.
    Accepts an IMAGE (B,H,W,C) or MASK (B,H,W); collapses channels by mean."""
    from PIL import Image, ImageFilter
    a = mask_tensor
    if hasattr(a, "cpu"):
        a = a.cpu().numpy()
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 4:      # (B,H,W,C)
        a = a[0]
        a = a.mean(axis=2) if a.ndim == 3 else a
    elif a.ndim == 3:    # (B,H,W) mask or (H,W,C) image
        a = a[0] if a.shape[0] in (1, 2, 3, 4) and a.shape[-1] not in (1, 2, 3, 4) else a.mean(axis=2)
    a = np.clip(a, 0.0, 1.0)
    pil = Image.fromarray((a * 255.0).astype(np.uint8), mode="L").resize((out_w, out_h), Image.BILINEAR)
    if feather and feather > 0:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(feather)))
    return np.asarray(pil, dtype=np.float32) / 255.0


def _preserve_composite(original_tensor, edited_pil, mask_tensor, keep_white, feather):
    """Blend an edited image back over the original so painted regions stay pixel-identical.
    keep_white=True: white mask = KEEP original (lock openings). False: white = allow the edit
    (standard inpaint polarity). Returns a (1,H,W,3) IMAGE tensor at the edited image's size."""
    edited = np.asarray(edited_pil.convert("RGB"), dtype=np.float32) / 255.0
    h, w = edited.shape[:2]
    from PIL import Image
    orig_pil = Image.fromarray((original_tensor[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)).convert("RGB")
    orig = np.asarray(orig_pil.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    m = _mask_to_gray01(mask_tensor, w, h, feather)[..., None]   # HxWx1, 1 = white
    keep = m if keep_white else (1.0 - m)                        # weight on the ORIGINAL
    out = orig * keep + edited * (1.0 - keep)
    return torch.from_numpy(np.clip(out, 0.0, 1.0)).unsqueeze(0)


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


def _parse_resolution(val):
    """SeedVR2 resolution: an int (shortest-edge pixels, e.g. 1080) or a scale
    factor (e.g. '2x', '1.5x'); 'auto' / empty means 1x (no rescale)."""
    s = str(val).strip().lower()
    if s in ("", "auto"):
        return ScaleFactor(value=1)
    try:
        return int(s)
    except ValueError:
        return ScaleFactor.parse(s)


# --------------------------------------------------------------------------- #
# image-role injection (Phase 3)
# --------------------------------------------------------------------------- #
def inject_image(profile, out, paths, gen):
    """Fill the model's real image-role args from a materialized payload of paths
    {primary, mask, aux, strength}. Driven by profile.image_role_args, so each
    family gets exactly the args its generate_image declares."""
    primaries = list(paths.get("primaries") or [paths["primary"]])
    if "controls" in profile.image_role_args:
        # Z-Image Union ControlNet: one ControlSpec per image in the MfluxImage chain, typed by the
        # sampler's control_type widget. A single type applies to every image; a comma-separated list
        # ("canny,depth") pairs with the chain in order, which is how a stack is built. Strengths come
        # from the chain the same way a stacked FLUX controlnet gets one per net.
        types = [t.strip().lower() for t in str(paths.get("control_type") or "canny").split(",") if t.strip()]
        if len(types) == 1:
            types = types * len(primaries)
        if len(types) != len(primaries):
            raise ValueError(
                f"control_type has {len(types)} entries but {len(primaries)} control image(s) are "
                f"connected. Give one type, or one per image in chain order."
            )
        valid = {t.value for t in D.ControlType}
        bad = [t for t in types if t not in valid]
        if bad:
            raise ValueError(f"unknown control type(s) {bad}; valid: {sorted(valid)}")
        per_image = paths.get("strengths") or [paths.get("strength") or 1.0] * len(primaries)
        out["controls"] = [
            D.ControlSpec(type=D.ControlType(t), image_path=p, strength=float(s))
            for t, p, s in zip(types, primaries, per_image)
        ]
        return
    for arg, req in profile.image_role_args.items():
        if arg in PRIMARY_SCALAR:
            # A stack of controlnets takes one control image each, in the order of the MfluxImage
            # chain; mflux accepts a list here and pairs it with the checkpoints. Every other scalar
            # role (img2img, diptych) keeps taking just the primary.
            if arg == "controlnet_image_path" and len(primaries) > 1:
                out[arg] = primaries
            else:
                out[arg] = paths["primary"]
        elif arg in PRIMARY_LIST:
            out[arg] = list(paths.get("primaries") or [paths["primary"]])
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
        roles = profile.image_role_args
        for arg in gen & STRENGTH_SCALAR:
            # `image_strength` must NOT be forced on full instruction-edit models (image_paths role):
            # it defaults to None there (a full edit), and forcing a value turns the edit into a degraded
            # partial img2img (observed: flux2-edit/qwen-edit reinterpret the scene entirely). Apply it
            # only for real img2img (scalar image_path). controlnet_strength is unaffected.
            if arg == "image_strength" and "image_path" not in roles:
                continue
            # A stacked controlnet gets one strength per net, from the MfluxImage chain.
            if arg == "controlnet_strength" and len(primaries) > 1:
                per_image = paths.get("strengths")
                out[arg] = [float(s) for s in per_image] if per_image else [float(strength)] * len(primaries)
                continue
            out[arg] = float(strength)
        for arg in gen & STRENGTH_LIST:       # redux_image_strengths — one per reference image
            per_image = paths.get("strengths")
            out[arg] = [float(s) for s in per_image] if per_image else [float(strength)]


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
    # Families whose image roles are all optional but that still need one source at run time
    # (krea2-depth: DepthPro needs a photo unless a precomputed depth map is provided on map_image).
    if profile.family in D.NEEDS_ANY_IMAGE and not has_image:
        raise ValueError(
            f"'{profile.family}' needs a room photo (connect 'image' on MfluxImage — DepthPro derives "
            f"the depth) or a precomputed depth map (connect 'map_image')."
        )

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

    # TIER 3 — drop-with-warning, keyed on EFFECTIVE behavior (mirrors mflux-capabilities):
    # a family can take a negative prompt and still never encode it, either because CFG is
    # disabled outright (guidance-distilled models) or because its encoder only builds the
    # negative branch at guidance > 1.0 and the effective guidance sits at or below that.
    neg = requested.get("negative_prompt", "")
    if neg:
        eff_guidance = out.get("guidance", profile.negative_guidance_default)
        if not profile.supports_negative:
            notes.append(f"[mflux] negative_prompt has no effect on '{profile.family}'; dropped.")
        elif not profile.supports_guidance:
            notes.append(f"[mflux] negative_prompt has no effect on '{profile.family}': CFG is disabled on this model; dropped.")
        elif profile.negative_guidance_default is not None and float(eff_guidance) <= 1.0:
            notes.append(
                f"[mflux] negative_prompt has no effect on '{profile.family}' at guidance <= 1.0 "
                f"(effective guidance {float(eff_guidance):g}); raise guidance above 1.0 to enable it; dropped."
            )
        else:
            out["negative_prompt"] = neg
    sch = requested.get("scheduler", "auto")
    if sch and sch != "auto" and "scheduler" in gen:
        out["scheduler"] = sch

    # TIER 4 — signature-gated extras: forwarded only when the model's generate_image
    # declares them AND the user moved them off their inert default. Requested-but-
    # unsupported gets the same drop-with-note treatment as negative_prompt, never a
    # silent swallow. This is what actually exposes Z-Image's shift/sigma_schedule/
    # mcf_max_change and the PiD decoder without a per-model whitelist.
    extras = (
        ("pid_decode", False),
        ("pid_degrade_sigma", 0.0),
        ("shift", 0.0),
        ("sigma_schedule", ""),
        ("mcf_max_change", 0.0),
    )
    for name, inert in extras:
        value = requested.get(name, inert)
        if value == inert or value is None:
            continue
        if name in gen:
            out[name] = value
        else:
            notes.append(f"[mflux] '{name}' is not supported by '{profile.family}'; dropped.")

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
                "lora_name": (["(none)"] + _lora_files(), {"default": "(none)", "tooltip": "LoRA from your ComfyUI loras folder. It must match the loaded model's architecture (e.g. an Ideogram-4 LoRA only works on ideogram4)."}),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            },
            "optional": {"lora_in": ("MFLUX_LORA",)},
        }

    RETURN_TYPES = ("MFLUX_LORA",)
    RETURN_NAMES = ("lora",)
    FUNCTION = "build"
    CATEGORY = "MLX/mflux/inputs"

    def build(self, lora_name, lora_scale, lora_in=None):
        chain = list(lora_in) if lora_in else []
        if lora_name and lora_name != "(none)":
            path = lora_name
            if folder_paths is not None:
                try:
                    resolved = folder_paths.get_full_path("loras", lora_name)
                    if resolved:
                        path = resolved
                except Exception:
                    pass
            chain.append((path, float(lora_scale)))
        return (chain,)


class MfluxImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "image_in": ("MFLUX_IMAGE", {"tooltip": "Chain another MfluxImage to add reference images (multi-image edit, e.g. qwen-edit / flux2-edit). The first image in the chain is the primary/viewpoint reference."}),
                "mask": ("IMAGE", {"tooltip": "Mask. On flux-fill it is the native inpaint mask. On edit/img2img models (FLUX.2-edit, qwen-edit, krea2) it drives the sampler's mask-preserve composite: white = lock to original (hard-lock windows/doors) or take-the-edit, per the sampler's mask_mode. IMAGE type — convert a MaskEditor/SAM MASK with a 'Convert Mask to Image' node."}),
                "map_image": ("IMAGE", {"tooltip": "Depth map / control image for depth & controlnet models."}),
                "strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MFLUX_IMAGE",)
    RETURN_NAMES = ("mflux_image",)
    FUNCTION = "build"
    CATEGORY = "MLX/mflux/inputs"

    def build(self, image, image_in=None, mask=None, map_image=None, strength=0.6):
        s = float(strength) if strength is not None else 1.0
        if image_in is not None:
            # Append to the chain; keep the upstream mask/aux unless this node overrides them.
            # strengths grows with the chain (one per image); scalar strength stays the primary's,
            # so redux gets a per-reference blend while img2img/controlnet keep a single value.
            prev = list(image_in.strengths) if image_in.strengths else [image_in.strength if image_in.strength is not None else 1.0]
            return (MfluxImagePayload(
                images=list(image_in.images) + [image],
                mask=mask if mask is not None else image_in.mask,
                aux=map_image if map_image is not None else image_in.aux,
                strength=image_in.strength if image_in.strength is not None else s,
                strengths=prev + [s],
            ),)
        return (MfluxImagePayload(images=[image], mask=mask, aux=map_image, strength=s, strengths=[s]),)


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
                "base_model": (["(none)"] + list(MODEL_CHOICES), {"default": "(none)"}),
                "model_path": ("STRING", {"default": "", "tooltip": "HF org/model or local path; overrides 'model' as the load target."}),
                "lora": ("MFLUX_LORA",),
                "controlnet_path": ("STRING", {"multiline": True, "default": "", "tooltip": "ControlNet checkpoint: a local path or an HF repo. REQUIRED for krea-2-depth (the depth-control checkpoint). On flux-controlnet you can STACK several by putting ONE PER LINE (e.g. a depth net and a canny net); then chain one MfluxImage per controlnet, in the same order, each with its own strength."}),
                "controlnet_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Krea-2-depth only: scales the control deltas merged into the base weights at load time. (flux-controlnet takes a strength per controlnet from the MfluxImage chain instead.)"}),
                "keep_loaded": ("BOOLEAN", {"default": True}),
                "free_comfy_first": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MFLUX_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "MLX/mflux/loaders"

    def load(self, model, quantize, base_model="(none)", model_path="",
             lora=None, controlnet_path="", controlnet_strength=1.0,
             keep_loaded=True, free_comfy_first=True):
        model = strip_mark(model)          # drop the dropdown "needs download" mark
        base = "" if base_model in ("(none)", "") else base_model
        q = None if quantize == "none" else int(quantize)
        lp = [p for p, s in lora] if lora else None
        ls = [s for p, s in lora] if lora else None
        # One checkpoint per line, so several controlnets can be stacked (mflux sums their residuals).
        cn_paths = [ln.strip() for ln in controlnet_path.splitlines() if ln.strip()]

        cls, family, cfg, path = D.resolve_config_and_path(model, base, model_path.strip())

        # Fail early with a clear message rather than an opaque TypeError from the constructor.
        if D.requires_controlnet_path(cls) and not cn_paths:
            raise ValueError(
                f"'{model}' is a ControlNet model and needs a checkpoint — set 'controlnet_path' "
                f"on the loader to the local depth-control checkpoint."
            )
        if len(cn_paths) > 1 and not D.supports_controlnet_stack(cls):
            raise ValueError(
                f"'{model}' takes a single ControlNet checkpoint, but {len(cn_paths)} were given "
                f"(one per line). Stacking is supported on flux-controlnet."
            )
        for p in cn_paths:
            # An HF repo (org/name) is resolved on load; anything else must be a real local path.
            if not os.path.exists(p) and not (p.count("/") == 1 and not p.startswith((".", "/", "~"))):
                raise ValueError(f"controlnet_path does not exist and is not an HF repo id: {p}")

        cache_key = (family, path or model, q, tuple(lp or ()), tuple(ls or ()), base,
                     tuple(cn_paths), float(controlnet_strength))

        lora_note = ""
        if keep_loaded and _CACHE["key"] == cache_key:
            instance = _CACHE["model"]
        else:
            if free_comfy_first:
                _free_comfy()
            try:
                instance = D.build_model(
                    cls, quantize=q, model_config=cfg, model_path=path,
                    lora_paths=lp, lora_scales=ls, bake_lora=True,
                    controlnet_paths=cn_paths, controlnet_path=cn_paths[0] if cn_paths else None,
                    controlnet_strength=float(controlnet_strength),
                )
            except Exception as e:
                blob = (type(e).__name__ + " " + str(e)).lower()
                if "no lora layers were applied" in blob and lp:
                    # LoRA built for another architecture -> load WITHOUT it and keep going,
                    # so switching models with a LoRA attached never hard-crashes.
                    names = ", ".join(os.path.basename(p) for p in lp)
                    lora_note = (f"[mflux] LoRA [{names}] is incompatible with '{model}' (built for a "
                                 f"different architecture) — loaded WITHOUT it. Use a LoRA made for '{model}'.")
                    print(lora_note)
                    instance = D.build_model(
                        cls, quantize=q, model_config=cfg, model_path=path,
                        lora_paths=None, lora_scales=None, bake_lora=True,
                        controlnet_paths=cn_paths, controlnet_path=cn_paths[0] if cn_paths else None,
                        controlnet_strength=float(controlnet_strength),
                    )
                elif any(t in blob for t in ("gated", "restricted", "403", "401")):
                    raise RuntimeError(
                        f"'{model}' is gated or not downloaded: mflux tried to fetch it from "
                        f"HuggingFace and was denied. Request access on its HF page and set "
                        f"HF_TOKEN, or pick a model shown without the '{NEEDS_DL.strip()}' mark."
                    ) from e
                else:
                    raise
            _CACHE["key"] = cache_key if keep_loaded else None
            _CACHE["model"] = instance if keep_loaded else None

        profile = C.build_profile(cls, family, cfg)
        handle = MfluxModelHandle(
            instance=instance, family=family, model_config=cfg, alias=model,
            profile=profile, cache_key=cache_key, free_comfy_first=free_comfy_first,
            lora_note=lora_note,
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
                "mask_mode": (["preserve", "inpaint", "off"], {"default": "preserve", "tooltip": "How to use a mask connected on MfluxImage for edit/img2img models that don't inpaint natively (FLUX.2-edit, qwen-edit, krea2, img2img). preserve: painted (white) areas stay pixel-identical to the original — lock windows/doors. inpaint: only painted (white) areas take the edit. off: ignore. (flux-fill uses its mask natively; this does not apply.)"}),
                "mask_feather": ("INT", {"default": 4, "min": 0, "max": 200, "tooltip": "Gaussian feather (px) on the mask edge for a seamless composite."}),
                "live_preview": ("BOOLEAN", {"default": True, "tooltip": "Stream the image as it denoises into the node, and drive the progress bar. Also enables the Cancel button mid-generation. Costs one VAE decode per shown step."}),
                "preview_stride": ("INT", {"default": 2, "min": 1, "max": 20, "tooltip": "Show a preview every N steps (the last step is always shown). 1 = every step (slowest, smoothest). Higher = fewer decodes."}),
                # Appended last on purpose. ComfyUI maps `widgets_values` positionally, so inserting a
                # widget anywhere above would shift every value in already-saved graphs (mask_mode
                # landing in this slot, mask_feather in mask_mode, and so on) with no error shown.
                # A new widget at the end leaves old workflows reading exactly as before.
                "control_type": ("STRING", {"default": "canny", "tooltip": "z-image-controlnet only. What each connected control image is: canny, depth, hed, mlsd or pose. One value applies to every image; a comma-separated list ('canny,depth') pairs with the MfluxImage chain in order, which is how you stack controls. Ignored by every other model."}),
                # The block below is also appended last (same widgets_values reasoning as above).
                # Every widget's default is inert: an untouched graph forwards nothing new.
                "pid_decode": ("BOOLEAN", {"default": False, "tooltip": "Decode through NVIDIA's PiD pixel-diffusion decoder instead of the VAE: a 4x super-resolving re-render (512 in, 2048 out). Downloads one PiD checkpoint per model family plus the gated google/gemma-2-2b-it on first use. The live preview stays VAE-based at base resolution, so it will not match the 4x output. Dropped with a note on models without PiD support."}),
                "pid_degrade_sigma": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "PiD only: degrade the source latent before re-rendering, trading fidelity for invented detail. 0 = off."}),
                "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 12.0, "step": 0.1, "tooltip": "Z-Image: override the automatic sigma shift (mu). 0 = model default. Dropped with a note on models that do not take it."}),
                "sigma_schedule": ("STRING", {"default": "", "tooltip": "Z-Image: sigma schedule shape — linear (default), cosine, karras or exponential. Empty = model default. Dropped with a note on models that do not take it."}),
                "mcf_max_change": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01, "tooltip": "Z-Image MCF sampler: cap the mean absolute change per denoising step (typical 0.05-0.5). 0 = off. Dropped with a note on models that do not take it."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    FUNCTION = "generate"
    CATEGORY = "MLX/mflux"

    def generate(self, model, prompt, params_mode, negative_prompt="", seed=0,
                 steps=0, guidance=0.0, width=1024, height=1024, scheduler="auto",
                 preset="(model default)", strict_caption_validation=False,
                 image=None, image_strength=0.6, mflux_image=None, control_type="canny",
                 mask_mode="preserve", mask_feather=4,
                 live_preview=True, preview_stride=2,
                 pid_decode=False, pid_degrade_sigma=0.0, shift=0.0, sigma_schedule="",
                 mcf_max_change=0.0, unique_id=None):
        handle = model
        prompt = _clean_caption_prompt(prompt)
        if handle.free_comfy_first:
            _free_comfy()

        # resolve the image payload (typed MfluxImage wins over the simple image)
        payload = mflux_image
        if payload is None and image is not None:
            payload = MfluxImagePayload(images=[image], strength=image_strength)

        temps = []

        def materialize(t):
            p = _image_to_temp_png(t)
            temps.append(p)
            return p

        try:
            image_paths = None
            if payload is not None:
                prim_paths = [materialize(t) for t in payload.images]
                # per-image strengths aligned with the references (redux blend); fall back to the
                # scalar for every image if the payload predates per-image strengths.
                scalar_s = payload.strength if payload.strength is not None else image_strength
                per_image = payload.strengths if payload.strengths else [scalar_s] * len(prim_paths)
                image_paths = {
                    "primary": prim_paths[0],           # scalar image roles use the first (viewpoint) ref
                    "primaries": prim_paths,            # list image roles (edit) get every reference
                    "mask": materialize(payload.mask) if payload.mask is not None else None,
                    "aux": materialize(payload.aux) if payload.aux is not None else None,
                    "strength": scalar_s,
                    "strengths": [float(s) for s in per_image],
                    "control_type": control_type,   # only read by the z-image-controlnet role
                }
            requested = {
                "seed": int(seed), "prompt": prompt, "negative_prompt": negative_prompt,
                "steps": steps, "guidance": guidance, "width": int(width), "height": int(height),
                "scheduler": scheduler, "preset": preset,
                "strict_caption_validation": strict_caption_validation,
                "pid_decode": bool(pid_decode), "pid_degrade_sigma": float(pid_degrade_sigma),
                "shift": float(shift), "sigma_schedule": (sigma_schedule or "").strip(),
                "mcf_max_change": float(mcf_max_change),
            }
            forwarded, notes = normalize_and_validate(handle.profile, params_mode, requested, image_paths)
            if forwarded.get("pid_decode") and live_preview:
                notes.append("[mflux] pid_decode: the live preview decodes through the VAE at base resolution; the final image is PiD's 4x re-render and will differ.")
            if live_preview:
                from .preview import ComfyLivePreview, _RegistryGuard, resolve_latent_creator
                lc = resolve_latent_creator(handle.family)
                cb = ComfyLivePreview(handle.instance, lc, node_id=unique_id, stride=int(preview_stride))
                with _RegistryGuard(handle.instance, cb):
                    gen = handle.instance.generate_image(**forwarded)
            else:
                gen = handle.instance.generate_image(**forwarded)
        finally:
            for p in temps:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

        if getattr(handle, "lora_note", ""):
            notes = [handle.lora_note] + notes
        info = f"model={handle.alias} family={handle.family} forwarded={sorted(forwarded)}"

        # Families return a GeneratedImage (metadata + .image), except where they do not: the
        # Z-Image ControlNet is annotated `-> Image.Image` and in fact returns a GeneratedImage.
        # Read whichever actually came back rather than trusting the annotation either way.
        gen_pil = getattr(gen, "image", gen)

        # Mask-preserve composite: for edit/img2img models that do NOT consume a mask natively,
        # a connected mask lets painted regions stay locked to the original (hard-lock openings)
        # or restrict the edit to the painted region. flux-fill consumes its mask natively -> skip.
        out_image = _pil_to_image(gen_pil)
        native_mask = bool(set(handle.profile.image_role_args) & MASK_ARGS)
        if (payload is not None and payload.mask is not None and not native_mask
                and mask_mode != "off" and payload.images):
            try:
                out_image = _preserve_composite(
                    original_tensor=payload.images[0], edited_pil=gen_pil,
                    mask_tensor=payload.mask, keep_white=(mask_mode == "preserve"),
                    feather=int(mask_feather),
                )
                notes.append(f"[mflux] mask composite: mode={mask_mode}, feather={mask_feather}px "
                             f"(painted={'kept from original' if mask_mode == 'preserve' else 'took the edit'}).")
            except Exception as e:
                notes.append(f"[mflux] mask composite skipped ({type(e).__name__}: {e}).")

        if notes:
            info += "\n" + "\n".join(notes)
            for n in notes:
                print(n)
        return (out_image, info)


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
                "resolution": ("STRING", {"default": "2x", "tooltip": "Scale factor (e.g. 2x, 1.5x) or shortest-edge target in pixels (e.g. 1080)."}),
            },
            "optional": {
                "softness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
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
            gen = inst.generate_image(seed=int(seed), image_path=path, resolution=_parse_resolution(resolution), softness=float(softness))
        finally:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        return (_pil_to_image(gen.image),)
