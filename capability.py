"""
Capability registry for the flexible mflux node.

A CapabilityProfile is built per loaded model and carried on the MFLUX_MODEL
handle. The sampler reads it to decide which params it honors, ignores-with-warning,
or hard-blocks — which is what structurally closes the silent steps/guidance
degradation trap and prevents feeding image inputs to a model that ignores them.

Mostly derived by introspection from each variant's generate_image signature
(so new mflux kwargs are auto-tracked). Only the facts that are NOT derivable —
or are *wrongly* derivable — from the signature live in the small tables below.
"""
import inspect
from dataclasses import dataclass, field

# --- facts the signature does not (or wrongly) reveal ---

# Preset-driven models: the preset fixes steps + per-step guidance + noise schedule
# together, so forwarding loose steps/guidance silently decalibrates them.
PRESET_ONLY = {"ideogram4"}

# negative_prompt is present in flux's signature but encode_prompt never receives it
# (verified inert). qwen / z-image / fibo / ernie do thread it through -> effective.
NEGATIVE_INERT = {"flux"}

# generate_image params that denote an input image the model consumes.
IMAGE_ROLE_PARAMS = {
    "image_path", "masked_image_path", "depth_image_path", "redux_image_paths",
    "controlnet_image_path", "image_paths", "left_image_path", "right_image_path",
    # Z-Image Union ControlNet takes ControlSpec objects rather than paths, but it is still an
    # image role: without it the sampler would read `controls` as an unsatisfiable required arg
    # and refuse to drive the model at all.
    "controls",
}


def presets_for(family: str):
    if family not in PRESET_ONLY:
        return []
    try:
        from mflux.models.ideogram4.model.ideogram4_scheduler import Ideogram4Scheduler
        return sorted(Ideogram4Scheduler.PRESETS)
    except Exception:
        return []


@dataclass
class CapabilityProfile:
    family: str
    gen_kwargs: set                              # introspected generate_image params
    image_role_args: dict = field(default_factory=dict)  # {param: 'req'|'opt'}
    supports_guidance: bool = True
    supports_negative: bool = False
    supports_img2img: bool = False
    supports_lora: bool = True
    is_preset_only: bool = False
    presets: list = field(default_factory=list)
    required_gen_params: set = field(default_factory=set)  # generate_image params with no default

    @property
    def required_image_args(self):
        return {k for k, v in self.image_role_args.items() if v == "req"}

    @property
    def needs_image(self):
        return bool(self.required_image_args)

    @property
    def phase2_drivable(self):
        """True if the sampler can supply every required generate_image arg in
        Phase 2 (seed/prompt + optional img2img). Variant/edit families with a
        required image role are excluded until the typed feeders ship (Phase 3)."""
        return not (self.required_gen_params - {"seed", "prompt"})


def build_profile(cls, family: str, model_config=None) -> CapabilityProfile:
    sig = inspect.signature(cls.generate_image)
    params = sig.parameters
    gen_kwargs = {p for p in params if p != "self"}

    image_role_args = {}
    for name in gen_kwargs & IMAGE_ROLE_PARAMS:
        p = params[name]
        image_role_args[name] = "opt" if p.default is not inspect.Parameter.empty else "req"

    required_gen_params = {
        name for name, p in params.items()
        if name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }

    ctor_params = set(inspect.signature(cls.__init__).parameters)
    mc_guidance = getattr(model_config, "supports_guidance", True)

    return CapabilityProfile(
        family=family,
        gen_kwargs=gen_kwargs,
        image_role_args=image_role_args,
        supports_guidance=("guidance" in gen_kwargs) and (mc_guidance is not False),
        supports_negative=("negative_prompt" in gen_kwargs) and (family not in NEGATIVE_INERT),
        supports_img2img={"image_path", "image_strength"} <= gen_kwargs,
        supports_lora="lora_paths" in ctor_params,
        is_preset_only=(family in PRESET_ONLY),
        presets=presets_for(family),
        required_gen_params=required_gen_params,
    )
