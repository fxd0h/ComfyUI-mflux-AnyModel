"""
Self-test for Phase 1 of the flexible mflux node (dispatch + capability).

Runs against the REAL installed mflux fork, no pytest, no weight downloads:
    python selftest_phase1.py
Exits non-zero on any failure. This is the anti-drift guard — if mflux renames a
class or changes a generate_image signature, these assertions catch it.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import mflux_dispatch as D
import capability as C
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
from mflux.models.ideogram4.variants.txt2img.ideogram4 import Ideogram4
from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
from mflux.cli.defaults import defaults as ui
from mflux.models.common.config.model_config import AVAILABLE_MODELS

_checks = 0


def ok(cond, msg):
    global _checks
    _checks += 1
    if not cond:
        print(f"  FAIL: {msg}")
        raise AssertionError(msg)


# ---- 1. dispatch: no variant alias silently degrades to a base txt2img class ----
_BASE_TXT2IMG = {Flux1, FIBO}  # the classes a flux-/fibo- variant would wrongly fall back to
for alias in D.ALIAS_DISPATCH:
    cls, family = D.pick_model_class(alias)
    ok(cls not in _BASE_TXT2IMG, f"{alias} degraded to a base txt2img class (trap reintroduced)")
    ok(family.startswith("flux-") or family.startswith("fibo-"), f"{alias} family={family} unexpected")
print("1. variant aliases dispatch to dedicated classes (no txt2img degradation) OK")

# ---- 2. every dropdown-eligible alias resolves to a config + class ----
dropdown = [m for m in (list(ui.MODEL_CHOICES) + list(AVAILABLE_MODELS))
            if m not in D.SEEDVR2_ALIASES]
for alias in dropdown:
    cls, family, cfg, path = D.resolve_config_and_path(alias)
    ok(cls is not None and cfg is not None, f"{alias} did not resolve")
    ok(path is None, f"builtin {alias} should resolve to path None, got {path!r}")
    ok(D.is_builtin_name(alias) or alias == "ideogram4", f"{alias} not detected builtin")
print(f"2. all {len(dropdown)} builtin aliases resolve (class+config, path=None) OK")

# ---- 3. constructor-kwarg adaptation (pure, no weight load) ----
kw_ideo = D.build_kwargs(Ideogram4, quantize=8, bake_lora=True, lora_paths=["x"], lora_scales=[1.0])
ok("bake_lora" not in kw_ideo, "Ideogram4 must NOT receive bake_lora")
ok("lora_paths" in kw_ideo and "model_config" in kw_ideo, "Ideogram4 must accept lora + config")
kw_cn = D.build_kwargs(Flux1Controlnet, controlnet_path="cn", bake_lora=True)
ok("controlnet_path" in kw_cn, "Flux1Controlnet must receive controlnet_path")
ok("bake_lora" in kw_cn, "Flux1Controlnet must receive bake_lora")
kw_flux = D.build_kwargs(Flux1)
ok("controlnet_path" not in kw_flux, "Flux1 must NOT receive controlnet_path")
print("3. build_kwargs adapts to each constructor (ideo no bake_lora, cn has controlnet_path) OK")

# ---- 4. capability profiles (introspection + small tables) ----
def prof(alias_or_cls, family=None):
    if isinstance(alias_or_cls, str):
        cls, fam = D.pick_model_class(alias_or_cls)
        return C.build_profile(cls, fam)
    return C.build_profile(alias_or_cls, family)

p_ideo = prof("ideogram4")
ok(p_ideo.is_preset_only, "ideogram4 must be preset_only")
ok(p_ideo.presets, "ideogram4 must list presets")
ok(not p_ideo.image_role_args, "ideogram4 takes no image input")
ok(not p_ideo.supports_negative, "ideogram4 has no negative_prompt")

p_flux = prof("dev")
ok(p_flux.supports_negative is False, "flux negative_prompt is inert -> supports_negative False")
ok(p_flux.supports_img2img, "flux supports img2img")

p_qwen = prof("qwen")
ok(p_qwen.supports_negative is True, "qwen negative_prompt is effective")

p_z = prof("z-image-turbo")
ok({"shift", "sigma_schedule", "mcf_max_change"} <= p_z.gen_kwargs, "z-image missing #353 kwargs")

p_fill = prof("dev-fill")
ok("masked_image_path" in p_fill.required_image_args, "fill must require masked_image_path")
ok("image_path" in p_fill.required_image_args, "fill must require image_path")

p_redux = prof("dev-redux")
ok("redux_image_paths" in p_redux.required_image_args, "redux must require redux_image_paths")

p_cn = prof("dev-controlnet-canny")
ok("controlnet_image_path" in p_cn.required_image_args, "controlnet must require controlnet_image_path")

p_edit = prof("qwen-image-edit")
ok("image_paths" in p_edit.required_image_args, "qwen-edit must require image_paths")

p_f2e = prof(Flux2KleinEdit, "flux2-edit")
ok("guidance" in p_f2e.gen_kwargs, "flux2-edit should expose guidance (PR #421 in 0.18.0)")
print("4. capability profiles correct (preset-only, negative inert/effective, image roles) OK")

print(f"\nALL PHASE-1 SELF-TESTS PASSED ({_checks} checks)")
