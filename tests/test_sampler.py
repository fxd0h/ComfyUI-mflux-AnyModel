"""
Self-test for the flexible mflux node sampler logic (Phase 2 + Phase 3).
No pytest, no weight loads:
    python selftest_phase2.py
Drives normalize_and_validate / inject_image through the cases that matter:
the preset trap, negative inert/effective, image gating, and per-family image-role
injection (img2img, fill, depth, redux, controlnet, qwen-edit, fibo-edit).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mflux_dispatch as D
import capability as C
from nodes import normalize_and_validate, model_choices, QUANTIZE_CHOICES, strip_mark

_checks = 0


def ok(cond, msg):
    global _checks
    _checks += 1
    if not cond:
        raise AssertionError("FAIL: " + msg)


def prof(alias):
    cls, fam = D.pick_model_class(alias)
    return C.build_profile(cls, fam)


def req(**over):
    base = dict(seed=42, prompt="x", negative_prompt="", steps=0, guidance=0.0,
                width=1024, height=1024, scheduler="auto", preset="(model default)",
                strict_caption_validation=False)
    base.update(over)
    return base


def img(primary="/tmp/p.png", mask=None, aux=None, strength=0.6):
    return {"primary": primary, "mask": mask, "aux": aux, "strength": strength}


# --- the steps/guidance trap on Ideogram (preset-only) ---
pi = prof("ideogram4")
fwd, notes = normalize_and_validate(pi, "auto", req(steps=8, guidance=5.0), None)
ok("num_inference_steps" not in fwd, "auto: steps must NOT reach a preset-only model")
ok("guidance" not in fwd, "auto: guidance must NOT reach a preset-only model")
ok(any("IGNORED" in n for n in notes), "auto: must warn that steps/guidance are ignored")
fwd, notes = normalize_and_validate(pi, "override", req(steps=8, guidance=5.0), None)
ok(fwd.get("num_inference_steps") == 8, "override: steps must be forwarded")
ok(any("degrade" in n for n in notes), "override: must warn about degradation")
fwd, _ = normalize_and_validate(pi, "auto", req(preset="V4_TURBO_12"), None)
ok(fwd.get("preset") == "V4_TURBO_12", "ideogram preset must be forwarded")
print("1. Ideogram preset trap OK")

# --- non-preset honors steps/guidance ---
pf = prof("dev")
fwd, _ = normalize_and_validate(pf, "auto", req(steps=20, guidance=3.5), None)
ok(fwd.get("num_inference_steps") == 20 and fwd.get("guidance") == 3.5, "flux honors steps/guidance")
print("2. non-preset honors steps/guidance OK")

# --- negative inert on flux / effective on qwen ---
fwd, notes = normalize_and_validate(pf, "auto", req(negative_prompt="blurry"), None)
ok("negative_prompt" not in fwd and any("negative_prompt" in n for n in notes), "flux negative dropped+noted")
fwd, _ = normalize_and_validate(prof("qwen"), "auto", req(negative_prompt="blurry"), None)
ok(fwd.get("negative_prompt") == "blurry", "qwen negative forwarded")
print("3. negative_prompt inert/effective OK")

# --- image gating ---
fwd, _ = normalize_and_validate(pf, "auto", req(), img())
ok(fwd.get("image_path") == "/tmp/p.png" and "image_strength" in fwd, "flux img2img passes")
try:
    normalize_and_validate(pi, "auto", req(), img()); ok(False, "ideogram must reject image")
except ValueError:
    ok(True, "ideogram rejects image")
try:
    normalize_and_validate(prof("dev-fill"), "auto", req(), None); ok(False, "fill must require image")
except ValueError:
    ok(True, "fill requires image")
print("4. image gating OK")

# --- scheduler auto/explicit ---
ok("scheduler" not in normalize_and_validate(pf, "auto", req(scheduler="auto"), None)[0], "auto scheduler not forwarded")
ok(normalize_and_validate(pf, "auto", req(scheduler="linear"), None)[0].get("scheduler") == "linear", "explicit scheduler forwarded")
print("5. scheduler handling OK")

# --- Phase 3: per-family image-role injection ---
# fill: primary + mask -> image_path + masked_image_path
fwd, _ = normalize_and_validate(prof("dev-fill"), "auto", req(), img(mask="/tmp/m.png"))
ok(fwd.get("image_path") == "/tmp/p.png" and fwd.get("masked_image_path") == "/tmp/m.png", "fill wires image+mask")
# fill without mask -> clear error
try:
    normalize_and_validate(prof("dev-fill"), "auto", req(), img()); ok(False, "fill needs mask")
except ValueError:
    ok(True, "fill without mask hard-blocks")
# depth: primary + map -> image_path + depth_image_path
fwd, _ = normalize_and_validate(prof("dev-depth"), "auto", req(), img(aux="/tmp/d.png"))
ok(fwd.get("depth_image_path") == "/tmp/d.png", "depth wires map -> depth_image_path")
# redux: primary -> redux_image_paths=[primary] (+ strength list)
fwd, _ = normalize_and_validate(prof("dev-redux"), "auto", req(), img())
ok(fwd.get("redux_image_paths") == ["/tmp/p.png"], "redux wires list")
ok(fwd.get("redux_image_strengths") == [0.6], "redux strength list")
# controlnet: primary -> controlnet_image_path (+ controlnet_strength)
fwd, _ = normalize_and_validate(prof("dev-controlnet-canny"), "auto", req(), img())
ok(fwd.get("controlnet_image_path") == "/tmp/p.png", "controlnet wires image")
# qwen-edit: primary -> image_paths=[primary]
fwd, _ = normalize_and_validate(prof("qwen-image-edit"), "auto", req(), img())
ok(fwd.get("image_paths") == ["/tmp/p.png"], "qwen-edit wires image_paths list")
# fibo-edit: primary -> image_path
fwd, _ = normalize_and_validate(prof("fibo-edit"), "auto", req(), img())
ok(fwd.get("image_path") == "/tmp/p.png", "fibo-edit wires image_path")
print("6. Phase-3 variant image-role injection OK")

# --- dispatch fixes from review ---
from mflux.models.fibo.variants.edit.fibo_edit import FIBOEdit
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.flux.variants.txt2img.flux import Flux1
ok(D.pick_model_class("fibo-edit")[0] is FIBOEdit and D.pick_model_class("fibo")[0] is FIBO, "fibo-edit/fibo dispatch")
for a in ("klein-4b", "klein-9b-kv"):
    ok(D.pick_model_class(a)[0] is Flux2Klein, f"{a} -> Flux2Klein")
ok(D.pick_model_class("dev")[0] is Flux1, "dev -> Flux1")
for a in ("seedvr2", "seedvr2-7B"):
    try:
        D.resolve_config_and_path(a); ok(False, f"{a} must raise")
    except ValueError:
        ok(True, f"{a} guarded")
ok(D.resolve_config_and_path("ideogram4", "", "/local/x")[3] == "/local/x", "ideogram model_path honored")
print("7. dispatch fixes (fibo-edit/klein/seedvr2/ideogram path) OK")

# --- anti-drift: no variant produces a forwarded dict missing a required arg ---
for alias, (cls, fam) in D.ALIAS_DISPATCH.items():
    p = C.build_profile(cls, fam)
    try:
        fwd, _ = normalize_and_validate(p, "auto", req(), img(mask="/tmp/m.png", aux="/tmp/d.png"))
        ok(not (p.required_gen_params - set(fwd)),
           f"{alias}: forwarded missing {sorted(p.required_gen_params - set(fwd))}")
    except ValueError:
        ok(True, f"{alias} hard-blocks cleanly")
# override warning only when steps/guidance set
ok(not any("degrade" in n for n in normalize_and_validate(pi, "override", req(), None)[1]), "override 0/0 no warn")
print("8. anti-drift + override gating OK")

# --- dropdown: base + wired variants in, unwired/seedvr2 out (marks stripped) ---
mc = [strip_mark(m) for m in model_choices()]
ok("schnell" in mc and "ideogram4" in mc, "base models in dropdown")
ok("dev-fill" in mc and "dev-kontext" in mc and "qwen-image-edit" in mc, "wired variants in dropdown")
ok("dev-fill-catvton" not in mc and "dev-controlnet-upscaler" not in mc, "unwired variants excluded")
ok(not any(s in mc for s in D.SEEDVR2_ALIASES), "seedvr2 excluded")
ok("none" in QUANTIZE_CHOICES and "8" in QUANTIZE_CHOICES, "quantize choices")
print("9. dropdown (base + wired variants) OK")

print(f"\nALL SELF-TESTS PASSED ({_checks} checks)")
