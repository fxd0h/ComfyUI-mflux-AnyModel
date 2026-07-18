"""Vision-language node for ComfyUI-mflux-AnyModel, running natively on MLX.

MfluxVLM wraps mflux's FiboVLM (Qwen3-VL: vision tower + decoder, all in MLX), so a workflow can
read a room photo and write the prompt for it without leaving Apple Silicon. It replaces the
llama.cpp mmproj route: the vision encoder and its projector are native here, not a GGUF side-file.

What this model actually is, verified by running it rather than by reading its card:

FIBO-vlm ALWAYS answers in FIBO's structured JSON. `inspire` and `generate` are control tokens
(`<inspire>`, `<generate>`) the model was finetuned on, and every task but `edit` is passed through
`_format_json_output`. There is no system prompt to talk it out of JSON; asking for a paragraph gets
you JSON that ignores the request. So the JSON is the survey, and _to_prose() renders it into the
dense paragraph FLUX and Krea 2 want. Both outputs are exposed: `prompt` (prose, for the samplers)
and `survey` (the raw JSON, readable and reusable).

Three modes, each mapped to the FIBO-vlm method that actually does that job (verified by running
both on a real room + a deliberately violent override brief):
  analyze   inspire(image)          describe the room that is actually there              ~40s
  expand    generate(brief)         turn YOUR brief into a scene, no photo                ~15s
  renovate  edit(image, brief)      apply YOUR changes to the real room, keeping geometry  ~11min

The distinction that matters: inspire(image, prompt=brief) reads as "describe this photo, loosely
inspired by the text" — it keeps the room but IGNORES material/color overrides (ask for emerald
walls, get the photo's white). edit() runs FIBO's FIDELITY RULE system prompt, which changes only
what the instruction asks and keeps the rest, so "emerald walls, moody dusk" actually lands. edit()
is far slower (long system prompt + full token budget), but it is a once-per-brief step, not per-seed.
"""
import json
import re

import numpy as np

MODES = ["analyze (mirá la foto y describí el ambiente)",
         "expand (armá la escena desde MI brief, sin foto)",
         "renovate (aplicá MIS cambios a la foto real)"]

_VLM_CACHE = {"key": None, "model": None}


def _load_vlm(model_id, quantize):
    key = (model_id, quantize)
    if _VLM_CACHE["key"] == key and _VLM_CACHE["model"] is not None:
        return _VLM_CACHE["model"]
    from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
    model = FiboVLM(model_id=model_id, quantize=quantize)
    _VLM_CACHE.update(key=key, model=model)
    return model


def _loads_tolerant(text):
    """Parse FIBO's JSON, salvaging what is there when the token budget cut it short.

    A truncated answer is still worth most of its content: the fields FIBO emits first
    (short_description, the early objects) are the ones that carry the scene. Rather than lose
    everything to a dangling string, close the open brackets and parse the prefix.
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    stack, in_str, esc, last_safe = [], False, False, None
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
        elif ch == "," and not stack[1:]:
            last_safe = i  # a comma at depth 1 closes a complete top-level field
    if last_safe is None:
        return None
    repaired = text[:last_safe]
    depth, in_str, esc = [], False, False
    for ch in repaired:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            depth.append("}" if ch == "{" else "]")
        elif ch in "}]" and depth:
            depth.pop()
    try:
        return json.loads(repaired + "".join(reversed(depth)))
    except json.JSONDecodeError:
        m = re.search(r'"short_description"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        return {"short_description": m.group(1)} if m else None


def _join(parts):
    out, seen = [], set()
    for p in parts:
        p = (str(p or "")).strip().rstrip(".")
        k = p.lower()
        if p and k not in seen:
            seen.add(k)
            out.append(p)
    return ". ".join(out) + "." if out else ""


def _to_prose(data, max_objects=6):
    """Render FIBO's scene JSON into the dense visual paragraph a diffusion text encoder wants.

    FLUX and Krea 2 read prose through T5; FIBO's JSON is for FIBO. Ordering matters more than
    completeness here, so lead with the summary and setting, then the objects with their placement,
    then light, palette and style. Fields FIBO omits are simply skipped.
    """
    if not isinstance(data, dict):
        return ""
    parts = [data.get("short_description"), data.get("background_setting")]

    for obj in (data.get("objects") or [])[:max_objects]:
        if not isinstance(obj, dict):
            continue
        desc = (obj.get("description") or "").strip()
        if not desc:
            continue
        loc = (obj.get("location") or "").strip()
        parts.append(f"{desc.rstrip('.')}, {loc}" if loc else desc)

    light = data.get("lighting")
    if isinstance(light, dict):
        parts.append(_join([light.get("conditions"), light.get("direction"), light.get("shadows")]))
    elif light:
        parts.append(light)

    aes = data.get("aesthetics")
    if isinstance(aes, dict):
        parts.append(_join([aes.get("composition"), aes.get("color_palette"),
                            aes.get("mood_atmosphere"), aes.get("artistic_style")]))
    elif aes:
        parts.append(aes)

    for k in ("style_medium", "camera_settings", "photography_style"):
        v = data.get(k)
        if isinstance(v, dict):
            v = _join(list(v.values()))
        parts.append(v)

    return _join(parts)


class MfluxVLM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (MODES, {"default": MODES[2], "tooltip": "analyze: describe la foto tal cual es (~40s). expand: arma la escena desde tu brief, sin foto (~15s). renovate: aplica TUS cambios (materiales, colores, luz) a la foto real y mantiene la geometria (~11min, pero es la unica que respeta de verdad lo que pedis)."}),
                "brief": ("STRING", {"multiline": True, "default": "replace the floor with dark walnut planks, warm modern palette, add a three-seat sofa and an 85-inch TV, cove lighting, keep the windows and layout", "tooltip": "En 'renovate' escribí los CAMBIOS como ordenes ('replace the floor with...', 'repaint the walls...'). En 'expand' es el programa completo. Escribilo en INGLES: el VLM contesta en el idioma del brief y el T5 de FLUX/Krea2 entiende ingles. Ignorado en 'analyze'."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "La foto del ambiente. Requerida en 'analyze' y 'analyze + expand'; ignorada en 'expand'."}),
                "model_id": ("STRING", {"default": "briaai/FIBO-vlm", "tooltip": "Pesos del VLM (Qwen3-VL en MLX). Tiene que ser briaai/FIBO-vlm: un Qwen3-VL generico NO carga (otro layout de pesos)."}),
                "quantize": (["8", "4", "none"], {"default": "8", "tooltip": "8 bits entra comodo y carga en ~6s."}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Bajo = obediente y factual. Alto = explora."}),
                "max_tokens": ("INT", {"default": 4096, "min": 256, "max": 8192, "tooltip": "El JSON de FIBO es largo: con menos de ~1500 se corta. Si se corta igual, el nodo rescata lo que llego. En 'renovate' lo fija el metodo edit() (4096)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
                "keep_loaded": ("BOOLEAN", {"default": True, "tooltip": "Dejar el VLM en memoria entre corridas (~9GB en 8 bits)."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "survey")
    FUNCTION = "run"
    CATEGORY = "MLX/mflux/inputs"

    def run(self, mode, brief, image=None, model_id="briaai/FIBO-vlm", quantize="8",
            temperature=0.2, max_tokens=4096, seed=0, keep_loaded=True):
        from PIL import Image as PILImage

        is_analyze = mode.startswith("analyze")
        is_renovate = mode.startswith("renovate")
        needs_image = is_analyze or is_renovate
        needs_brief = is_renovate or mode.startswith("expand")
        if needs_image and image is None:
            raise ValueError("Los modos 'analyze' y 'renovate' necesitan la foto: conectá 'image'.")
        if needs_brief and not (brief or "").strip():
            raise ValueError(f"El modo '{mode.split()[0]}' necesita un brief.")

        vlm = _load_vlm(model_id, None if quantize == "none" else int(quantize))

        pil = None
        if needs_image:
            arr = (image[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil = PILImage.fromarray(arr).convert("RGB")

        try:
            if is_renovate:
                # edit() runs FIBO's FIDELITY RULE: change only what the instruction asks, keep the
                # rest of the real room. It is the only path that honors material/color overrides.
                # Its signature is fixed (image, edit_instruction, seed) — no temperature/max_tokens.
                survey = vlm.edit(image=pil, edit_instruction=brief.strip(), seed=int(seed))
            elif is_analyze:
                survey = vlm.inspire(image=pil, prompt=None, temperature=float(temperature),
                                     max_tokens=int(max_tokens), seed=int(seed))
            else:
                survey = vlm.generate(prompt=brief.strip(), temperature=float(temperature),
                                      max_tokens=int(max_tokens), seed=int(seed))

            survey = (survey or "").strip()
            data = _loads_tolerant(survey)
            prompt = _to_prose(data)
            if not prompt:
                # FIBO went off-script and answered in prose: pass it straight through.
                prompt = survey
                print("[mflux] VLM: respuesta no-JSON, la paso tal cual")
            print(f"[mflux] VLM {mode.split()[0]}: prompt={len(prompt)} chars, survey={len(survey)} chars")
            return (prompt, survey)
        finally:
            # evict the cached model even if generation or parsing raised
            if not keep_loaded:
                _VLM_CACHE.update(key=None, model=None)
