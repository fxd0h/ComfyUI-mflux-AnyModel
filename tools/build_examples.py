"""Generate a library of small, focused example workflows for ComfyUI-mflux-AnyModel.

Each file demonstrates ONE capability so users can try everything the node does. Generated from
the live /object_info (widget order/defaults are exact). Layout: a note on top, then a left-to-right
row of nodes (stride > node width, so nothing overlaps). All text is English.
"""
import json, os, urllib.request

with urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=30) as _r:
    OI = json.load(_r)
LINKY = {"IMAGE", "MASK", "MFLUX_MODEL", "MFLUX_IMAGE", "MFLUX_LORA", "LATENT", "CONDITIONING", "STRING", "*"}
OUTDIR = os.path.expanduser("~/Documents/github/ComfyUI/user/default/workflows/mflux_examples")
os.makedirs(OUTDIR, exist_ok=True)


def spec(node):
    info = OI[node]["input"]; sockets, widgets = [], []
    for sec in ("required", "optional"):
        for k, v in (info.get(sec) or {}).items():
            t = v[0] if isinstance(v, list) else v
            opts = v[1] if isinstance(v, list) and len(v) > 1 and isinstance(v[1], dict) else {}
            if isinstance(t, str) and t in LINKY and t != "STRING":
                sockets.append((k, t))
            else:
                widgets.append((k, opts.get("default", (t[0] if isinstance(t, list) and t else ""))))
                if opts.get("control_after_generate"):
                    widgets.append(("__cag", "randomize"))
    return sockets, widgets


class WF:
    def __init__(self):
        self.nodes, self.links, self.groups, self.nid, self.lid = [], [], [], 0, 0

    def add(self, type_, pos, ov=None, title=None, color=None, size=None, mode=0):
        self.nid += 1; sockets, widgets = spec(type_)
        vals = [(ov or {}).get(k, d) for k, d in widgets]
        ins = [{"name": k, "type": t, "link": None} for k, t in sockets]
        outs = [{"name": n, "type": (OI[type_]["output"][i] if i < len(OI[type_]["output"]) else n),
                 "links": [], "slot_index": i}
                for i, n in enumerate(OI[type_]["output_name"] or OI[type_]["output"])]
        n = {"id": self.nid, "type": type_, "pos": [float(pos[0]), float(pos[1])],
             "size": size or [340, max(58, 34 + 26 * len(vals))], "flags": {}, "order": self.nid - 1,
             "mode": mode, "inputs": ins, "outputs": outs,
             "properties": {"Node name for S&R": type_}, "widgets_values": vals}
        if title:
            n["title"] = title
        if color:
            n["color"], n["bgcolor"] = color
        self.nodes.append(n); return self.nid

    def note(self, pos, text, size, title="Guide", color=("#2a363b", "#1e2a30")):
        self.nid += 1
        self.nodes.append({"id": self.nid, "type": "MarkdownNote", "pos": [float(pos[0]), float(pos[1])],
                           "size": [float(size[0]), float(size[1])], "flags": {}, "order": self.nid - 1,
                           "mode": 0, "inputs": [], "outputs": [], "title": title, "properties": {},
                           "widgets_values": [text], "color": color[0], "bgcolor": color[1]})
        return self.nid

    def link(self, src, slot, dst, dname):
        self.lid += 1
        s = next(n for n in self.nodes if n["id"] == src); d = next(n for n in self.nodes if n["id"] == dst)
        s["outputs"][slot]["links"].append(self.lid)
        di = next(i for i, x in enumerate(d["inputs"]) if x["name"] == dname); d["inputs"][di]["link"] = self.lid
        self.links.append([self.lid, src, slot, dst, di, s["outputs"][slot]["type"]])

    def dump(self):
        return {"last_node_id": self.nid, "last_link_id": self.lid, "nodes": self.nodes,
                "links": self.links, "groups": self.groups, "config": {}, "extra": {}, "version": 0.4}


BLUE, GREEN, PURPLE, RED, GREY, ORANGE = "#3f789e", "#37734a", "#7b3f9e", "#9e3f3f", "#555", "#9e6b3f"
NX0, NY, DX = 20, 300, 360   # node row: start x, y, x-stride
FIRST_LORA = next((m for m in (OI["MfluxLora"]["input"]["required"]["lora_name"][0] or []) if m), "")


def col(i):
    return (NX0 + i * DX, NY)


def save(w, name, ndesc):
    with open(os.path.join(OUTDIR, name), "w") as _f:
        json.dump(w.dump(), _f, indent=1, ensure_ascii=False)
    print(f"  {name:34} nodes={len(w.nodes)} links={len(w.links)}  ({ndesc})")


# --------------------------------------------------------------------------- #
def ex_txt2img():
    w = WF()
    w.note((20, 20), """# 01 · Text to image (any model)

The base capability: pick **any** of the models in the Loader dropdown and generate from a prompt.
`schnell` here is fast (4 steps). Swap it for `dev`, `krea-2`, `qwen`, `z-image-turbo`, `ideogram`, etc.
Turn on **live preview** in the Sampler to watch it denoise.""", (1040, 200), title="01 · Text to image")
    ld = w.add("MfluxModelLoader", col(0), {"model": "schnell", "quantize": "8"}, "Loader", color=(BLUE, "#233"), size=[340, 240])
    s = w.add("MfluxModelSampler", col(1), {"prompt": "a cozy scandinavian living room, soft daylight, photorealistic",
              "seed": 1, "width": 1024, "height": 768}, "Sampler", color=(GREEN, "#232"), size=[340, 440])
    pv = w.add("PreviewImage", col(2), {}, "Result", color=(GREY, "#222"), size=[340, 300])
    w.link(ld, 0, s, "model"); w.link(s, 0, pv, "images")
    save(w, "01_txt2img.json", "schnell")


def ex_img2img():
    w = WF()
    w.note((20, 20), """# 02 · Image to image

Feed a starting image straight into the Sampler's `image` input and set `image_strength`
(lower = closer to your image, higher = more freedom). Works on the base models.""", (1040, 170), title="02 · img2img")
    img = w.add("LoadImage", col(0), {"image": "room.png"}, "Start image", color=(BLUE, "#233"), size=[340, 300])
    ld = w.add("MfluxModelLoader", col(1), {"model": "schnell", "quantize": "8"}, "Loader", color=(BLUE, "#233"), size=[340, 240])
    s = w.add("MfluxModelSampler", col(2), {"prompt": "the same room, warm evening light, cinematic",
              "seed": 1, "image_strength": 0.55, "width": 1024, "height": 768}, "Sampler (image + strength)", color=(GREEN, "#232"), size=[340, 440])
    pv = w.add("PreviewImage", col(3), {}, "Result", color=(GREY, "#222"), size=[340, 300])
    w.link(img, 0, s, "image"); w.link(ld, 0, s, "model"); w.link(s, 0, pv, "images")
    save(w, "02_img2img.json", "schnell + image")


def ex_lora():
    w = WF()
    w.note((20, 20), f"""# 03 · LoRA

Stack a LoRA on any model. Pick the LoRA in `lora_name` and set `lora_scale`.
Chain more **mflux LoRA** nodes through `lora_in` to stack several.
(Default here: `{FIRST_LORA}` — change it to any LoRA you have installed.)""", (1040, 190), title="03 · LoRA")
    lora = w.add("MfluxLora", col(0), {"lora_name": FIRST_LORA, "lora_scale": 1.0}, "mflux LoRA", color=(ORANGE, "#332"), size=[340, 130])
    ld = w.add("MfluxModelLoader", col(1), {"model": "dev", "quantize": "8"}, "Loader (+ LoRA)", color=(BLUE, "#233"), size=[340, 260])
    s = w.add("MfluxModelSampler", col(2), {"prompt": "a portrait in the LoRA's style", "seed": 1,
              "steps": 20, "guidance": 3.5, "width": 1024, "height": 1024}, "Sampler", color=(GREEN, "#232"), size=[340, 440])
    pv = w.add("PreviewImage", col(3), {}, "Result", color=(GREY, "#222"), size=[340, 300])
    w.link(lora, 0, ld, "lora"); w.link(ld, 0, s, "model"); w.link(s, 0, pv, "images")
    save(w, "03_lora.json", "dev + LoRA")


def ex_edit_restyle():
    w = WF()
    w.note((20, 20), """# 04 · Restyle a room (edit model) ⭐

The star capability. **FLUX.2-klein-edit** keeps your room and applies your changes: adds furniture,
swaps the floor, materials, light. Write the prompt as commands and end with what to KEEP.
`mask_mode: off` restyles the whole room. Fast (~10-40s).""", (1040, 200), title="04 · Restyle (edit)")
    img = w.add("LoadImage", col(0), {"image": "room.png"}, "Your room", color=(BLUE, "#233"), size=[340, 300])
    ld = w.add("MfluxModelLoader", col(1), {"model": "flux2-klein-edit", "quantize": "8"}, "Loader · klein-edit", color=(GREEN, "#232"), size=[340, 260])
    mi = w.add("MfluxImage", col(2), {}, "Photo", color=(GREEN, "#232"), size=[340, 120])
    s = w.add("MfluxModelSampler", col(3), {"prompt": ("add a three-seat sofa and a wooden coffee table, "
              "replace the floor with dark walnut wood, warm modern styling, keep the room, the windows and the viewpoint. "
              "Clean professional interior photograph, no watermark, no text."), "seed": 3, "width": 1024, "height": 768,
              "mask_mode": "off"}, "Sampler · your changes", color=(GREEN, "#232"), size=[340, 440])
    pv = w.add("PreviewImage", col(4), {}, "Result", color=(GREY, "#222"), size=[340, 300])
    w.link(img, 0, mi, "image"); w.link(ld, 0, s, "model"); w.link(mi, 0, s, "mflux_image"); w.link(s, 0, pv, "images")
    save(w, "04_edit_restyle.json", "flux2-klein-edit")


def ex_replace_object():
    w = WF()
    w.note((20, 20), """# 05 · Replace an object (edit model)

The edit model can swap specific things: **"replace the TV with a fireplace"**,
"replace the speakers with plants", "remove the rug". End with "keep the rest of the room the same".
For surgical precision over WHERE, use example 06 (mask).""", (1040, 200), title="05 · Replace object")
    img = w.add("LoadImage", col(0), {"image": "room.png"}, "Your room", color=(BLUE, "#233"), size=[340, 300])
    ld = w.add("MfluxModelLoader", col(1), {"model": "flux2-klein-edit", "quantize": "8"}, "Loader · klein-edit", color=(GREEN, "#232"), size=[340, 260])
    mi = w.add("MfluxImage", col(2), {}, "Photo", color=(GREEN, "#232"), size=[340, 120])
    s = w.add("MfluxModelSampler", col(3), {"prompt": ("replace the TV with a modern wall-mounted fireplace showing a warm flame, "
              "keep the rest of the room, the windows and the viewpoint the same. Clean photograph, no watermark, no text."),
              "seed": 3, "width": 1024, "height": 768, "mask_mode": "off"}, "Sampler · replace", color=(GREEN, "#232"), size=[340, 440])
    pv = w.add("PreviewImage", col(4), {}, "Result", color=(GREY, "#222"), size=[340, 300])
    w.link(img, 0, mi, "image"); w.link(ld, 0, s, "model"); w.link(mi, 0, s, "mflux_image"); w.link(s, 0, pv, "images")
    save(w, "05_edit_replace_object.json", "flux2-klein-edit")


def ex_region_mask():
    w = WF()
    w.note((20, 20), """# 06 · Region-only edit (auto-mask)

Change ONLY one region, pixel-identical everywhere else. **Auto-mask** segments the photo (ADE20K):
pick `floor`, `walls`, `ceiling`, `windows`, `doors`, `furniture`... The Sampler's `mask_mode: inpaint`
restricts the edit to that region. `preserve` does the opposite (locks the region, changes the rest).""", (1120, 210), title="06 · Region edit")
    img = w.add("LoadImage", col(0), {"image": "room.png"}, "Your room", color=(BLUE, "#233"), size=[340, 300])
    am = w.add("MfluxAutoMask", col(1), {"region": "floor", "dilate": 6, "feather": 6}, "Auto-mask (floor)", color=(GREEN, "#232"), size=[340, 320])
    ld = w.add("MfluxModelLoader", col(2), {"model": "flux2-klein-edit", "quantize": "8"}, "Loader · klein-edit", color=(GREEN, "#232"), size=[340, 260])
    mi = w.add("MfluxImage", col(3), {}, "Photo + mask", color=(GREEN, "#232"), size=[340, 150])
    s = w.add("MfluxModelSampler", col(4), {"prompt": "replace the floor with dark walnut herringbone parquet",
              "seed": 3, "width": 1024, "height": 768, "mask_mode": "inpaint", "mask_feather": 6}, "Sampler · inpaint region", color=(GREEN, "#232"), size=[340, 440])
    pv = w.add("PreviewImage", col(5), {}, "Result", color=(GREY, "#222"), size=[340, 300])
    w.link(img, 0, am, "image"); w.link(img, 0, mi, "image"); w.link(am, 1, mi, "mask")
    w.link(ld, 0, s, "model"); w.link(mi, 0, s, "mflux_image"); w.link(s, 0, pv, "images")
    save(w, "06_edit_region_mask.json", "auto-mask + klein-edit")


def ex_controlnet_depth():
    w = WF()
    w.note((20, 20), """# 07 · Depth ControlNet (self-contained)

Lock the 3D perspective of a photo. **Depth map** is generated inside the workflow (DepthPro), so
nothing is prepared outside. Set the Sampler `width/height` to your photo's aspect and `steps` to ~24.
Good for a strict viewpoint; note it does not keep/add furniture (use example 04 for that).""", (1120, 210), title="07 · Depth ControlNet")
    img = w.add("LoadImage", col(0), {"image": "room.png"}, "Your room", color=(BLUE, "#233"), size=[340, 300])
    dm = w.add("MfluxDepthMap", col(1), {}, "Depth map (DepthPro)", color=(GREEN, "#232"), size=[340, 130])
    ld = w.add("MfluxModelLoader", col(2), {"model": "dev-controlnet-canny", "quantize": "8",
              "controlnet_path": "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"}, "Loader · FLUX.1 + depth CN", color=(RED, "#322"), size=[340, 260])
    mi = w.add("MfluxImage", col(3), {"strength": 0.7}, "ControlNet ← depth", color=(RED, "#322"), size=[340, 120])
    s = w.add("MfluxModelSampler", col(4), {"prompt": "a warm modern living room, dark walnut floor, cream walls, soft daylight",
              "params_mode": "override", "seed": 11, "steps": 24, "guidance": 3.5, "width": 1024, "height": 768}, "Sampler · steps 24", color=(RED, "#322"), size=[340, 440])
    pv = w.add("PreviewImage", col(5), {}, "Result", color=(GREY, "#222"), size=[340, 300])
    w.link(img, 0, dm, "image"); w.link(dm, 0, mi, "image"); w.link(ld, 0, s, "model")
    w.link(mi, 0, s, "mflux_image"); w.link(s, 0, pv, "images")
    save(w, "07_controlnet_depth.json", "dev + Union-Pro depth")


def ex_multi_controlnet():
    w = WF()
    w.note((20, 20), """# 08 · Multi-ControlNet (stack depth + canny)

Stack TWO controlnets and sum their guidance. **One checkpoint per line** in `controlnet_path`
(depth first, canny second) and one **mflux Image** per net, chained in the SAME order, each with its
own strength. Canny is derived from the photo automatically (by the checkpoint name).""", (1200, 210), title="08 · Multi-ControlNet")
    img = w.add("LoadImage", col(0), {"image": "room.png"}, "Your room", color=(BLUE, "#233"), size=[340, 300])
    dm = w.add("MfluxDepthMap", col(1), {}, "Depth map", color=(GREEN, "#232"), size=[340, 130])
    ld = w.add("MfluxModelLoader", col(2), {"model": "dev-controlnet-canny", "quantize": "8",
              "controlnet_path": "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0\nInstantX/FLUX.1-dev-Controlnet-Canny"},
              "Loader · depth + canny", color=(RED, "#322"), size=[340, 280])
    mi1 = w.add("MfluxImage", col(3), {"strength": 0.65}, "CN#1 ← depth (0.65)", color=(RED, "#322"), size=[340, 120])
    mi2 = w.add("MfluxImage", (NX0 + 3 * DX, NY + 200), {"strength": 0.35}, "CN#2 ← photo→canny (0.35)", color=(RED, "#322"), size=[340, 130])
    s = w.add("MfluxModelSampler", col(4), {"prompt": "a warm modern living room, walnut floor, cream walls, framed art",
              "params_mode": "override", "seed": 11, "steps": 24, "guidance": 3.5, "width": 1024, "height": 768}, "Sampler", color=(RED, "#322"), size=[340, 440])
    pv = w.add("PreviewImage", col(5), {}, "Result", color=(GREY, "#222"), size=[340, 300])
    w.link(img, 0, dm, "image"); w.link(dm, 0, mi1, "image"); w.link(mi1, 0, mi2, "image_in"); w.link(img, 0, mi2, "image")
    w.link(ld, 0, s, "model"); w.link(mi2, 0, s, "mflux_image"); w.link(s, 0, pv, "images")
    save(w, "08_multi_controlnet.json", "depth + canny stack")


def ex_redux():
    w = WF()
    w.note((20, 20), """# 09 · Mood board (Redux style transfer)

Transfer the **style** of one or more reference photos onto a new image. Each reference has its own
weight (`strength`): 1.0 dominates, 0.4 barely tints. Chain **mflux Image** nodes via `image_in` to
blend several. Redux makes a NEW image in that style; it does not keep a specific room.""", (1200, 210), title="09 · Redux mood board")
    ra = w.add("LoadImage", (NX0, NY), {"image": "room.png"}, "Style ref A", color=(PURPLE, "#323"), size=[340, 300])
    rb = w.add("LoadImage", (NX0, NY + 340), {"image": "room.png"}, "Style ref B", color=(PURPLE, "#323"), size=[340, 300])
    ld = w.add("MfluxModelLoader", col(1), {"model": "dev-redux", "quantize": "8"}, "Loader · FLUX.1 Redux", color=(PURPLE, "#323"), size=[340, 260])
    m1 = w.add("MfluxImage", col(2), {"strength": 1.0}, "Ref A (1.0)", color=(PURPLE, "#323"), size=[340, 120])
    m2 = w.add("MfluxImage", (NX0 + 2 * DX, NY + 200), {"strength": 0.5}, "Ref B (0.5)", color=(PURPLE, "#323"), size=[340, 120])
    s = w.add("MfluxModelSampler", col(3), {"prompt": "a bright modern living room interior in the referenced style",
              "seed": 42, "guidance": 2.5, "width": 1024, "height": 1024}, "Sampler", color=(GREEN, "#232"), size=[340, 440])
    pv = w.add("PreviewImage", col(4), {}, "Result", color=(GREY, "#222"), size=[340, 300])
    w.link(ra, 0, m1, "image"); w.link(m1, 0, m2, "image_in"); w.link(rb, 0, m2, "image")
    w.link(ld, 0, s, "model"); w.link(m2, 0, s, "mflux_image"); w.link(s, 0, pv, "images")
    save(w, "09_redux_moodboard.json", "dev-redux, 2 refs")


def ex_vlm():
    w = WF()
    w.note((20, 20), """# 10 · VLM prompt writer (FIBO-vlm, MLX)

Turn a photo and/or a brief into a prompt, natively on MLX (no cloud). Modes:
**expand** (brief → prompt, ~15s), **analyze** (describe a photo, ~40s), **renovate** (apply a brief
to a photo, ~11 min). Read the result in the display node, or wire `prompt` into any Sampler.""", (1120, 210), title="10 · VLM")
    img = w.add("LoadImage", col(0), {"image": "room.png"}, "Photo (for analyze/renovate)", color=(BLUE, "#233"), size=[340, 300])
    vlm = w.add("MfluxVLM", col(1), {"mode": "expand (build a prompt from my brief, no photo)",
              "brief": "warm modern living room, three-seat sofa, coffee table, dark walnut floor, framed art, plants",
              "quantize": "8"}, "VLM · brief → prompt", color=(ORANGE, "#332"), size=[340, 300])
    disp = w.add("Display Any (rgthree)", col(2), {}, "Prompt out", color=(ORANGE, "#332"), size=[340, 220])
    w.link(img, 0, vlm, "image"); w.link(vlm, 0, disp, "source")
    save(w, "10_vlm_prompt.json", "FIBO-vlm")


def ex_upscale():
    w = WF()
    w.note((20, 20), """# 11 · Upscale (SeedVR2)

One-step MLX upscaler. `resolution` takes `2x`, `1.5x`, or a short-edge target in pixels (e.g. `1080`).
It is its own model, loaded separately. Point it at any image.""", (1040, 170), title="11 · Upscale")
    img = w.add("LoadImage", col(0), {"image": "room.png"}, "Image", color=(BLUE, "#233"), size=[340, 300])
    up = w.add("MfluxUpscale", col(1), {"model": "seedvr2-3b", "resolution": "2x"}, "Upscale (SeedVR2)", color=(GREEN, "#232"), size=[340, 230])
    sv = w.add("SaveImage", col(2), {"filename_prefix": "upscaled"}, "Save", color=(GREY, "#222"), size=[340, 320])
    w.link(img, 0, up, "image"); w.link(up, 0, sv, "images")
    save(w, "11_upscale.json", "seedvr2-3b")


def ex_qwen_edit():
    w = WF()
    w.note((20, 20), """# 12 · Instruction edit with Qwen-Image-Edit

A second edit model alongside FLUX.2-klein-edit, with a different look. Same idea: give it your photo
and an instruction ("replace the floor with dark walnut, add a grey sofa, keep the room and the windows").

It takes a LIST of reference images, so you can chain **mflux Image** nodes through `image_in` to feed
several references at once. It also accepts a `negative_prompt`, which klein-edit does not.""", (1120, 210), title="12 · Qwen edit")
    img = w.add("LoadImage", col(0), {"image": "room.png"}, "Your room", color=(BLUE, "#233"), size=[340, 300])
    ld = w.add("MfluxModelLoader", col(1), {"model": "qwen-image-edit", "quantize": "8"}, "Loader · qwen-image-edit", color=(GREEN, "#232"), size=[340, 260])
    mi = w.add("MfluxImage", col(2), {}, "Photo (chain image_in for more)", color=(GREEN, "#232"), size=[340, 130])
    s = w.add("MfluxModelSampler", col(3), {"prompt": ("Replace the floor with dark walnut wood and add a grey three-seat sofa. "
              "Keep the room, the windows, the TV and the viewpoint the same."),
              "seed": 3, "width": 1024, "height": 768, "mask_mode": "off"}, "Sampler · your instruction", color=(GREEN, "#232"), size=[340, 440])
    pv = w.add("PreviewImage", col(4), {}, "Result", color=(GREY, "#222"), size=[340, 300])
    w.link(img, 0, mi, "image"); w.link(ld, 0, s, "model"); w.link(mi, 0, s, "mflux_image"); w.link(s, 0, pv, "images")
    save(w, "12_qwen_edit.json", "qwen-image-edit")


print("Building example library ->", OUTDIR)
for fn in (ex_txt2img, ex_img2img, ex_lora, ex_edit_restyle, ex_replace_object, ex_region_mask,
           ex_controlnet_depth, ex_multi_controlnet, ex_redux, ex_vlm, ex_upscale, ex_qwen_edit):
    fn()
print("done.")
