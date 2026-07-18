"""Build the professional interior-design workflow for ComfyUI, generated from the LIVE
/object_info so widget order, defaults and combo values are exactly what the server expects.

Layout is column-based: every node is stacked under the previous one in its column via a
per-column Y cursor, so nodes can never overlap. Groups are computed from their members' extents.
All in-canvas text is English.
"""
import json, os, urllib.request

with urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=30) as _r:
    OI = json.load(_r)
LINKY = {"IMAGE", "MASK", "MFLUX_MODEL", "MFLUX_IMAGE", "MFLUX_LORA", "LATENT", "CONDITIONING", "STRING", "*"}


def spec(node):
    info = OI[node]["input"]
    sockets, widgets = [], []
    for sec in ("required", "optional"):
        for k, v in (info.get(sec) or {}).items():
            t = v[0] if isinstance(v, list) else v
            opts = v[1] if isinstance(v, list) and len(v) > 1 and isinstance(v[1], dict) else {}
            if isinstance(t, str) and t in LINKY and t != "STRING":
                sockets.append((k, t))
            else:
                d = opts.get("default", (t[0] if isinstance(t, list) and t else ""))
                widgets.append((k, d))
                if opts.get("control_after_generate"):
                    widgets.append(("__cag", "randomize"))
    return sockets, widgets


class WF:
    def __init__(self):
        self.nodes, self.links, self.groups, self.nid, self.lid = [], [], [], 0, 0

    def add(self, type_, pos, ov=None, title=None, promote=(), color=None, size=None, mode=0):
        self.nid += 1
        sockets, widgets = spec(type_)
        vals = [(ov or {}).get(k, d) for k, d in widgets]
        ins = [{"name": k, "type": t, "link": None} for k, t in sockets]
        for wname in promote:
            ins.append({"name": wname, "type": "STRING", "link": None, "widget": {"name": wname}})
        outs = [{"name": n, "type": (OI[type_]["output"][i] if i < len(OI[type_]["output"]) else n),
                 "links": [], "slot_index": i}
                for i, n in enumerate(OI[type_]["output_name"] or OI[type_]["output"])]
        n = {"id": self.nid, "type": type_, "pos": [float(pos[0]), float(pos[1])],
             "size": size or [340, max(58, 34 + 26 * len(vals))],
             "flags": {}, "order": self.nid - 1, "mode": mode, "inputs": ins, "outputs": outs,
             "properties": {"Node name for S&R": type_}, "widgets_values": vals}
        if title:
            n["title"] = title
        if color:
            n["color"], n["bgcolor"] = color
        self.nodes.append(n)
        return self.nid

    def note(self, pos, text, size, title="Guide", color=("#432", "#653"), mode=0):
        self.nid += 1
        self.nodes.append({"id": self.nid, "type": "MarkdownNote", "pos": [float(pos[0]), float(pos[1])],
                           "size": [float(size[0]), float(size[1])], "flags": {}, "order": self.nid - 1,
                           "mode": mode, "inputs": [], "outputs": [], "title": title, "properties": {},
                           "widgets_values": [text], "color": color[0], "bgcolor": color[1]})
        return self.nid

    def link(self, src, slot, dst, dname):
        self.lid += 1
        s = next(n for n in self.nodes if n["id"] == src)
        d = next(n for n in self.nodes if n["id"] == dst)
        t = s["outputs"][slot]["type"]
        s["outputs"][slot]["links"].append(self.lid)
        di = next(i for i, x in enumerate(d["inputs"]) if x["name"] == dname)
        d["inputs"][di]["link"] = self.lid
        self.links.append([self.lid, src, slot, dst, di, t])

    def group_from(self, title, ids, color, pad=22):
        ns = [n for n in self.nodes if n["id"] in ids]
        x0 = min(n["pos"][0] for n in ns) - pad
        y0 = min(n["pos"][1] for n in ns) - pad - 34   # room for the group title bar
        x1 = max(n["pos"][0] + n["size"][0] for n in ns) + pad
        y1 = max(n["pos"][1] + n["size"][1] for n in ns) + pad
        self.groups.append({"title": title, "bounding": [x0, y0, x1 - x0, y1 - y0],
                            "color": color, "font_size": 24, "flags": {}})

    def dump(self):
        return {"last_node_id": self.nid, "last_link_id": self.lid, "nodes": self.nodes,
                "links": self.links, "groups": self.groups, "config": {}, "extra": {}, "version": 0.4}


# --------------------------------------------------------------------------- #
# Column layout: each column has an independent Y cursor; nodes stack downward.
# --------------------------------------------------------------------------- #
COLW, STRIDE, GAPY = 400, 460, 30
YCUR = {}


def place(col, h, gap=GAPY):
    y = YCUR.get(col, 0.0)
    YCUR[col] = y + h + gap
    return (col * STRIDE, y)


w = WF()
BLUE, GREEN, PURPLE, RED, GREY, ORANGE = "#3f789e", "#37734a", "#7b3f9e", "#9e3f3f", "#555", "#9e6b3f"

# ============================ TOP BANNER (read me) ============================
w.note((0, -840), """# 🏠 Interior Design · MLX / Apple Silicon

Everything runs **locally** (mflux + MLX). No cloud.

## The 3 paths — turn ONE on at a time with the **Bypasser** (top-left)

| Path | What for | Model |
|---|---|---|
| **B · Restyle** ⭐ | Keeps YOUR room AND adds/replaces furniture, floor, materials. The best path. | FLUX.2-klein-edit |
| **A · Strict perspective** | Locks the 3D viewpoint but EMPTIES the room (keeps no furniture, adds none). Niche. | FLUX.1 + depth ControlNet |
| **C · Mood board** | Explore styles from reference photos. Does NOT keep your room. | FLUX.1 Redux |

**Start with path B.** It is an edit model: give it your photo and an instruction, and it restyles while keeping the rest. No mask = the whole room; with a mask = one region only. Fast (~10-40s) and needs no VLM. Path A only locks a strict perspective and leaves the room empty.

## Workflow
1. Load your photo into **① Room photo**.
2. In **path B**, write the changes as commands (in English).
3. Generate, and watch the **live preview** as it runs.
4. Compare in **⑦ Before / After** and enlarge with **⑦ Upscale**.

## Several seeds at once
Set **Batch count = N** on the **Run** button and leave the `seed` on **increment**. Each result lands in the gallery and the live preview shows each one as it renders.

## Rules that pay off
- Write changes as commands: "add...", "replace...", "remove...", "repaint...", "keep...".
- End with **"keep the room, the windows and the viewpoint"**: it anchors what you do NOT want touched.
- Path B can **replace objects**: "replace the TV with a fireplace", "replace the speakers with plants".
- If a stock-style watermark appears, the "no watermark, no text" phrase in path B's prompt kills it; if it still shows, change the seed.""",
       size=(940, 720), title="READ ME FIRST", color=("#2a363b", "#1e2a30"))

# ============================ COL 0 · INPUTS ============================
C = 0
byp = w.add("Fast Groups Bypasser [Eclipse]", place(C, 240), {}, "⚡ Turn paths on/off", color=("#8a4", "#252"), size=[COLW, 240])
room = w.add("LoadImage", place(C, 320), {"image": "room.png"}, "① Room photo", color=(BLUE, "#233"), size=[COLW, 320])
refa = w.add("LoadImage", place(C, 320), {"image": "room.png"}, "② Style reference A", color=(PURPLE, "#323"), size=[COLW, 320])
refb = w.add("LoadImage", place(C, 320), {"image": "room.png"}, "② Style reference B", color=(PURPLE, "#323"), size=[COLW, 320])
w.group_from("① INPUTS · your images", [room, refa, refb], BLUE)

# ============================ COL 1 · ANALYSIS ============================
C = 1
w.note(place(C, 300), """### ② Automatic analysis

**Depth (DepthPro, MLX)** → feeds path A.
Near = white. Do NOT invert: that is the
convention the depth models here expect.

**Auto-mask (ADE20K)** → optional for path B
(region edits). Pick a region: `floor`, `walls`,
`ceiling`, `windows`, `doors`, `openings`,
`surfaces`, `furniture`, or `custom` + `custom_classes`.

`dilate` covers edges · `feather` softens the seam ·
`invert` = everything EXCEPT that region.

The two previews show what was detected before
you spend a generation.""", size=(COLW, 300), title="Guide · Analysis")
depth = w.add("MfluxDepthMap", place(C, 130), {}, "② Depth map (DepthPro)", color=(GREEN, "#232"), size=[COLW, 130])
dprev = w.add("PreviewImage", place(C, 260), {}, "👁 Depth", color=(GREY, "#222"), size=[COLW, 260])
amask = w.add("MfluxAutoMask", place(C, 320), {"region": "floor", "dilate": 6, "feather": 6}, "② Auto-mask (ADE20K)", color=(GREEN, "#232"), size=[COLW, 320])
mprev = w.add("PreviewImage", place(C, 260), {}, "👁 Mask", color=(GREY, "#222"), size=[COLW, 260])
w.group_from("② ANALYSIS · depth + regions", [depth, dprev, amask, mprev], GREEN)

w.link(room, 0, depth, "image")
w.link(depth, 0, dprev, "images")
w.link(room, 0, amask, "image")
w.link(amask, 1, mprev, "images")   # slot 1 = mask_image (IMAGE)

# ============================ COL 2 · VLM (optional, off) ============================
C = 2
w.note(place(C, 400), """### ③ VLM · OPTIONAL (turned off)

You don't need it for path B: there you type the
instruction directly. This group is a helper to
**draft or enrich** a prompt if you'd rather not
write it by hand.

**It is muted** (grey). To use it: right-click both
nodes → **Mode → Always**, run, read the `prompt`
in **Prompt final**, and copy it into the `prompt`
of the path you use.

**FIBO-vlm**, a native-MLX Qwen3-VL (local, no cloud). Modes:
- **expand** → builds a prompt from your brief, no photo (~15s).
- **analyze** → describes your photo as it is (~40s).
- **renovate** → looks at your photo and applies the
  brief (~11 min; for real restyling, path B is
  faster and more direct).""", size=(COLW, 400), title="Guide · VLM (optional)")
vlm = w.add("MfluxVLM", place(C, 300), {
    "mode": "renovate (apply my changes to the real photo)",
    "brief": ("replace the floor with dark walnut, warm modern palette, add a three-seat sofa, "
              "a coffee table and framed art, keep the windows, the doors and the camera viewpoint"),
    "quantize": "8", "temperature": 0.2, "max_tokens": 4096,
}, "③ VLM · brief → prompt (OPTIONAL, off)", color=(ORANGE, "#332"), size=[COLW, 300], mode=2)
pshow = w.add("Display Any (rgthree)", place(C, 180), {}, "③ Prompt final (read it)", color=(ORANGE, "#332"), size=[COLW, 180], mode=2)
w.group_from("③ VLM (OPTIONAL) · enable the group to use it", [vlm, pshow], ORANGE)

w.link(room, 0, vlm, "image")
w.link(vlm, 0, pshow, "source")   # slot 0 = prompt (prose)

# ============================ COL 3 · PATH B (MAIN) ============================
C = 3
w.note(place(C, 430), """### Ⓑ Restyle · THE BEST PATH ⭐

**Keeps YOUR room and applies your changes:** adds
furniture, swaps the floor, materials, light, and
leaves the rest as is. Edit model (FLUX.2-klein-edit):
takes your photo + an instruction. Fast (~10-40s)
and needs NO VLM.

**Write the `prompt` as COMMANDS, in English:**
> add a three-seat sofa and a coffee table,
> replace the floor with dark walnut,
> keep the room, the windows and the TV

`mask_mode`:
- **off** → restyles the WHOLE room (the usual here).
- **inpaint** → changes ONLY region ② (the auto-mask
  is wired): "restyle the floor only".
- **preserve** → locks what is painted (windows/doors)
  and changes the rest.

The *"no watermark, no text"* phrase in the prompt
kills the stock watermark; if it still shows, change
the seed.""", size=(COLW, 430), title="Guide · Path B (main)")
ldB = w.add("MfluxModelLoader", place(C, 260), {"model": "flux2-klein-edit", "quantize": "8"},
            "Ⓑ Loader · FLUX.2-klein-edit", color=(GREEN, "#232"), size=[COLW, 260])
imgB = w.add("MfluxImage", place(C, 120), {}, "Ⓑ Photo (+ optional mask)", color=(GREEN, "#232"), size=[COLW, 120])
sampB = w.add("MfluxModelSampler", place(C, 440), {
    "prompt": ("add a large three-seat grey fabric sofa and a low wooden coffee table in the center, "
               "replace the floor with dark walnut wood, warm modern styling, "
               "keep the room, the windows, the TV and the camera viewpoint. "
               "Clean professional interior photograph, no watermark, no text, no logo."),
    "seed": 3, "width": 1024, "height": 768, "mask_mode": "off", "mask_feather": 6,
}, "Ⓑ Sampler · write your changes here", color=(GREEN, "#232"), size=[COLW, 440])
prevB = w.add("PreviewImage", place(C, 300), {}, "👁 Ⓑ Result", color=(GREY, "#222"), size=[COLW, 300])
w.group_from("Ⓑ PATH B · RESTYLE (keeps your room) ⭐", [ldB, imgB, sampB, prevB], GREEN)

w.link(room, 0, imgB, "image")
w.link(amask, 1, imgB, "mask")
w.link(ldB, 0, sampB, "model")
w.link(imgB, 0, sampB, "mflux_image")
w.link(sampB, 0, prevB, "images")

# ============================ COL 4 · PATH A (depth) ============================
C = 4
w.note(place(C, 520), """### Ⓐ Strict perspective · depth ControlNet

**Secondary. Use path B to actually restyle.**

This locks your photo's 3D viewpoint, but because
depth is only geometry (it doesn't know what
furniture is there), it **EMPTIES the room**: it
keeps no TV/sofa and adds no new furniture (depth
resists it). Good for restyling a near-empty room or
fixing an exact perspective, not for furnishing.

**depth** (Union-Pro-2.0) ← the ② map · strength 0.7
locks the 3D geometry.

## TWO things you MUST set right
1. **width/height = YOUR photo's aspect.**
   Landscape → 1024 x 768. Portrait → 768 x 1024.
   Square on a landscape photo squashes the geometry
   into a different room.
2. **steps 24.** FLUX.1-dev needs ~24; with 8 the
   ControlNet can't lock the structure.

*Optional, to also lock existing objects: add a 2nd
line in controlnet_path with InstantX Canny and a 2nd
mflux Image (your photo, strength ~0.3). It pins edges
but barely restyles and can add artifacts.*

Needs FLUX.1-dev (~31G) + Union-Pro-2.0 (~4G).""", size=(COLW, 520), title="Guide · Path A")
ldA = w.add("MfluxModelLoader", place(C, 260), {
    "model": "dev-controlnet-canny", "quantize": "8",
    "controlnet_path": "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0",
}, "Ⓐ Loader · FLUX.1 + depth ControlNet", color=(RED, "#322"), size=[COLW, 260])
imgA1 = w.add("MfluxImage", place(C, 120), {"strength": 0.7}, "Ⓐ ControlNet ← depth (0.7)", color=(RED, "#322"), size=[COLW, 120])
sampA = w.add("MfluxModelSampler", place(C, 440), {
    "prompt": "a warm modern living room, dark walnut floor, cream walls, framed art, soft daylight",
    "params_mode": "override", "seed": 11, "steps": 24, "guidance": 3.5, "width": 1024, "height": 768,
}, "Ⓐ Sampler · steps 24, match your photo aspect", color=(RED, "#322"), size=[COLW, 440])
prevA = w.add("PreviewImage", place(C, 300), {}, "👁 Ⓐ Result", color=(GREY, "#222"), size=[COLW, 300])
w.group_from("Ⓐ PATH A · strict perspective (depth)", [ldA, imgA1, sampA, prevA], RED)

w.link(depth, 0, imgA1, "image")
w.link(ldA, 0, sampA, "model")
w.link(imgA1, 0, sampA, "mflux_image")
w.link(sampA, 0, prevA, "images")

# ============================ COL 5 · PATH C (Redux) ============================
C = 5
w.note(place(C, 300), """### Ⓒ Mood board · Redux

Blends the **style** of your reference photos ② with
the prompt. Each reference carries its own weight:
1.0 dominates, 0.4 barely tints.

**IMPORTANT:** Redux generates a **new** image in that
style. It does **NOT keep your room**. Use it to
explore a look, not to restyle your living room. For
that, path B.

Needs FLUX.1-dev + Redux (~32G).""", size=(COLW, 300), title="Guide · Path C")
ldC = w.add("MfluxModelLoader", place(C, 260), {"model": "dev-redux", "quantize": "8"},
            "Ⓒ Loader · FLUX.1 Redux", color=(PURPLE, "#323"), size=[COLW, 260])
imgC1 = w.add("MfluxImage", place(C, 120), {"strength": 1.0}, "Ⓒ Ref A (1.0)", color=(PURPLE, "#323"), size=[COLW, 120])
imgC2 = w.add("MfluxImage", place(C, 120), {"strength": 0.5}, "Ⓒ Ref B (0.5)", color=(PURPLE, "#323"), size=[COLW, 120])
sampC = w.add("MfluxModelSampler", place(C, 440), {
    "prompt": "a bright modern living room interior in the referenced style",
    "seed": 42, "guidance": 2.5, "width": 1024, "height": 1024,
}, "Ⓒ Sampler", color=(PURPLE, "#323"), size=[COLW, 440])
prevC = w.add("PreviewImage", place(C, 300), {}, "👁 Ⓒ Result", color=(GREY, "#222"), size=[COLW, 300])
w.group_from("Ⓒ PATH C · mood board (Redux)", [ldC, imgC1, imgC2, sampC, prevC], PURPLE)

w.link(refa, 0, imgC1, "image")
w.link(imgC1, 0, imgC2, "image_in")
w.link(refb, 0, imgC2, "image")
w.link(ldC, 0, sampC, "model")
w.link(imgC2, 0, sampC, "mflux_image")
w.link(sampC, 0, prevC, "images")

# ============================ COL 6 · OUTPUT ============================
C = 6
w.note(place(C, 300), """### ⑦ Output

**Before / After**: drag the divider over the image
to compare. `image_b` comes from **path B** (the main
one); rewire it if you use another path.

**Upscale (SeedVR2)**: one step, MLX. `resolution`
takes `2x`, `1.5x` or a short-edge target in pixels
(e.g. `1080`). It is its own model, loaded separately.

**Save**: goes to `ComfyUI/output/`.""", size=(COLW, 300), title="Guide · Output")
cmp = w.add("Image Comparer (rgthree)", place(C, 380), {}, "⑦ Before / After", color=(GREY, "#222"), size=[COLW, 380])
ups = w.add("MfluxUpscale", place(C, 230), {"model": "seedvr2-3b", "resolution": "2x"}, "⑦ Upscale (SeedVR2)", color=(GREY, "#222"), size=[COLW, 230])
save = w.add("SaveImage", place(C, 320), {"filename_prefix": "interior"}, "⑦ Save", color=(GREY, "#222"), size=[COLW, 320])
w.group_from("⑦ OUTPUT · compare, enlarge, save", [cmp, ups, save], GREY)

w.link(room, 0, cmp, "image_a")
w.link(sampB, 0, cmp, "image_b")   # path B is the main one
w.link(sampB, 0, ups, "image")
w.link(ups, 0, save, "images")

out = os.path.expanduser("~/Documents/github/ComfyUI/user/default/workflows/INTERIOR_DESIGN_PRO.json")
with open(out, "w") as _f:
    json.dump(w.dump(), _f, indent=1, ensure_ascii=False)
print("SAVED:", out)
print(f"nodes={len(w.nodes)} links={len(w.links)} groups={len(w.groups)}")
