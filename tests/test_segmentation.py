"""Self-test for the interior AutoMask node's pure logic (no model weights loaded):
ADE20K label->id resolution and mask post-processing (invert / dilate / feather)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np

import segmentation as S

_checks = 0


def ok(cond, msg):
    global _checks
    _checks += 1
    if not cond:
        raise AssertionError("FAIL: " + msg)


# a stand-in ADE20K label map (real one is resolved from the model at run time)
ID2LABEL = {0: "wall", 3: "floor", 5: "ceiling", 8: "windowpane", 14: "door",
            23: "sofa", 19: "chair", 28: "rug", 58: "screen door", 64: "coffee table"}

# ---- 1. label -> id resolution ----
ok(S._resolve_ids(ID2LABEL, ["floor"]) == [3], "floor resolves to id 3")
ok(set(S._resolve_ids(ID2LABEL, ["window", "door"])) == {8, 14, 58}, "window+door match windowpane, door, screen door")
ok(S._resolve_ids(ID2LABEL, ["wall"]) == [0], "wall matches only wall")
ok(S._resolve_ids(ID2LABEL, ["nonexistent"]) == [], "unknown keyword resolves to nothing")
ok(S._resolve_ids(ID2LABEL, [""]) == [], "empty keyword is ignored (no match-all)")
ok(set(S._resolve_ids(ID2LABEL, ["chair", "sofa"])) == {19, 23}, "furniture keywords match")
print("1. ADE20K label->id resolution OK")

# ---- 2. region presets are well-formed and reference real interior classes ----
ok("floor" in S.REGION_PRESETS and "walls" in S.REGION_PRESETS, "core presets present")
ok(S._resolve_ids(ID2LABEL, S.REGION_PRESETS["openings (windows + doors)"]) != [], "openings preset resolves against ADE20K")
ok(S.REGION_PRESETS["custom (use custom_classes only)"] == [], "custom preset has no built-in keywords")
print("2. region presets OK")

# ---- 3. post-processing: invert / dilate / feather ----
m = np.zeros((20, 20), dtype=np.float32)
m[8:12, 8:12] = 1.0                       # a 4x4 white square

inv = S._postprocess(m, invert=True, dilate=0, feather=0)
ok(inv[0, 0] == 1.0 and inv[10, 10] == 0.0, "invert flips foreground/background")

dil = S._postprocess(m, invert=False, dilate=3, feather=0)
ok(dil.sum() > m.sum(), "dilate grows the masked area")
ok(dil[10, 10] == 1.0, "dilate keeps the original region set")

fea = S._postprocess(m, invert=False, dilate=0, feather=3)
ok(0.0 < fea[7, 10] < 1.0, "feather produces a soft edge just outside the square")
ok(fea.max() <= 1.0 and fea.min() >= 0.0, "feathered mask stays in [0,1]")

plain = S._postprocess(m, invert=False, dilate=0, feather=0)
ok(np.array_equal(plain, m), "no-op post-process returns the mask unchanged")
print("3. post-processing (invert/dilate/feather) OK")

# ---- 4. node contract ----
ok(S.MfluxAutoMask.RETURN_TYPES == ("MASK", "IMAGE", "STRING"), "node returns mask + image + info")
it = S.MfluxAutoMask.INPUT_TYPES()
ok("image" in it["required"] and "region" in it["required"], "required inputs present")
ok(set(S.SEG_MODELS) and all("segformer" in m for m in S.SEG_MODELS), "seg models are SegFormer ADE20K")
print("4. node contract OK")

print(f"\nALL SEGMENTATION SELF-TESTS PASSED ({_checks} checks)")
