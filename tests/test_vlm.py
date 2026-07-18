"""MfluxVLM: JSON salvage + prose rendering. Pure, no weights loaded.

FIBO-vlm always answers in its scene JSON, and a real run with max_tokens=700 truncated mid-string
(`Unterminated string starting at: line 1 column 2798`). Both facts are load-bearing for the node,
so they are pinned here: the truncated shape below is the real one that broke, trimmed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from vlm import MODES, _loads_tolerant, _to_prose  # noqa: E402

FULL = {
    "short_description": "An inviting minimalist living room, low angle, warm directional sunlight.",
    "objects": [
        {"description": "A modern armchair with a circular wooden frame.", "location": "center foreground"},
        {"description": "A light grey two-seater sofa.", "location": "midground, left of center"},
        {"description": "no description here", "location": "x"},
    ],
    "background_setting": "A large window dominates the background.",
    "lighting": {"conditions": "bright daylight", "direction": "side-lit from left", "shadows": "long, soft shadows"},
    "aesthetics": {"composition": "rule of thirds", "color_palette": "warm neutrals"},
    "style_medium": "photograph",
}

# the real failure: valid prefix, then a string cut mid-token when the budget ran out
TRUNCATED = (
    '{"short_description":"Interior de un elegante salon con un sillon de 3 cuerpos y un TV.",'
    '"objects":[{"description":"Un sillon de 3 cuerpos de tejido suave beige.","location":"center right"}],'
    '"background_setting":"Grandes ventanas con luz natu'
)


def check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"  ok  {name}")


print("_loads_tolerant")
check("clean JSON parses", _loads_tolerant('{"a": 1}') == {"a": 1})
d = _loads_tolerant(TRUNCATED)
check("truncated JSON salvages instead of dying", isinstance(d, dict))
check("salvage keeps the leading summary", "elegante salon" in d["short_description"])
check("salvage keeps the complete object", len(d.get("objects", [])) == 1)
check("salvage drops the severed field", "background_setting" not in d)
check("empty is None", _loads_tolerant("") is None)
check("garbage is None", _loads_tolerant("not json at all") is None)
check("unsalvageable head is None", _loads_tolerant('{"short_desc') is None)

print("_to_prose")
p = _to_prose(FULL)
check("leads with the summary", p.startswith("An inviting minimalist living room"))
check("carries the setting", "large window" in p)
check("pairs object with its location", "circular wooden frame, center foreground" in p)
check("flattens the lighting dict", "bright daylight. side-lit from left. long, soft shadows" in p)
check("flattens aesthetics", "rule of thirds" in p and "warm neutrals" in p)
check("keeps style_medium", "photograph" in p)
check("no raw JSON leaks into the prompt", "{" not in p and '"' not in p)
check("ends as a sentence", p.endswith("."))
check("honors max_objects", "two-seater" not in _to_prose(FULL, max_objects=1))
check("non-dict is empty", _to_prose(None) == "" and _to_prose("x") == "")
check("empty dict is empty", _to_prose({}) == "")
check("salvaged JSON still renders", "sillon de 3 cuerpos" in _to_prose(d))

print("_to_prose resilience")
check("objects may be missing", _to_prose({"short_description": "a room"}) == "a room.")
check("survives junk in objects", "a room" in _to_prose({"short_description": "a room", "objects": ["str", None, {}]}))
check("lighting as plain string", "golden hour" in _to_prose({"short_description": "r", "lighting": "golden hour"}))
check("dedupes repeats", _to_prose({"short_description": "a room", "background_setting": "A ROOM"}) == "a room.")

print("node surface")
check("three modes", len(MODES) == 3)
check("default mode is the renovate path", MODES[2].startswith("renovate"))
check("max_tokens default clears the truncation cliff",
      MfluxVLM_max := __import__("vlm").MfluxVLM.INPUT_TYPES()["optional"]["max_tokens"][1]["default"] >= 1500)

print("\nVLM OK — salvage + prose verified")
