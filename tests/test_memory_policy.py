"""MemorySaver wiring: one-shot registration, no stacked preview, release drops the handle.

No weights. Run from this directory's parent with the ComfyUI interpreter or any
Python that can import the node modules:

    python tests/test_memory_policy.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import memory_policy as mp
import nodes


def check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"  ok  {name}")


class FakeRegistry:
    def __init__(self):
        self.before_loop, self.in_loop, self.after_loop, self.interrupt = [], [], [], []

    def register(self, cb):
        for attr in ("before_loop", "in_loop", "after_loop", "interrupt"):
            if hasattr(cb, "call_" + attr):
                getattr(self, attr).append(cb)


class FakeCallback:
    def __init__(self, name):
        self.name = name

    def call_in_loop(self, *args, **kwargs):
        return None


class FakeInstance:
    def __init__(self):
        self.callbacks = FakeRegistry()


print("saver_settings")
check("off is none", mp.saver_settings(False, False, 0) is None)
on = mp.saver_settings(True, False, 0)
check("one image drops both modules", on["keep_transformer"] is False and on["keep_text_encoder"] is False)
check("cache cap is 1 GB", on["cache_limit_bytes"] == 1000**3)
kept = mp.saver_settings(True, True, 0)
check("keep_loaded retains both modules", kept["keep_transformer"] is True and kept["keep_text_encoder"] is True)
multi = mp.saver_settings(True, False, 2)
check("two references keep the transformer only", multi["keep_transformer"] is True and multi["keep_text_encoder"] is False)

print("CallbackWindow")
inst = FakeInstance()
existing = FakeCallback("existing")
inst.callbacks.register(existing)
saver = FakeCallback("saver")
preview = FakeCallback("preview")
with mp.CallbackWindow(inst, [saver, preview]):
    names = [cb.name for cb in inst.callbacks.in_loop]
    check("saver and preview are both registered", names == ["existing", "saver", "preview"])
check("window restores the registry", [cb.name for cb in inst.callbacks.in_loop] == ["existing"])
for _ in range(5):
    with mp.CallbackWindow(inst, [FakeCallback("saver"), FakeCallback("preview")]):
        pass
check("five runs do not stack callbacks", len(inst.callbacks.in_loop) == 1)

print("release_model")
handle = type("H", (), {"instance": inst})()
cache = {"key": ("k",), "model": inst}
mp.release_model(handle, cache)
check("handle no longer owns the module", handle.instance is None)
check("loader cache no longer owns the module", cache["model"] is None and cache["key"] is None)

print("low_ram is on the loader signature and defaults on")
import inspect
params = inspect.signature(nodes.MfluxModelLoader.load).parameters
check("load takes low_ram", "low_ram" in params)
check("low_ram defaults on", params["low_ram"].default is True)
check("low_ram is the last parameter", list(params)[-1] == "low_ram")

print("keep_loaded off makes the loader run again on every queue")
import math
changed = nodes.MfluxModelLoader.IS_CHANGED
check("keep_loaded off is never 'unchanged' (NaN)", math.isnan(changed(model="schnell", keep_loaded=False)))
check("keep_loaded on keeps ComfyUI's cache", changed(model="schnell", keep_loaded=True) == changed(model="schnell", keep_loaded=True))

print("all memory-policy checks passed")
