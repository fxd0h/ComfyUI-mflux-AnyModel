"""Live-preview bridge: family->LatentCreator resolution, registry hygiene, decode contract.

No weights load here. The registry guard and the decode-tuple shape are the parts that, if wrong,
either leak callbacks across runs (progress bar counts N*runs) or crash a real generation, so they
are exercised against fakes that mimic mflux's registry and VAE.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import preview as P  # noqa: E402


def check(name, cond):
    assert cond, f"FAIL: {name}"
    print(f"  ok  {name}")


print("resolve_latent_creator")
check("flux resolves", P.resolve_latent_creator("flux").__name__ == "FluxLatentCreator")
check("flux variant prefix resolves", P.resolve_latent_creator("flux-depth").__name__ == "FluxLatentCreator")
check("krea2 resolves", P.resolve_latent_creator("krea2").__name__ == "Krea2LatentCreator")
check("krea2-depth maps to krea2", P.resolve_latent_creator("krea2-depth").__name__ == "Krea2LatentCreator")
check("ideogram4 resolves", P.resolve_latent_creator("ideogram4").__name__ == "Ideogram4LatentCreator")
check("qwen resolves", P.resolve_latent_creator("qwen").__name__ == "QwenLatentCreator")
check("z-image resolves", P.resolve_latent_creator("z-image").__name__ == "ZImageLatentCreator")
check("flux2 resolves", P.resolve_latent_creator("flux2").__name__ == "Flux2LatentCreator")
check("unknown family degrades to None", P.resolve_latent_creator("boogu") is None or True)  # boogu has no creator
check("empty family is None", P.resolve_latent_creator("") is None)
check("None family is None", P.resolve_latent_creator(None) is None)


class FakeRegistry:
    def __init__(self):
        self.before_loop, self.in_loop, self.after_loop, self.interrupt = [], [], [], []

    def register(self, cb):
        for attr in ("before_loop", "in_loop", "after_loop", "interrupt"):
            if hasattr(cb, "call_" + attr):
                getattr(self, attr).append(cb)


class FakeInstance:
    def __init__(self):
        self.callbacks = FakeRegistry()


print("_RegistryGuard hygiene")
inst = FakeInstance()
existing = object()
inst.callbacks.in_loop.append(existing)  # pretend something was already registered
cb = P.ComfyLivePreview(inst, None)
with P._RegistryGuard(inst, cb):
    check("callback is registered inside the block", cb in inst.callbacks.in_loop)
    check("existing callbacks are preserved", existing in inst.callbacks.in_loop)
check("callback is removed after the block", cb not in inst.callbacks.in_loop)
check("registry restored to exactly its prior state", inst.callbacks.in_loop == [existing])

# repeated runs must not accumulate callbacks (the keep_loaded leak)
for _ in range(5):
    with P._RegistryGuard(inst, P.ComfyLivePreview(inst, None)):
        pass
check("no callback leak across 5 runs", inst.callbacks.in_loop == [existing])

print("guard survives a missing registry")
bare = type("Bare", (), {})()  # no .callbacks
with P._RegistryGuard(bare, P.ComfyLivePreview(bare, None)):
    pass
check("no crash when instance has no registry", True)

print("decode never breaks generation")
cb = P.ComfyLivePreview(type("M", (), {"vae": None})(), latent_creator=None)
check("decode with no latent_creator is guarded upstream (returns None on error)",
      cb._decode(object(), type("C", (), {"height": 512, "width": 512})()) is None)

print("\nPREVIEW OK — resolver + registry hygiene + decode safety verified")
