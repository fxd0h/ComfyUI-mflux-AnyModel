"""Upstream-mflux smoke test: the node must LOAD on stock (non-fork) mflux.

Registry users install this node with plain `mflux` from PyPI, which lacks the fork-only
families (Krea 2, Boogu, FLUX.2-Klein edit, ...). A hard import of those used to crash the
whole node at ComfyUI load. This guards that regression: the modules import, base models
still resolve, and absent fork models are simply skipped (not None-routed, not crashing).
Passes on BOTH upstream and the fork.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import mflux_dispatch as D  # must not raise even when fork models are absent
import capability as C      # noqa: F401
import nodes                # noqa: F401  — the ComfyUI entry surface

# base families that exist in ANY mflux must still resolve to a real class
cls, fam = D.pick_model_class("schnell")
assert cls is not None and fam == "flux", f"schnell must resolve to Flux1/flux, got {cls}/{fam}"

# the dispatch tables only ever contain classes the installed mflux actually provides
for k, (kcls, _fam) in D.ALIAS_DISPATCH.items():
    assert kcls is not None, f"ALIAS_DISPATCH has a None class for {k} (resilience broken)"
for alias in D.DROPDOWN_EXTRA:
    c, _ = D.pick_model_class(alias)
    assert c is not None, f"DROPDOWN_EXTRA offers {alias} but it resolves to None"

# the model dropdown builds without error and lists at least the base models
choices = [nodes.strip_mark(m) for m in nodes.model_choices()]
assert any("schnell" in m for m in choices), "dropdown must contain the base schnell model"

n_fork = sum(getattr(D, c) is not None for c in ("Krea2", "Krea2Depth", "BooguImage"))
print(f"SMOKE OK — node loads. fork-only classes present: {n_fork}/3, dropdown models: {len(choices)}")
