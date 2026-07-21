"""Audit which advertised models actually generate inside a running ComfyUI.

We ship a dropdown of ~34 models. Two of them (krea-2-depth, qwen-image-edit) were completely broken
in ComfyUI for a while and nobody noticed, because nothing exercised them. This script does: for every
alias in the live dropdown it submits one small generation and reports what happened.

Run it against a running ComfyUI (default http://127.0.0.1:8188), using the ComfyUI venv's python so
that the node's own modules import:

    ComfyUI/.venv/bin/python tools/audit_models.py                 # only models whose weights are already local
    python tools/audit_models.py --all           # include ones that would download (slow, huge)
    python tools/audit_models.py --only dev,qwen # a specific subset

Two things this script gets right, because both bit us when done by hand:

- It uses the EXACT dropdown values from /object_info, including the download marker. Stripping the
  marker yields a name the server rejects with a 400, which looks like a broken model and is not.
- "Has a cache directory" is not "has weights". A directory can hold zero safetensors or a half
  finished download, so the dropdown's own marker is trusted instead of a directory listing.
- A plain text-to-image graph is wrong for image-conditioned models. Sending one to `dev-redux` or a
  ControlNet variant fails for want of an input, not because the model is broken. Each model's real
  requirements are read from the node's capability profile, an image is supplied when one is needed,
  and models that need a checkpoint we cannot guess are reported as SKIPPED rather than FAIL.
"""
import argparse, json, os, sys, time, urllib.error, urllib.request, uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_SERVER = "http://127.0.0.1:8188"
NEEDS_DL_MARKS = ("⬇", "↓")  # the dropdown's "would download" hint


def api(server, path):
    with urllib.request.urlopen(f"{server}{path}", timeout=30) as r:
        return json.load(r)


def dropdown_models(server):
    """Exact combo values, unmodified. Do not strip the marker: the server validates against these."""
    info = api(server, "/object_info/MfluxModelLoader")
    node = next(iter(info.values()))
    field = node["input"].get("required", {}).get("model") or node["input"]["optional"]["model"]
    return list(field[0])


def looks_local(value):
    """A marked entry is one the node already believes would trigger a download."""
    return not any(m in value for m in NEEDS_DL_MARKS)


def requirements(model):
    """(needs_image, needs_controlnet_path) straight from the node's own capability profile.

    This imports the node, so the script must run with an interpreter that has mflux installed
    (the ComfyUI venv). Running it with a bare system python used to make every import fail, which
    a broad `except` then reported as "this model needs nothing" - so every image-conditioned model
    came back FAILED for want of an input it was never given. Fail loudly instead.
    """
    import capability as C
    import mflux_dispatch as D
    cls, family = D.pick_model_class(model, None)
    if cls is None:
        return False, False
    profile = C.build_profile(cls, family)
    roles = getattr(profile, "image_role_args", {}) or {}
    needs_image = any(v == "req" for v in roles.values()) or family in getattr(D, "NEEDS_ANY_IMAGE", set())
    needs_cn = bool(D.requires_controlnet_path(cls)) if hasattr(D, "requires_controlnet_path") else False
    return needs_image, needs_cn


def submit(server, graph):
    body = json.dumps({"prompt": graph, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(server + "/prompt", body, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["prompt_id"], None
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode())
            msgs = [x.get("message", "") for n in payload.get("node_errors", {}).values()
                    for x in n.get("errors", [])]
            return None, "; ".join(msgs) or payload.get("error", {}).get("message", str(e))
        except Exception:
            return None, f"HTTP {e.code}"


def run_one(server, model, timeout, image_name):
    needs_image, needs_cn = requirements(model)
    if needs_cn:
        return "SKIPPED", "needs a controlnet_path checkpoint this audit cannot guess", 0.0
    graph = {
        "1": {"class_type": "MfluxModelLoader",
              "inputs": {"model": model, "quantize": "8", "keep_loaded": False, "free_comfy_first": True}},
        "2": {"class_type": "MfluxModelSampler",
              "inputs": {"model": ["1", 0], "prompt": "a cozy scandinavian living room, soft daylight",
                         "params_mode": "auto", "seed": 3, "width": 512, "height": 512,
                         "live_preview": False}},
        "3": {"class_type": "PreviewImage", "inputs": {"images": ["2", 0]}},
    }
    if needs_image:
        if not image_name:
            return "SKIPPED", "image-conditioned, pass --image <name in ComfyUI/input>", 0.0
        graph["8"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        graph["9"] = {"class_type": "MfluxImage", "inputs": {"image": ["8", 0], "strength": 0.5}}
        graph["2"]["inputs"]["mflux_image"] = ["9", 0]
    started = time.time()
    prompt_id, rejected = submit(server, graph)
    if rejected:
        return "REJECTED", rejected, 0.0
    while time.time() - started < timeout:
        try:
            history = api(server, f"/history/{prompt_id}")
        except Exception as e:
            return "SERVER_GONE", str(e)[:80], time.time() - started
        if prompt_id in history:
            status = history[prompt_id]["status"]
            errors = [m for m in status.get("messages", []) if m[0] == "execution_error"]
            detail = ""
            if errors:
                try:
                    detail = str(errors[0][1].get("exception_message", ""))[:120]
                except Exception:
                    detail = str(errors[0])[:120]
            return status.get("status_str", "?"), detail, time.time() - started
        time.sleep(5)
    return "TIMEOUT", f">{timeout}s", time.time() - started


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default=DEFAULT_SERVER)
    p.add_argument("--all", action="store_true", help="also try models that would download")
    p.add_argument("--only", help="comma separated substrings to match against the dropdown")
    p.add_argument("--timeout", type=int, default=1800, help="seconds per model")
    p.add_argument("--image", help="filename in ComfyUI/input to feed image-conditioned models")
    args = p.parse_args()

    try:
        import mflux_dispatch  # noqa: F401
    except Exception as e:
        raise SystemExit(
            f"Cannot import the node's modules ({type(e).__name__}: {e}).\n"
            "Run this with an interpreter that has mflux installed, e.g.\n"
            "    ComfyUI/.venv/bin/python tools/audit_models.py\n"
            "Without it, image-conditioned models get a text-to-image graph and fail spuriously.")

    models = dropdown_models(args.server)
    targets = models if args.all else [m for m in models if looks_local(m)]
    if args.only:
        wanted = [s.strip().lower() for s in args.only.split(",") if s.strip()]
        # still honour the download filter: --only narrows, it does not opt you into huge downloads
        pool = models if args.all else [m for m in models if looks_local(m)]
        targets = [m for m in pool if any(w in m.lower() for w in wanted)]

    skipped = len(models) - len(targets)
    print(f"{len(models)} models in the dropdown, auditing {len(targets)}"
          + (f", skipping {skipped} that would download (use --all)" if skipped and not args.all else ""))
    print()

    results = []
    for model in targets:
        state, detail, secs = run_one(args.server, model, args.timeout, args.image)
        ok = state == "success"
        skipped = state == "SKIPPED"
        results.append((model, state, ok, skipped))
        tag = "OK  " if ok else ("skip" if skipped else "FAIL")
        print(f"{tag} {model:28} {state:12} {secs:5.0f}s {detail}", flush=True)

    failed = [m for m, _, ok, sk in results if not ok and not sk]
    ran = [r for r in results if not r[3]]
    skipped = [m for m, _, _, sk in results if sk]
    print()
    print(f"{len(ran) - len(failed)}/{len(ran)} generated" + (f", {len(skipped)} skipped" if skipped else "") + ".")
    if failed:
        print("Failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
