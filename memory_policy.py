"""Turn on mflux's own MemorySaver for one generate call, then drop the model.

The mflux CLI always registers MemorySaver. Without it the MLX cache keeps every
freed step buffer, and a 4-bit pack that starts near 10 GB finishes near 50 GB.
`--low-ram` is that callback with a 1 GB cache cap: the text encoder is dropped
after encoding, and the transformer is dropped before the VAE decode when this
run does not need it again.

The callback is registered only for the generate call. A keep_loaded model would
otherwise stack one saver per run, because its CallbackRegistry survives.
`keep_loaded` keeps the text encoder and the transformer on the module so the
next prompt can reuse them. The cache cap still applies for that run.

MFLUX_BEFORE_LOAD and MFLUX_AFTER_FREE are optional shell commands. Unset means
the node does not wait. This is the whole external memory-lease hook.
"""
import gc
import os
import subprocess

# Decimal GB, matching mflux's CLI (`cache_limit_bytes or 1000**3`).
CACHE_LIMIT_BYTES = 1000**3


def saver_settings(low_ram, keep_loaded, n_images):
    """Arguments for MemorySaver, or None when low_ram is off.

    More than one reference image keeps the transformer through the decode, which
    is what the CLI does (`len(images) > 1`). The text encoder is only kept when
    the model itself stays loaded for another prompt.
    """
    if not low_ram:
        return None
    return {
        "keep_transformer": bool(keep_loaded) or int(n_images) > 1,
        "keep_text_encoder": bool(keep_loaded),
        "cache_limit_bytes": CACHE_LIMIT_BYTES,
    }


def make_saver(instance, low_ram, keep_loaded, n_images):
    settings = saver_settings(low_ram, keep_loaded, n_images)
    if settings is None:
        return None
    from mflux.callbacks.instances.memory_saver import MemorySaver
    return MemorySaver(model=instance, **settings)


class CallbackWindow:
    """Register callbacks for one generate call and restore the registry after.

    Same contract as preview._RegistryGuard: snapshot the four lists, register,
    restore on the way out, including when generate raises. A missing registry
    is a no-op so a fake or a partial model does not crash the node.
    """

    def __init__(self, instance, callbacks):
        self.instance = instance
        self.callbacks = [cb for cb in callbacks if cb is not None]
        self.registry = None
        self.saved = None

    def __enter__(self):
        reg = getattr(self.instance, "callbacks", None)
        self.registry = reg
        if reg is None:
            return self
        self.saved = (
            list(reg.before_loop), list(reg.in_loop),
            list(reg.after_loop), list(reg.interrupt),
        )
        for cb in self.callbacks:
            reg.register(cb)
        return self

    def __exit__(self, *exc):
        reg, saved = self.registry, self.saved
        if reg is not None and saved is not None:
            reg.before_loop, reg.in_loop, reg.after_loop, reg.interrupt = saved
        return False


def release_model(handle, cache):
    """Drop the MLX module the handle owns and clear MLX's cache.

    Clearing the loader cache dict is not enough: the handle returned to the
    graph still owns the module, and ComfyUI's /free does not see MLX allocations.
    """
    inst = getattr(handle, "instance", None)
    handle.instance = None
    if cache is not None and cache.get("model") is inst:
        cache["key"] = None
        cache["model"] = None
    del inst
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def run_hook(env_name, *, required):
    """Run the shell command in `env_name`. Empty or unset is success.

    `required` raises on a non-zero exit. The after-free hook is not required:
    the image is already decoded, and a resume failure must not throw it away.
    """
    cmd = os.environ.get(env_name, "").strip()
    if not cmd:
        return ""
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"{env_name} timed out after 180s"
        print(f"[mflux] {msg}")
        if required:
            raise RuntimeError(msg) from exc
        return msg
    tail = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()[-800:]
    if proc.returncode != 0:
        msg = f"{env_name} exited {proc.returncode}: {tail}"
        print(f"[mflux] {msg}")
        if required:
            raise RuntimeError(msg)
        return msg
    if tail:
        print(f"[mflux] {env_name}: {tail}")
    return tail
