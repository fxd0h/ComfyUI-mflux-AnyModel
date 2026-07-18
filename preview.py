"""Live preview + progress bar for the mflux sampler, bridged into ComfyUI.

mflux denoises inside its own loop and only returns the final image. ComfyUI, meanwhile, has a
first-class channel for streaming previews and driving the node's progress bar: comfy.utils.ProgressBar.
This module joins the two. It registers a duck-typed callback on the model instance's CallbackRegistry
(mflux calls call_before_loop / call_in_loop each step), decodes the running latents to a small PIL
image with mflux's own VAE, and pushes it through ProgressBar.update_absolute(step, total, preview).

The decode is the same recipe mflux's StepwiseHandler uses (unpack_latents -> vae decode -> denormalize),
so the preview is faithful, not a cheap approximation. A per-step VAE decode is not free, so previews
are strided (default: every other step, always the last). Registering the callback also makes ComfyUI's
Cancel button work mid-generation: update_absolute raises the interrupt exception the executor watches.
"""
import importlib

# family (as the loader labels it) -> the LatentCreator that unpacks that family's latents.
# Matched by prefix, longest first, so "krea2-depth" resolves before a bare "krea2" rule would.
_LATENT_CREATORS = {
    "krea2": ("mflux.models.krea2.latent_creator.krea2_latent_creator", "Krea2LatentCreator"),
    "ideogram4": ("mflux.models.ideogram4.latent_creator.ideogram4_latent_creator", "Ideogram4LatentCreator"),
    "flux2": ("mflux.models.flux2.latent_creator.flux2_latent_creator", "Flux2LatentCreator"),
    "qwen": ("mflux.models.qwen.latent_creator.qwen_latent_creator", "QwenLatentCreator"),
    "z-image": ("mflux.models.z_image.latent_creator.z_image_latent_creator", "ZImageLatentCreator"),
    "fibo": ("mflux.models.fibo.latent_creator.fibo_latent_creator", "FiboLatentCreator"),
    "ernie": ("mflux.models.ernie_image.latent_creator.ernie_latent_creator", "ErnieLatentCreator"),
    "flux": ("mflux.models.flux.latent_creator.flux_latent_creator", "FluxLatentCreator"),
}


def resolve_latent_creator(family):
    """Return the LatentCreator class for a family, or None if it cannot be imported.

    None is a valid answer: the sampler degrades to a progress bar with no image preview rather
    than failing. Prefix match so every flux variant (canny/depth/fill/redux/kontext) and Krea 2's
    depth control map to their base family's creator.
    """
    fam = (family or "").lower()
    for prefix, (module, cls) in sorted(_LATENT_CREATORS.items(), key=lambda kv: -len(kv[0])):
        if fam.startswith(prefix):
            try:
                return getattr(importlib.import_module(module), cls)
            except Exception:
                return None
    return None


class ComfyLivePreview:
    """A duck-typed mflux callback that streams decoded steps to ComfyUI's ProgressBar."""

    def __init__(self, model, latent_creator, node_id=None, stride=2, max_size=512):
        self.model = model
        self.latent_creator = latent_creator
        self.node_id = node_id
        self.stride = max(1, int(stride))
        self.max_size = int(max_size)
        self.pbar = None
        self.total = None

    def call_before_loop(self, seed, prompt, latents, config, canny_image=None, depth_image=None):
        import comfy.utils
        self.total = int(config.num_inference_steps)
        self.pbar = comfy.utils.ProgressBar(self.total, node_id=self.node_id)

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        if self.pbar is None:
            import comfy.utils
            self.total = int(config.num_inference_steps)
            self.pbar = comfy.utils.ProgressBar(self.total, node_id=self.node_id)
        step = int(t) + 1
        preview = None
        if self.latent_creator is not None and (step % self.stride == 0 or step >= self.total):
            preview = self._decode(latents, config)
        # even with no image, this advances the bar and lets Cancel interrupt the loop
        self.pbar.update_absolute(step, self.total, preview)

    def _decode(self, latents, config):
        try:
            from mflux.utils.image_util import ImageUtil
            unpacked = self.latent_creator.unpack_latents(
                latents=latents, height=config.height, width=config.width
            )
            vae = self.model.vae
            decoded = (vae.decode_packed_latents(unpacked)
                       if hasattr(vae, "decode_packed_latents") else vae.decode(unpacked))
            pil = ImageUtil._numpy_to_pil(ImageUtil._to_numpy(ImageUtil._denormalize(decoded)))
            return ("JPEG", pil.convert("RGB"), self.max_size)
        except Exception as e:
            # a preview must never break generation; drop it and keep denoising
            print(f"[mflux] live preview decode skipped ({type(e).__name__}: {e})")
            return None


class _RegistryGuard:
    """Register a preview callback on a model's CallbackRegistry, then restore it exactly.

    The model instance is cached across runs (keep_loaded), and its registry persists with it.
    Appending without cleanup would stack a new preview callback every generation. This snapshots
    the registry's lists on enter and restores them on exit, so nothing leaks between runs.
    """

    def __init__(self, instance, callback):
        self.registry = getattr(instance, "callbacks", None)
        self.callback = callback
        self.saved = None

    def __enter__(self):
        reg = self.registry
        if reg is None:
            return self
        self.saved = (list(reg.before_loop), list(reg.in_loop),
                      list(reg.after_loop), list(reg.interrupt))
        reg.register(self.callback)
        return self

    def __exit__(self, *exc):
        reg, saved = self.registry, self.saved
        if reg is not None and saved is not None:
            reg.before_loop, reg.in_loop, reg.after_loop, reg.interrupt = (
                saved[0], saved[1], saved[2], saved[3])
        return False
