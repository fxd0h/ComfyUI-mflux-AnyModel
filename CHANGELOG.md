# Changelog

All notable changes to this project are documented here. The format is loosely
based on Keep a Changelog.

## [0.1.1]

### Added
- LoRA feeder (`mflux LoRA`) now picks from your ComfyUI loras folder via a
  dropdown instead of a typed path; chain several with `lora_in`.
- Model dropdown marks models not present in the local HuggingFace cache and lists
  downloaded ones first, so you can see which would trigger a download.
- `mflux Upscale` resolution accepts a scale factor (`2x`, `1.5x`) or a shortest-edge
  pixel target.

### Fixed
- Ideogram-4 captions: the sampler extracts the JSON object from an LLM's output
  (stripping code fences or preamble) so mflux uses the structured caption instead of
  falling back to plain text, which otherwise triggers the model's safety-filter
  false positives.
- A LoRA that matches no layers of the selected model (wrong architecture) no longer
  hard-crashes: the model loads without it and the skip is reported in `info`.
- Selecting a gated or un-downloaded model raises a clear message instead of a raw
  HuggingFace 403 traceback.
- Download detection ignores partial/interrupted downloads (repos with `.incomplete`
  blobs are treated as not downloaded).

## [0.1.0]

### Added
- Initial release. Five nodes for running any mflux/MLX model in ComfyUI on Apple
  Silicon:
  - `mflux Model Loader` — resolve any model (builtin alias, HuggingFace repo, or
    local path) with quantization and an optional LoRA chain into a typed
    `MFLUX_MODEL` handle.
  - `mflux Sampler` — generate from the handle; reads a per-model capability profile
    (built by introspecting `generate_image`) and forwards only the parameters that
    model accepts, closing the silent steps/guidance degradation on preset-driven
    models such as Ideogram-4 and never feeding an image to a model that ignores it.
  - `mflux LoRA` — chainable LoRA feeder.
  - `mflux Image` — typed image feeder (primary image, optional mask, optional
    depth/control map).
  - `mflux Upscale` — SeedVR2 one-step upscaler.
- Variant dispatch so flux edit/fill/depth/redux/controlnet, qwen-edit and fibo-edit
  aliases resolve to their real class instead of degrading to plain text-to-image.
- Self-tests that introspect the installed mflux package (run on Apple Silicon CI).
