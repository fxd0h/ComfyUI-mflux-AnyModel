# Changelog

All notable changes to this project are documented here. The format is loosely
based on Keep a Changelog.

## [0.4.0]

### Added
- **Redux style reference (Mood Board)** via `dev-redux`, with **per-reference blend strength**.
  Chain an `mflux Image` per reference photo and each node's `strength` now weights that reference
  independently (`redux_image_strengths` receives the full aligned list instead of a single value),
  so a multi-image blend actually honors "70% this palette, 30% that". `interior_redux_moodboard.json`
  shows a two-reference blend. Redux generates a new image in the referenced style rather than
  restyling one specific room, so it is for direction; use `krea-2-depth` to keep an exact geometry.

### Fixed
- Redux multi-reference blends previously applied the feeder's strength only to the first reference
  (the injected `redux_image_strengths` was a one-element list regardless of image count). Each
  reference now carries its own strength through the `mflux Image` chain.

## [0.3.0]

### Added
- **Krea 2 depth ControlNet** (`krea-2-depth`): pick it in the loader, point `controlnet_path`
  at the depth-control checkpoint, and feed a room photo on `mflux Image`. DepthPro derives the
  depth automatically and the render keeps the room's layout while restyling it. A precomputed
  depth can be supplied instead on `map_image`. This is the geometry-lock path for interior
  renovation, where preserving room volume while changing materials/furniture is the whole point.
- **Loader `controlnet_path` + `controlnet_strength`** inputs. `controlnet_path` is required for
  `krea-2-depth` (validated early with a clear message) and also feeds a custom `flux-controlnet`.
  `controlnet_strength` scales the Krea 2 depth-control deltas at load time.
- **Mask-preserve composite** on the sampler for edit/img2img models that do not inpaint natively
  (`flux2-edit`, `qwen-edit`, `krea2`, base img2img). Connect a mask on `mflux Image` and choose
  `mask_mode`: `preserve` keeps the painted (white) region pixel-identical to the original (a
  hard-lock for windows and doors during a restyle), while `inpaint` lets only the painted region
  take the edit. `mask_feather` softens the seam. `flux-fill` still uses its mask natively, so the
  composite does not apply there.
- **Interior-design example workflows** (`example_workflows/`): `interior_krea2_depth.json`
  (depth-locked renovation) and `interior_flux2_edit_mask_preserve.json` (edit with locked openings).

## [0.2.0]

### Added
- New models, catching the dispatch up to mflux 0.18.23: **Krea 2** (Turbo and Raw),
  **Boogu-Image-0.1-Turbo**, and **FLUX.2-klein edit** (`flux2-klein-edit`) — the fast,
  structure-faithful edit model. All selectable from the dropdown.
- **Multi-image edit**: `mflux Image` is now chainable via `image_in`, so you can stack
  several reference images for the edit models that take an image list (`qwen-edit`,
  `flux2-edit`). The first image in the chain is the primary/viewpoint reference.

### Fixed
- Full instruction-edit models (`flux2-edit`, `qwen-edit`) no longer have `image_strength`
  forced on them. Their `image_strength` defaults to `None` (a full edit); forcing the
  feeder's strength turned the edit into a degraded partial img2img that reinterpreted the
  whole scene. Strength is now applied only to real img2img (`image_path`) and to redux.
- `qwen-image-layered` (image-to-RGBA-layers, no standard sampler API) is excluded from the
  sampler dropdown so selecting it can't crash the sampler.

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
