# Changelog

All notable changes to this project are documented here. The format is loosely
based on Keep a Changelog.

## [0.8.1]

### Fixed
- **`control_type` was inserted in the middle of the sampler's inputs in 0.8.0, which shifts every
  widget in already-saved graphs.** ComfyUI stores `widgets_values` positionally, so a workflow saved
  before 0.8.0 would have read `mask_mode` into `control_type`, `mask_feather` into `mask_mode`, and
  so on down the line, with nothing raised to say so. The widget is now appended last, where a new
  entry leaves every existing position alone. 0.8.0 reached the registry with this in it; upgrade
  past it rather than pinning to it.

### Added
- Two example workflows for the families 0.8.0 wired: `14_zimage_controlnet.json` and
  `15_mage_flow.json`, each with the traps written into the note node (feed the original photo rather
  than a pre-computed edge map, how a `canny,depth` stack pairs with the MfluxImage chain, and which
  aliases are distilled).

## [0.8.0]

### Added
- **Mage Flow**, text-to-image and instruction edit (`mage-flow`, `mage-flow-turbo`, `mage-flow-base`,
  `mage-flow-edit`, `mage-flow-edit-turbo`). The edit variant takes `image_paths`, the same list role
  qwen-edit and flux2-edit already use, so it rides the existing MfluxImage feeder. Edit is matched
  before txt2img in the ladder so `mage-flow-edit-turbo` cannot land on the txt2img class and silently
  drop its references.
- **Z-Image Union ControlNet** (`z-image-controlnet`), the only Z-Image ControlNet in MLX, previously
  reachable only from the CLI. Its `generate_image` takes `controls: list[ControlSpec]` rather than an
  image path, so `controls` joins `IMAGE_ROLE_PARAMS` (without that the profile reads an unsatisfiable
  required argument and the sampler refuses the model outright), and `inject_image` builds one spec per
  image in the MfluxImage chain. A `control_type` widget names them: one value applies to every image,
  a comma-separated list pairs with the chain in order. Per-control strengths come from the chain.

### Fixed
- The sampler read `gen.image` unconditionally. `ZImageTurboControlnet` is annotated `-> Image.Image`
  and in fact returns a `GeneratedImage`, so the output is now read from `.image` when it is present
  rather than trusting either the annotation or the wrapper.

## [0.7.4]

### Fixed
- **Live preview on Ideogram 4 when running against stock mflux.** The preview decoder chose its path
  with `hasattr(vae, "decode_packed_latents")`, which does not discriminate: Ideogram 4 uses
  `Flux2VAE`, so it always has that method. What differs is what `unpack_latents` returns. FLUX.2
  hands back patchified latents (128 channels) that still need the BN denorm and unpatchify that
  `decode_packed_latents` performs, while Ideogram 4 applies its own shift/scale and unpatchifies
  inside `unpack_latents`, so its latents are already VAE-ready (32 channels). The path is now chosen
  by channel count. mflux-CV was never affected, because its `decode_packed_latents` falls through to
  `decode()` on a channel mismatch, but stock mflux from PyPI ships Ideogram 4 without that
  fall-through and this node runs there too. Same bug one layer down: filipstrand/mflux#444 (@plz12345)
  and our #468.


## [0.7.3]

### Changed
- **Ship only what runs.** Added `.comfyignore` so `tools/` (the workflow generators) and `tests/`
  stay out of the published package. They are dev-only, and the generators call
  `urllib.request.urlopen("http://127.0.0.1:8188/object_info")`, which has no business being in a
  distributed node. Package drops from 96 KB to 68 KB.


## [0.7.2]

### Fixed
- **`krea-2-depth` and `qwen-image-edit` now run inside ComfyUI.** They failed with
  `RuntimeError: There is no Stream(gpu, 0) in current thread` raised from `mx.eval`, and the same
  failure took several minutes to surface. Root cause: MLX >= 0.31 keeps a Metal command encoder per
  *thread* and pins an unevaluated array to the stream of the thread that recorded it
  (ml-explore/mlx#3529). `QwenVAE` declares its latent constants as
  `mx.array([...]).reshape(1, 16, 1, 1, 1)`, and the trailing `reshape` records a primitive instead
  of materializing, so the constants stay lazy. ComfyUI imports custom nodes on the main thread but
  runs every node on its single prompt_worker thread, so the first `mx.eval` downstream of a QwenVAE
  encode/decode died. Only these two families were affected because Krea 2 reuses the Qwen VAE.
  The dispatch module now materializes those constants on the importing thread. Verified end to end:
  both models render (70s and 76s) with no stream error and no worker crash.

### Added
- Two more example workflows now that those models work: `12_qwen_edit.json` and
  `13_krea2_depth.json` (the latter documents the local-checkpoint requirement, the
  `image` vs `map_image` trap, and Krea-2-Turbo's 8 steps / guidance 1.0).


## [0.7.1]

### Fixed
- **A failed generation no longer kills ComfyUI's prompt worker.** When a node raises, ComfyUI
  formats every input with `str()` to build the error report. `MfluxModelHandle` is a dataclass, so
  its default `__repr__` printed the `instance` field, walking the entire MLX module tree; some MLX
  layers have a broken `__repr__` (`Conv3d` reads a missing `groups`), so the recursion raised inside
  the error handler and took down the worker thread. One failed run left the queue dead until the
  server was restarted. The handle now has a shallow `__repr__` (alias + family).

### Known issues
- `krea-2-depth` and `qwen-image-edit` fail inside ComfyUI with an MLX stream error. **Fixed in 0.7.2.**

## [0.7.0]

### Added
- **`mflux VLM` node — write the prompt from a room photo, natively on MLX.** Runs FIBO-vlm
  (a Qwen3-VL) locally, so a workflow can read a photo and produce the prose prompt FLUX/Krea2
  want without leaving Apple Silicon and without a llama.cpp `mmproj` side-file. Three modes:
  `analyze` (describe the photo), `expand` (a brief into a prompt, no photo), and `renovate`
  (apply the brief to the real room). FIBO-vlm always answers in structured JSON, so the node
  flattens it to a paragraph and salvages a truncated reply. Outputs `(prompt, survey)`.
- **`mflux Depth Map` node (DepthPro).** Generates a depth map inside the workflow, so the
  depth-ControlNet path is self-contained and needs no external preprocessing.
- **Live preview and progress bar in the sampler.** The image now streams as it denoises and
  drives the ComfyUI progress bar; the Cancel button works mid-generation. New `live_preview`
  and `preview_stride` widgets. `mflux Upscale` seed gains the missing `control_after_generate`.
- **Openable example workflows** in `example_workflows/` (01–11 plus `INTERIOR_DESIGN_PRO.json`),
  one per capability, each with an in-canvas note. Generators in `tools/`.

## [0.6.0]

### Added
- **Multi-ControlNet stacking** (needs mflux-CV >= 0.18.25). Put **one checkpoint per line** in the
  loader's `controlnet_path` (a local path or an HF repo) to stack several controlnets on
  `flux-controlnet`, then chain one `mflux Image` per net, in the same order, each carrying that
  net's control image and `strength`. mflux sums their residuals, so nets with different block
  counts stack correctly. A single checkpoint behaves exactly as before.
  Example: `Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0` (depth) + `InstantX/FLUX.1-dev-Controlnet-Canny`
  holds a room's geometry while the prompt restyles its materials. See `interior_multi_controlnet.json`.
- Stacking is offered only where mflux supports it; asking for several checkpoints on a
  single-checkpoint model (`krea-2-depth`, which is input-concat) fails with a clear message, as does
  a path that is neither a local file nor an HF repo id.

## [0.5.0]

### Added
- **`mflux Auto Mask` node — regional interior editing without hand-painting.** Segments a room
  photo with an ADE20K semantic model (SegFormer, local on Apple Silicon MPS) and turns a named
  region (floor, walls, ceiling, windows, doors, openings, surfaces, furniture, or custom ADE20K
  labels) into a mask. Feed it to `mflux Image` -> mask and pick the sampler's `mask_mode`:
  `inpaint` restyles only that region ("restyle only the floor"), `preserve` locks it ("keep the
  windows"). Options for invert, dilate, feather, and the SegFormer size (b0/b2/b4/b5). This is the
  regional-control capability of a serious interior workflow, reached without a segmentation
  ControlNet (none exists for FLUX/Krea) by using segmentation as a mask preprocessor for the
  existing mask-preserve composite. Validated end to end: floor swapped to dark parquet, the rest
  of the room pixel-identical. See `interior_regional_restyle.json`.

## [0.4.2]

### Fixed
- **`dev-redux` now loads.** The FLUX.1-Redux-dev HuggingFace repo is adapter-only (SigLIP encoder
  plus the redux embedder, no base transformer/VAE/text-encoders), so selecting it failed with a
  missing-VAE error because the base was looked for in the adapter repo. The loader now defaults the
  redux base to `black-forest-labs/FLUX.1-dev` (an explicit `model_path` still wins). Validated
  end to end: a two-reference blend with distinct per-image strengths renders in the referenced style.

## [0.4.1]

### Fixed
- **The node now loads on stock (non-fork) mflux.** Model classes for the fork-only families
  (Krea 2, Boogu, FLUX.2-Klein edit, and any others a given mflux lacks) are imported defensively
  and are simply absent when the installed mflux does not provide them, instead of a hard import
  that crashed the whole node at ComfyUI load for every user on upstream mflux from PyPI. Dispatch,
  the dropdown, and the alias table skip the absent families; base models are unaffected.

### CI
- The self-test job now installs the mflux-CV fork (the supported config that ships every family
  the node drives). A second job installs stock mflux from PyPI and runs an import smoke test, so
  the "node loads on upstream" contract is verified on every push. Added `tests/smoke_import.py`.

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
