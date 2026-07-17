# ComfyUI-mflux-AnyModel

Run any [mflux](https://github.com/filipstrand/mflux) model inside ComfyUI on Apple
Silicon (MLX/Metal). A single loader and sampler pair drives every mflux model
family — FLUX.1, FLUX.2 Klein (incl. **edit**), Qwen-Image (incl. edit), Krea 2,
Boogu, Z-Image, Ideogram 4, FIBO, ERNIE-Image — plus their image-conditioned
variants, through one consistent interface. Edit models that take several reference
images (`qwen-edit`, `flux2-edit`) are driven by chaining the `mflux Image` feeder.

The node is built around a **capability registry**: for the selected model it
inspects the real `generate_image` signature and forwards only the parameters that
model actually accepts. Parameters a model ignores are dropped with a note instead
of silently corrupting the result, and an input image is never handed to a model
that cannot use it.

## Why this exists

Most mflux models are driven the same way, but not all. Ideogram 4 is preset-driven:
its step count, per-step guidance schedule, and noise schedule are calibrated
together, so passing a loose `steps`/`guidance` value silently replaces the
calibrated schedule and degrades the image with no warning. Edit and image-to-image
variants each require a different image argument (`masked_image_path`,
`depth_image_path`, `redux_image_paths`, `controlnet_image_path`, `image_paths`).

This node encodes those differences once, so the common case stays trivial and the
edge cases fail loudly and clearly rather than producing a wrong image.

## Requirements

- Apple Silicon (M1–M5). MLX and Metal only; there is no CUDA path.
- ComfyUI.
- `mflux >= 0.18.0` (installed automatically as a dependency).

## Installation

ComfyUI Manager: search for **ComfyUI-mflux-AnyModel** and install.

Manual:

```
cd ComfyUI/custom_nodes
git clone https://github.com/fxd0h/ComfyUI-mflux-AnyModel
ComfyUI/.venv/bin/pip install "mflux>=0.18.0"
```

Restart ComfyUI.

## Nodes

| Node | Purpose |
|------|---------|
| **mflux Model Loader** | Resolve a model (builtin alias, HuggingFace repo, or local path) with quantization and an optional LoRA chain. Outputs a typed `MFLUX_MODEL` handle that carries the model and its capability profile. |
| **mflux Sampler** | Generate from the handle. Reads the capability profile and forwards only valid parameters. Outputs the image and an `info` string listing what was forwarded or dropped. |
| **mflux LoRA** | Chainable LoRA feeder (local file, HuggingFace repo, or `repo:filename.safetensors`). Stack several to compose. |
| **mflux Image** | Typed image feeder: a primary image, an optional mask (native inpaint for fill, or the mask-preserve composite for edit models), and an optional depth/control map (for depth and controlnet models). Chain via `image_in` for multi-image edits. |
| **mflux Auto Mask** | Segments a room photo (ADE20K SegFormer) and turns a named region (floor / walls / ceiling / windows / doors / furniture / custom) into a mask, for regional restyling without hand-painting. Runs locally on MPS. |
| **mflux Upscale (SeedVR2)** | One-step SeedVR2 upscaler. Loads its own model. |

## Supported models

Text-to-image (loader + sampler, no image input):

`dev`, `schnell`, `krea-dev`, `qwen`, `z-image`, `z-image-turbo`,
`flux2-klein-4b`, `flux2-klein-9b`, `ernie-image`, `ernie-image-turbo`, `fibo`,
`fibo-lite`, `ideogram4`.

Image-conditioned (loader + sampler + **mflux Image**):

`dev-kontext` (instruction edit), `dev-fill` (inpaint, needs a mask),
`dev-depth` (depth-guided, needs a depth map), `dev-redux` (image reference),
`dev-controlnet-canny` (needs a control image), `qwen-image-edit`, `fibo-edit`,
`flux2-klein-edit`, `krea-2-depth` (depth ControlNet, needs a `controlnet_path`).

A HuggingFace repo or local path can be typed into the loader's `model_path` to run
a model that is not in the dropdown; it is dispatched to the right architecture by
name, and rejected with a clear message if it is not a sampler model (for example a
SeedVR2 upscaler).

## Interior design / renovation

The edit and depth-ControlNet models make this node a practical interior-renovation
tool: restyle a real room photo (floors, walls, lighting, furniture) while keeping
the room's geometry. Two paths, both in `example_workflows/`:

- **Depth-locked restyle (`krea-2-depth`).** Load `krea-2-depth`, set `controlnet_path`
  to the depth-control checkpoint, and feed the room photo on `mflux Image`. DepthPro
  derives the depth automatically, so the render preserves the room's layout and volume
  while the prompt drives the new look. Feed a precomputed depth on `map_image` to skip
  the DepthPro step. See `interior_krea2_depth.json`.
- **Instruction edit with locked openings (`flux2-klein-edit` / `qwen-image-edit`).**
  Edit the whole room from a prompt, then paint a mask on `mflux Image` and set the
  sampler's `mask_mode` to `preserve` to hold windows and doors pixel-identical to the
  original (`inpaint` instead restricts the edit to the painted region). `mask_feather`
  softens the seam. See `interior_flux2_edit_mask_preserve.json`.

Anchor scale in the prompt with the room's real dimensions, and add an anti-hallucination
clause ("do not add or remove windows or doors") for the edit path.

**Style reference (Mood Board).** `dev-redux` blends the look of one or more reference
photos (a Pinterest shot, a magazine spread) with your prompt. Chain an `mflux Image` per
reference; each node's `strength` weights that reference in the blend, so you can dial "70%
this palette, 30% that". Redux generates a new image in the referenced style rather than
restyling one specific room, so use it for direction and inspiration, and the depth path
above when you need to keep an exact room's geometry. See `interior_redux_moodboard.json`.

**Regional restyle (restyle only the floor, keep the windows).** `mflux Auto Mask` segments
the room with an ADE20K model and turns a named region into a mask, so you can restyle one
surface without hand-painting. Wire it into `mflux Image` -> mask, then on the sampler pick
`inpaint` to change only that region ("restyle only the floor") or `preserve` to lock it
("keep the windows"). Segmentation runs locally on Apple Silicon (MPS). See
`interior_regional_restyle.json`.

## How the capability system works

On load, the node resolves the alias to the correct mflux variant class (so a fill
or controlnet model is never silently run as plain text-to-image), then builds a
`CapabilityProfile`:

- the set of `generate_image` parameters, read by introspection;
- which parameters are required, and which image roles the model declares;
- a small table of facts that introspection cannot reveal — which models are
  preset-driven, and where `negative_prompt` is accepted but ignored.

The sampler then applies three rules:

1. **Hard-block** an input image on a model that does not accept one, and a missing
   required image on a model that needs one.
2. For **preset-driven** models, ignore `steps`/`guidance` in `auto` mode (matching
   the mflux CLI) and note it. `override` mode forwards them and warns that the
   calibrated schedule is being replaced.
3. **Drop with a note** any parameter the model accepts but ignores, and forward
   only the arguments the model's `generate_image` actually declares.

This is verified by self-tests that introspect the installed mflux package, so they
catch upstream signature changes rather than drifting.

## Notes on mflux features

The node is fork-agnostic: it adapts to whatever mflux is installed. Capabilities
added by recent mflux work — LyCORIS LoKr LoRA loading, Z-Image sigma schedules and
shift, Qwen edit fixes — are exposed automatically when present and simply absent
when not.

## Running the tests

```
python tests/test_dispatch.py
python tests/test_sampler.py
```

No weights are downloaded; the tests only introspect the installed mflux package.

## Credits

- [mflux](https://github.com/filipstrand/mflux) by Filip Strand — the MLX
  implementation this node drives.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## License

MIT. See [LICENSE](LICENSE).
