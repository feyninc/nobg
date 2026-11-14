# NoBG for ComfyUI

The [NoBG](https://github.com/feyninc/nobg) pack adds **NoBG Model Loader**,
**NoBG Remove Background**, and **NoBG Refine Foreground**. Install the pack
from the [Comfy Registry](https://registry.comfy.org/publishers/chonk-lain/nodes/nobg)
via ComfyUI Manager and restart ComfyUI. It installs the `nobg` Python library
in ComfyUI's environment, not model weights. For a source checkout, install
the library with ComfyUI's Python (`python -m pip install -e /path/to/nobg`,
substituting ComfyUI's Python executable), link `/path/to/nobg/comfyui` as
`ComfyUI/custom_nodes/nobg`, and restart. Disable the older `ComfyUI-NoBG`
pack if installed to avoid duplicate node names.

In **NoBG Model Loader**, choose `feyninc/FeyNobg` for matting or
`facebook/sam3` for a text-prompted subject. The Hub presets download on
first use and cache weights; CPU inference can be slow, especially with
SAM3 or refinement. For offline use, place `config.json` and
`model.safetensors` in `ComfyUI/models/nobg/<name>/`, restart, and choose
`local: <name>`. The loader identifies the architecture from `config.json`,
not the folder name. A gated upstream `facebook/sam3` snapshot requires
approved Hub access to download its `config.json` and `model.safetensors`;
its config/weights differ from nobg-format SAM3 checkpoints. Offline SAM3
also needs tokenizer files in the checkpoint directory or a previously
cached `openai/clip-vit-large-patch14` tokenizer.

SAM3's [checkpoint](https://huggingface.co/facebook/sam3) is gated under
Meta's [SAM License](https://huggingface.co/facebook/sam3/blob/main/LICENSE):
request/accept access, then run `hf auth login` with an authorized token in
ComfyUI's environment, or set `HF_TOKEN` before launching ComfyUI. To prompt
SAM3, type a description such as `the dog` into the removal node's `prompt`
text widget. Leave it empty for SAM3's default prompt; BiRefNet ignores it.

Connect **Load Image** and **NoBG Model Loader** to **NoBG Remove
Background**, then connect its `cutout` output directly to **Save Image**
for transparent RGBA PNGs. `foreground` is RGB without transparency;
`mask` is foreground alpha (1 = keep, 0 = transparent). ComfyUI's **Load
Image** mask inverts alpha (1 = transparent), so invert it before using it
as a NoBG foreground mask. To remove halos on hair or fur, enable
`refine_foreground` or pass an RGB image and foreground mask to **NoBG
Refine Foreground**, then save its `cutout`. Image batches are supported;
the removal node's `batch_size` input sets 1–16 images per inference pass
(default 1). Increase it for throughput if memory allows, or leave it at
1 when GPU memory is tight.

The two JSON/JPG workflow pairs in [`example_workflows/`](example_workflows/)
are included with the Registry download for ComfyUI's template browser. The
JPG thumbnails preview a dog cutout; the JSON workflows do not include the dog
photo. After opening either template, select or upload your own image in
**Load Image**; the referenced `example.png` is a placeholder, not a bundled input.
The SAM3 template also requires approved Hub access and authentication as
described above.
See the [main README](https://github.com/feyninc/nobg#comfyui-integration)
for more usage and release guidance.

Registry identity: https://registry.comfy.org/publishers/chonk-lain/nodes/nobg

`nobg` 0.3.1 on PyPI does **not** contain the node implementation. The first
Registry release (0.1.0) requires a *published* `nobg>=0.3.2` wheel containing
`nobg/comfyui/nodes.py`. Publish that PyPI version first; do not install the
Registry pack against 0.3.1. Subsequent Registry versions are independent of
the library version.
