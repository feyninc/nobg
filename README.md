<div align="center">

# nobg

**Open-source background removal & image matting, with first-class HuggingFace Hub integration.**

[![PyPI](https://img.shields.io/pypi/v/nobg?color=blue&label=PyPI)](https://pypi.org/project/nobg/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/nobg?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/nobg)
[![Python](https://img.shields.io/pypi/pyversions/nobg?label=Python)](https://pypi.org/project/nobg/)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow)](LICENSE)


[![](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Model-blue)](https://huggingface.co/feyninc/FeyNobg)
[![](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Space-blue)](https://huggingface.co/spaces/feyninc/feynobg)
[![](https://img.shields.io/badge/%F0%9F%A4%97%20HF-Org-blue)](https://huggingface.co/feyninc)
[![](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/feyninc/nobg)
[![](https://img.shields.io/badge/Contributing-Guide-green)](CONTRIBUTING.md)

| Input | Output |
|:-----:|:------:|
| <img src="assets/feyn_mark.png" width="320"> | <img src="assets/feyn_mark_cutout.png" width="320"> |

</div>


## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Zoo](#model-zoo)
- [Usage](#usage)
  - [AutoModel & AutoProcessor](#automodel--autoprocessor)
  - [SAM3: text-promptable cutouts](#sam3-text-promptable-cutouts)
  - [Batched inference](#batched-inference)
  - [Refining the foreground](#refining-the-foreground)
  - [GPU & half precision](#gpu--half-precision)
  - [Fine-tuning on custom data](#fine-tuning-on-custom-data)
  - [Re-parameterizing a checkpoint](#re-parameterizing-a-checkpoint)
  - [Push to HuggingFace Hub](#push-to-huggingface-hub)
  - [ONNX export](#onnx-export)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [License](#license)

## Installation

```bash
uv add nobg
```

<details>
<summary>From source (with <code>uv</code>)</summary>

```bash
git clone https://github.com/feyninc/nobg.git
cd nobg
uv sync
```

</details>

Requires Python ≥ 3.10, `torch` ≥ 2.0 and `torchvision` ≥ 0.15. Those two are deliberately
**not** installed for you — pick the build that matches your hardware (a CPU wheel, a CUDA
one, ROCm) and install it yourself:

```bash
uv add torch torchvision
```

See [`pyproject.toml`](https://github.com/feyninc/nobg/blob/20af1e135d042b74f8a161a82bcd6f7f53ed7c33/pyproject.toml) for the full dependency set.

## Quick Start

Remove a background in three lines:

```python
from nobg import AutoModel

model = AutoModel.from_pretrained("feyninc/FeyNobg")
model.process("input.jpg").save("output.png")
```

`process` handles the whole pipeline — load, preprocess, forward under `no_grad` in eval
mode, post-process, composite — and builds the processor the model's own config implies, so
there is nothing else to load. `image` takes anything
[`loadimg`](https://github.com/not-lain/loadimg) accepts: a path, URL, base64 string, numpy
array or PIL image. Pass a list to get a list back, each matte returned at its own original
resolution:

```python
for cut, path in zip(model.process(["a.jpg", "b.jpg"]), ("a.png", "b.png")):
    cut.save(path)
```

Useful keywords: `batch_size` (images per forward pass; defaults to 1 to keep peak memory
flat), `return_type="alpha"` for the raw `(H, W)` matte tensors instead of RGBA cutouts, and
any remaining kwargs go to the processor.

`predict` is the same call with the processor passed in — reach for it when you already have
one, or when a checkpoint's `preprocessor_config.json` differs from its `config.image_size`
(`process` trusts the model config):

```python
from nobg import AutoProcessor

processor = AutoProcessor.from_pretrained("feyninc/FeyNobg")
model.predict(processor, "input.jpg").save("output.png")
```

The processor comes first, then the inputs that vary: `predict(processor, image, prompt,
boxes)`, each one optional after `image` (BiRefNet takes neither prompt nor boxes; SAM3 takes
both). `model.default_processor()` returns the one `process` would build, if you want it
without the Hub round-trip.

Or drive the steps yourself when you need the intermediates:

```python
import torch
from loadimg import load_img

image = load_img("input.jpg").convert("RGB")
inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(pixel_values=inputs["pixel_values"])

alpha = processor.post_process_alpha_matting(
    outputs, target_sizes=[(image.height, image.width)]
)[0]
processor.cutout(image, alpha).save("output.png")
```

Or try it in the browser first: **[🤗 FeyNobg Space](https://huggingface.co/spaces/feyninc/feynobg)**.

## Model Zoo

| Model | Repo | Params | Resolution | Task | Notes |
|:------|:-----|:------:|:----------:|:-----|:------|
| **FeyNobg** | [`feyninc/FeyNobg`](https://huggingface.co/feyninc/FeyNobg) | 0.3 B | 1024 × 1024 | Background removal / matting | Strongest published model, start here |
| **SAM3** | [`facebook/sam3`](https://huggingface.co/facebook/sam3) | 0.84 B | 1008 × 1008 | Background removal + text-promptable segmentation | Load with `Sam3.from_origin`. Picks the subject well; edges stay softer than FeyNobg. Gated; weights are under Meta's [SAM License](https://huggingface.co/facebook/sam3/blob/main/LICENSE) |

## Usage

### AutoModel & AutoProcessor

`AutoModel` reads the repo tags and returns the concrete class. `AutoProcessor`
reads `preprocessor_config.json` (or falls back to the model config) and returns
the matching image processor.

```python
from nobg import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("feyninc/FeyNobg")
processor = AutoProcessor.from_pretrained("feyninc/FeyNobg")
```

Concrete classes work too, if you'd rather be explicit:

```python
from nobg import BiRefNet, BiRefNetImageProcessor

model = BiRefNet.from_pretrained("feyninc/FeyNobg")
processor = BiRefNetImageProcessor.from_pretrained("feyninc/FeyNobg")
```

Constructing from scratch (random init) uses the config dataclass:

```python
from nobg import BiRefNet
from nobg.birefnet.modeling_birefnet import BiRefNetConfig

model = BiRefNet(BiRefNetConfig(image_size=512, embed_dim=128))
```

### SAM3: text-promptable cutouts

`Sam3` wraps [`transformers`' SAM3](https://huggingface.co/docs/transformers/model_doc/sam3)
and exposes its prompt-conditioned segmentation as a single alpha matte, so it drops into the
same flow as BiRefNet:

```python
from nobg import Sam3

model = Sam3.from_origin("facebook/sam3")
model.process("input.jpg").save("output.png")
```

Because SAM3 is open-vocabulary, you can cut out *specific* things by passing a prompt as
the second argument — this is the capability BiRefNet doesn't have:

```python
model.process("input.jpg", "the dog").save("dog.png")
```

With no `prompt`, the processor supplies `default_prompt` (`"the main foreground subject"`),
which is what makes prompt-free background removal work.

The third argument is `boxes` — a visual prompt, in the original image's pixel
coordinates. Use it when the thing you want is easier to point at than to name:

```python
model.process("input.jpg", None, [[120, 80, 460, 720]]).save("cutout.png")
```

Unlike `prompt`, `boxes` is **per-image**: pass `[[x1, y1, x2, y2], ...]` for one image, or
one such list per image for a batch. Boxes and a prompt can be combined; with boxes and no
prompt, SAM3 segments what the boxes point at.

```python
images = ["a.jpg", "b.jpg"]
boxes = [[[10, 10, 200, 300]], [[40, 60, 380, 500], [400, 20, 620, 260]]]
cuts = model.process(images, "the dog", boxes)
```

`process` builds its processor from the model config: image size and `default_prompt` come
straight from it, and the CLIP tokenizer is loaded from whatever repo `from_origin` read the
weights from, falling back to the ungated `openai/clip-vit-large-patch14` (SAM3's text tower
*is* CLIP's). Pass `tokenizer=` a repo id, a directory or an instance to override that. When
you'd rather hold the processor yourself, `predict` is the same call with it passed in first:

```python
from nobg import Sam3Processor

processor = Sam3Processor.from_pretrained("facebook/sam3")
model.predict(processor, "input.jpg", "the dog").save("dog.png")
```

By default the matte comes from SAM3's own prompt-conditioned **semantic** head
(`config.aggregate="semantic"`). Set `aggregate` to `"max"` or `"mean"` to build it from the
union of per-object instance masks instead — that path respects `score_threshold` (how many
detected objects land in the matte, falling back to the best-scoring one so the matte is
never empty), but produces a noticeably softer alpha:

```python
model = Sam3.from_origin("facebook/sam3", aggregate="max")
model.predict(processor, "input.jpg", score_threshold=0.5)
```

Measured against FeyNobg on two photos, the semantic head is the clear default: MAE
0.035/0.039 with 19/29 % of pixels at intermediate alpha, versus 0.144/0.150 and 71/79 % for
the instance union. Reach for `"max"`/`"mean"` when you specifically want the matte to track
the detected instance set.

The step-by-step form, when you want the instance-level outputs:

```python
import torch
from loadimg import load_img

image = load_img("input.jpg").convert("RGB")
inputs = processor(images=image, text="the dog", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

alpha = processor.post_process_alpha_matting(
    outputs, target_sizes=[(image.height, image.width)]
)[0]
processor.cutout(image, alpha).save("output.png")
```

Because the matte is never empty, a prompt for something that *isn't in the image* still
returns one. Read `presence_logits` to tell the difference — SAM3's presence head is a
reliable confidence signal (on a cosplay photo: `"the person"` → 0.97, `"the hat"` → 0.85,
`"the dog"` → 0.001):

```python
confidence = outputs["presence_logits"].sigmoid().item()
```

The per-object outputs come through untouched (`pred_masks`, `pred_boxes`, `pred_logits`,
`presence_logits`, `semantic_seg`), so `processor.image_processor.post_process_instance_segmentation`
still works for instance-level use.

**Which model to reach for.** SAM3 finds the right subject — on a test photo its matte
agrees with FeyNobg at IoU 0.98 — but it's a detector, not a matting model: masks are
predicted at a fraction of the input resolution and upsampled, so edges stay softer
(19 – 29 % of pixels land at intermediate alpha, versus 3 % for FeyNobg). Use **SAM3 when you
need to choose *what* to cut out**, and **FeyNobg when you need hair-level edges**.

> [!NOTE]
> nobg ships **no SAM weights** — `from_origin("facebook/sam3")` downloads them from Meta's
> gated repo, under Meta's [SAM License](https://huggingface.co/facebook/sam3/blob/main/LICENSE)
> rather than nobg's Apache-2.0. Accept it on the Hub first. Only the Apache-2.0
> `transformers` implementation is used in code.

Note that `pixel_values` must be exactly `config.image_size` square — the vision tower's
rotary embeddings are fixed-size — so always preprocess through `Sam3Processor`.

### Batched inference

`predict` takes a list and returns one result per input, each at its original resolution.
`batch_size` sets how many go through each forward pass:

```python
paths = ("a.jpg", "b.jpg", "c.jpg")
for cut, path in zip(model.predict(processor, list(paths), batch_size=4), paths):
    cut.save(path.replace(".jpg", ".png"))
```

Or drive it manually — `post_process_alpha_matting` takes one target size per image, so
mattes come back at each original resolution:

```python
images = [load_img(p).convert("RGB") for p in paths]
inputs = processor(images, return_tensors="pt")

with torch.no_grad():
    outputs = model(pixel_values=inputs["pixel_values"])

mattes = processor.post_process_alpha_matting(
    outputs, target_sizes=[(im.height, im.width) for im in images]
)
for im, alpha, path in zip(images, mattes, ("a.png", "b.png", "c.png")):
    processor.cutout(im, alpha).save(path)
```

The same pattern handles video: decode to frames, batch them, composite back.

### Refining the foreground

A soft matte leaves the old background mixed into every semi-transparent pixel, so
compositing the original pixels onto a new background shows a halo of the old one —
most visible on hair, fur and motion blur. `refine_foreground` estimates the unmixed
foreground color for those pixels, and `cutout(refine=True)` applies it in place:

```python
processor.cutout(image, alpha, refine=True).save("output.png")
```

It is pure torch, so it runs wherever its inputs live — keep the tensors on the GPU
and the refinement stays there too:

```python
alpha = processor.post_process_alpha_matting(
    outputs, target_sizes=[(image.height, image.width)]
)[0]
foreground = processor.refine_foreground(pixel_tensor.cuda(), alpha.cuda())
```

`r` (default `90`) sets how far the estimator reaches for a color to borrow; the cost
grows about linearly with it.

### GPU & half precision

```python
model = AutoModel.from_pretrained("feyninc/FeyNobg").eval().to("cuda")
inputs = processor(image, return_tensors="pt").to("cuda")

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    outputs = model(pixel_values=inputs["pixel_values"])
```

### Fine-tuning on custom data

NoBg provides the model, processor, and loss needed to train BiRefNet on your own image and mask pairs. Because the model plugs into the Hugging Face `Trainer`, you get its full training loop, checkpointing, and evaluation for free.

```python
from transformers import Trainer, TrainingArguments
from nobg import AutoProcessor, AutoModel

model = AutoModel.from_pretrained("nobg/FeyNobg")
processor = AutoProcessor.from_pretrained("nobg/FeyNobg")


def collate(examples):
    batch = processor(
        images=[ex["image"] for ex in examples],
        segmentation_maps=[ex["mask"].convert("L") for ex in examples],
        return_tensors="pt",
    )
    return {"pixel_values": batch["pixel_values"], "labels": batch["labels"]}


trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="outputs", learning_rate=2e-5),
    train_dataset=dataset,
    data_collator=collate,
)
trainer.train()
```

<details>
<summary>Swapping the loss</summary>

`model.criterion` is a plain function attribute, not a submodule, so it never
enters the state dict and you can replace it outright:

```python
from nobg.loss import birefnet_loss, iou_loss, ssim_loss


def my_loss(scaled_preds, gt):
    return birefnet_loss(scaled_preds, gt) + 5 * iou_loss(
        scaled_preds[-1].sigmoid(), gt
    )


model.criterion = my_loss
```

</details>

### Re-parameterizing a checkpoint

`BiRefNet.from_origin` builds a new model from an existing one, injecting every
weight whose key and shape still match and freshly initializing the rest. Handy
for changing resolution, growing the decoder, or migrating pre-0.2.0 checkpoints.

```python
from nobg import BiRefNet

model = BiRefNet.from_origin("feyninc/FeyNobg", image_size=2048)
```

`origin` may be a Hub repo id, a local directory with `config.json` +
`model.safetensors`, or a live `nn.Module`.

### Push to HuggingFace Hub

```python
model.push_to_hub("your-username/model-name")
processor.push_to_hub("your-username/model-name")
```

A bare name is auto-prefixed with your Hub username, and a model card is
generated from the shared template.

### ONNX export

Every model has an ONNX counterpart of each of those three calls. They need the extra:

```bash
uv add "nobg[onnx]"
```

```python
model.onnx_save_pretrained("onnx-out")          # -> onnx-out/model.onnx + config.json + README.md
model.onnx_push_to_hub("your-username/model-name-onnx")
```

Loading gives back an `OnnxModel` — the graph under `onnxruntime`, with the same `process`,
`predict`, `default_processor` and `config` as the torch model, so it drops into the code
above unchanged:

```python
from nobg import BiRefNet

model = BiRefNet.onnx_from_pretrained("your-username/model-name-onnx")
model.process("input.jpg").save("output.png")
```

`providers=` picks the execution provider (defaults to everything installed, so an
`onnxruntime-gpu` build uses the GPU); `session_options=` takes an
`onnxruntime.SessionOptions`.

Two things differ from the torch model. **Shapes are fixed at export time**, batch size
included — transformers' Swin windowing reshapes with Python ints, which pins the batch no
matter what `dynamic_axes` claims, so export at the batch size you'll run at and read it back
off `model.batch_size`:

```python
model.onnx_save_pretrained("onnx-out", batch_size=4)
```

And **only the matte is exported**: the graph returns `logits` alone, without BiRefNet's
`intermediate_logits` or SAM3's instance heads (`pred_masks`, `presence_logits`, …). Keep the
torch model for those.

The graph's inputs come from `onnx_dummy_inputs()` — `pixel_values` for BiRefNet, plus
`input_ids`/`attention_mask` for SAM3, whose text prompt is therefore baked in as *shape* only:
any prompt `Sam3Processor` produces (it pads to 32 tokens) runs on the same graph. Pass
`dummy_inputs=` to trace a variant, e.g. a box-promptable SAM3:

```python
inputs = model.onnx_dummy_inputs()
inputs["input_boxes"] = torch.zeros(1, 1, 4)  # one box per image — only the shape is traced
model.onnx_save_pretrained("onnx-out", dummy_inputs=inputs)
```

Anything else goes to `torch.onnx.export`, `opset_version` included (the default, 19, is the
floor for BiRefNet's `DeformConv`).

## Acknowledgement

- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) by Peng Zheng et al., the
  architecture and training recipe this library builds on.
- [SAM 3](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
  by Nicolas Carion et al. (Meta AI), wrapped here through its Apache-2.0 `transformers`
  implementation — no SAM weights are redistributed.
- [`transformers`](https://github.com/huggingface/transformers) and
  [`huggingface_hub`](https://github.com/huggingface/huggingface_hub) for the
  backbone, processor base and Hub integration.

## Citation

```bibtex
@software{nobg,
  title={nobg: Open Source Background Removal Models for Image and Video Matting},
  author={Hichri, Hafedh},
  year={2026},
  url={https://github.com/feyninc/nobg},
  license={Apache-2.0},
}
```
