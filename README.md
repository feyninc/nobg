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
  - [Batched inference](#batched-inference)
  - [GPU & half precision](#gpu--half-precision)
  - [Fine-tuning on custom data](#fine-tuning-on-custom-data)
  - [Re-parameterizing a checkpoint](#re-parameterizing-a-checkpoint)
  - [Push to HuggingFace Hub](#push-to-huggingface-hub)
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

Requires Python ≥ 3.10 and `torch` ≥ 2.0. See [`pyproject.toml`](https://github.com/feyninc/nobg/blob/20af1e135d042b74f8a161a82bcd6f7f53ed7c33/pyproject.toml) for the full dependency set.

## Quick Start

Remove a background in ten lines:

```python
import torch
from loadimg import load_img
from nobg import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("feyninc/FeyNobg").eval()
processor = AutoProcessor.from_pretrained("feyninc/FeyNobg")

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

### Batched inference

Pass a list of images; `post_process_alpha_matting` takes one target size per image,
so mattes come back at each original resolution.

```python
images = [load_img(p).convert("RGB") for p in ("a.jpg", "b.jpg", "c.jpg")]
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
    return birefnet_loss(scaled_preds, gt) + 5 * iou_loss(scaled_preds[-1].sigmoid(), gt)

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

## Acknowledgement

- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) by Peng Zheng et al., the
  architecture and training recipe this library builds on.
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
