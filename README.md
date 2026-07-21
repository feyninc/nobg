# nobg

A library for image and video matting with HuggingFace Hub integration.

## Installation

```bash
pip install nobg
```

## Usage

### AutoModel

Automatically detect and load the correct model architecture from a HuggingFace repo:

```python
from nobg import AutoModel

model = AutoModel.from_pretrained("nobg/BiRefNet")
```

### Load a specific model

```python
from nobg import AutoModel

model = AutoModel.from_pretrained("nobg/BiRefNet")
```

### AutoProcessor

`AutoProcessor` loads the matching image processor for a repo. It handles the
resize + ImageNet normalization on the way in, and turns raw model logits back
into an alpha matte at the original resolution on the way out.

```python
from nobg import AutoProcessor

processor = AutoProcessor.from_pretrained("nobg/FeyNobg")
```

### Remove a background with FeyNobg

`nobg/FeyNobg` is the strongest published model. The processor produces the
normalized `pixel_values` the model expects and post-processes the raw logits
`(B, 1, 1024, 1024)` into an alpha matte you can composite onto the image.

```python
import torch
from loadimg import load_img

from nobg import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("nobg/FeyNobg").eval()
processor = AutoProcessor.from_pretrained("nobg/FeyNobg")

image = load_img("input.jpg").convert("RGB")
inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(pixel_values=inputs["pixel_values"])

# Alpha matte in [0, 1] resized back to the original image size
alpha = processor.post_process_alpha_matting(
    outputs, target_sizes=[(image.height, image.width)]
)[0]

# Composite the cutout onto a transparent background
processor.cutout(image, alpha).save("output.png")
```

| Input | Output |
|:-----:|:------:|
| ![input](assets/feyn_mark.png) | ![cutout](assets/feyn_mark_cutout.png) |

### Push to HuggingFace Hub

```python
model.push_to_hub("your-username/model-name")
```

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
