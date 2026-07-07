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
from nobg import ANN, GPT2, BiRefNet

model = ANN.from_pretrained("nobg/ann")
model = GPT2.from_pretrained("nobg/gpt2")
model = BiRefNet.from_pretrained("nobg/BiRefNet")
```

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
  license={MIT},
}
```
