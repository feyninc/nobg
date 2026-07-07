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

model = AutoModel.from_pretrained("nobg/ann")
```

### Load a specific model

```python
from nobg import ANN, GPT2

model = ANN.from_pretrained("nobg/ann")
model = GPT2.from_pretrained("nobg/gpt2")
```

### Push to HuggingFace Hub

```python
model.push_to_hub("your-username/model-name")
```