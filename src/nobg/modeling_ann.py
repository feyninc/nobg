from torch import nn
from .mixin import Revised_Mixin

model_card_template = """
---
{{ card_data }}
---

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin) integration.

Library: [nobg]({{repo_url}})

## how to load
```
pip install nobg
```

use the AutoModel class
```python
from nobg AutoModel
model = AutoModel.from_pretrained("{{ repo_id | default("nobg/ann", true) }}")
```
or you can use the model class directly
```python
from nobg import ANN
model = ANN.from_pretrained("{{ repo_id | default("nobg/ann", true ) }}")
```

## Contributions
Any contributions are welcome at https://github.com/feyninc/nobg.

<img src="https://usefeyn.com/feyn/feyn_mark.svg"/>

"""
default_conf = {"a": 2, "b": 1}


class ANN(
    nn.Module,
    Revised_Mixin,
    library_name="nobg",
    repo_url="https://github.com/feyninc/nobg",
    tags=["nobg", "ann"],
    model_card_template=model_card_template,
):
    """an AI model for visual question answering"""

    def __init__(self, cfg: dict = default_conf):
        super().__init__()
        self.cfg = cfg
        self.layer = nn.Linear(cfg["a"], cfg["b"], bias=False)

    def forward(self, input_ids):
        return self.layer(input_ids)
