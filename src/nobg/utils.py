from functools import wraps


# update docs for specific functions
def set_doc(doc):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__doc__ = doc
        return wrapper

    return decorator


# general template for models
def model_card_template(
    *, class_name: str, default_repo: str, citation: str | None = None
) -> str:
    citation_section = (
        f"""
## Citation
If you use this model, please cite:
```bibtex
{citation}
```
"""
        if citation
        else ""
    )
    return f"""---
{{{{ card_data }}}}
---

<p align="center">
<img src="https://usefeyn.com/feyn/feyn_mark.svg"/>
</p>

This model has been pushed to the Hub using the [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin) integration.

Library: [nobg]({{{{repo_url}}}})

## how to load
```
pip install nobg
```

use the AutoModel class
```python
from nobg import AutoModel
model = AutoModel.from_pretrained("{{{{ repo_id | default("{default_repo}", true) }}}}")
```
or you can use the model class directly
```python
from nobg import {class_name}
model = {class_name}.from_pretrained("{{{{ repo_id | default("{default_repo}", true) }}}}")
```

{citation_section}
## Contributions
Any contributions are welcome at https://github.com/feyninc/nobg

"""
