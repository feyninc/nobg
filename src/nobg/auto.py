from .modeling_gpt2 import GPT2
from .modeling_ann import ANN
from huggingface_hub import model_info, PyTorchModelHubMixin

from functools import wraps


def set_doc(doc):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__doc__ = doc
        return wrapper

    return decorator


# Inspect repo parameters and return the appropriate model class


class AutoModel:
    @classmethod
    @set_doc(PyTorchModelHubMixin.from_pretrained.__doc__)
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        
        tags = model_info(pretrained_model_name_or_path).tags
        if "gpt2" in tags:
            return GPT2.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        elif "ann" in tags:
            return ANN.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        else:
            raise ValueError("this model is not part of nobg")
