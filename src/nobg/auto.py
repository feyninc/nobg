from .birefnet.modeling_birefnet import BiRefNet
from .utils import set_doc
from huggingface_hub import model_info, PyTorchModelHubMixin


# Inspect repo parameters and return the appropriate model class
class AutoModel:
    @classmethod
    @set_doc(PyTorchModelHubMixin.from_pretrained.__doc__)
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        tags: list[str] = model_info(pretrained_model_name_or_path).tags or []
        if "birefnet" in tags:
            return BiRefNet.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        else:
            raise ValueError("this model is not part of nobg")
