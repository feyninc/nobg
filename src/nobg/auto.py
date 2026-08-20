import json

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download, model_info
from transformers.image_processing_base import ImageProcessingMixin

from .birefnet.image_processing_birefnet import BiRefNetImageProcessor
from .birefnet.modeling_birefnet import BiRefNet
from .sam3.image_processing_sam3 import DEFAULT_PROMPT, Sam3Processor
from .sam3.modeling_sam3 import Sam3
from .utils import set_doc

BIREFNET_TAGS = ("nobg-birefnet", "birefnet")
SAM3_TAGS = ("nobg-sam3", "sam3")


# Inspect repo parameters and return the appropriate model class
class AutoModel:
    @classmethod
    @set_doc(PyTorchModelHubMixin.from_pretrained.__doc__)
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        tags: list[str] = model_info(pretrained_model_name_or_path).tags or []
        if any(t in tags for t in BIREFNET_TAGS):
            return BiRefNet.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        elif any(t in tags for t in SAM3_TAGS):
            return Sam3.from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        else:
            raise ValueError("this model is not part of nobg")


# Maps the `image_processor_type` recorded in preprocessor_config.json (or the
# nested `image_processor` block of processor_config.json) to the nobg class that
# handles it. Mirrors transformers' IMAGE_PROCESSOR_MAPPING; append new
# processors here. The `Sam3ImageProcessor*` entries are upstream type names:
# a raw transformers SAM3 repo such as `facebook/sam3` records those, and nobg's
# Sam3Processor is a drop-in superset, so map them onto it rather than raising.
PROCESSOR_TYPES = {
    "BiRefNetImageProcessor": BiRefNetImageProcessor,
    "Sam3Processor": Sam3Processor,
    "Sam3ImageProcessor": Sam3Processor,
    "Sam3ImageProcessorFast": Sam3Processor,
}


# Inspect repo parameters and return the appropriate image processor
class AutoProcessor:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        try:
            config_dict, _ = ImageProcessingMixin.get_image_processor_dict(
                pretrained_model_name_or_path, **kwargs
            )
        except OSError:
            # fallback for repos that don't have a preprocessor_config.json (e.g. legacy BiRefNet)
            return cls._from_model_config(pretrained_model_name_or_path, **kwargs)

        processor_type = config_dict.get("image_processor_type")
        processor_class = PROCESSOR_TYPES.get(processor_type)
        if processor_class is None:
            raise ValueError("this model is not part of nobg")
        # Re-load through the resolved class (mirrors transformers, which reads the
        # type then calls the class's own from_pretrained to handle all kwargs).
        return processor_class.from_pretrained(pretrained_model_name_or_path, **kwargs)

    @staticmethod
    def _from_model_config(pretrained_model_name_or_path, **kwargs):
        tags: list[str] = model_info(pretrained_model_name_or_path).tags or []
        is_birefnet = any(t in tags for t in BIREFNET_TAGS)
        is_sam3 = any(t in tags for t in SAM3_TAGS)
        if not is_birefnet and not is_sam3:
            raise ValueError("this model is not part of nobg")
        config_file = hf_hub_download(
            pretrained_model_name_or_path,
            "config.json",
            token=kwargs.get("token"),
        )
        with open(config_file) as f:
            config = json.load(f)
        if is_birefnet:
            image_size = config.get("image_size", 1024)
            return BiRefNetImageProcessor(
                size={"height": image_size, "width": image_size}
            )
        # SAM3 also needs a text tokenizer, which no config.json describes, so
        # take it from the repo alongside the size-only image processor.
        from transformers import AutoTokenizer
        from transformers.models.sam3.image_processing_sam3 import Sam3ImageProcessor

        image_size = config.get("image_size", 1008)
        return Sam3Processor(
            Sam3ImageProcessor(size={"height": image_size, "width": image_size}),
            AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, token=kwargs.get("token")
            ),
            default_prompt=config.get("default_prompt", DEFAULT_PROMPT),
        )
