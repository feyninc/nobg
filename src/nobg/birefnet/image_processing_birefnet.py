from typing import TYPE_CHECKING, Union

import torch
from transformers.image_processing_backends import TorchvisionBackend
from transformers.image_processing_base import BatchFeature
from transformers.image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
)
from transformers.processing_utils import ImagesKwargs, Unpack
from transformers.utils import TensorType

from ..utils import cutout, post_process_alpha_matting, refine_foreground, set_doc

if TYPE_CHECKING:
    from PIL.Image import Image


class BiRefNetImageProcessor(TorchvisionBackend):
    """Image processor for BiRefNet.

    Encapsulates the preprocessing used everywhere in the repo: RGB convert,
    square resize to ``image_size`` (bilinear), scale to ``[0, 1]`` and ImageNet
    normalization. ``preprocess`` optionally takes ``segmentation_maps`` and
    returns binarized ``labels`` for training. ``post_process_alpha_matting``
    turns raw model logits into alpha mattes, ``refine_foreground`` estimates
    unmixed foreground colors, and ``cutout`` composites a matte onto the
    original image.
    """

    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    # transformers reads/writes this class attr as part of the
    # preprocessor_config.json schema, so a dict default is the required shape.
    size = {"height": 1024, "width": 1024}  # noqa: RUF012
    default_to_square = True
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    rescale_factor = 1 / 255

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        super().__init__(**kwargs)

    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = None,
        **kwargs: Unpack[ImagesKwargs],
    ) -> BatchFeature:
        return super().preprocess(images, segmentation_maps, **kwargs)

    def _preprocess_image_like_inputs(  # ty: ignore[invalid-method-override]
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        return_tensors: str | TensorType | None,
        device: Union[str, "torch.device"] | None = None,
        **kwargs,
    ) -> BatchFeature:
        images = self._prepare_image_like_inputs(
            images=images,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
        )
        # `_preprocess` returns a BatchFeature; index it so the outer call below
        # controls tensor stacking via `return_tensors`.
        pixel_values = self._preprocess(images, return_tensors=None, **kwargs)
        data = {"pixel_values": pixel_values["pixel_values"]}

        if segmentation_maps is not None:
            masks = self._prepare_image_like_inputs(
                images=segmentation_maps,
                expected_ndims=2,
                do_convert_rgb=False,
                input_data_format=ChannelDimension.FIRST,
            )
            # Resize + rescale to [0, 1] like the images (mirrors L-convert -> Resize
            # -> ToTensor) but never normalize, then binarize to a hard alpha mask.
            mask_kwargs = {**kwargs, "do_normalize": False, "do_rescale": True}
            processed = self._preprocess(masks, return_tensors=None, **mask_kwargs)  # ty: ignore[invalid-argument-type]
            # Keep the channel dim: BiRefNet expects labels of shape (B, 1, H, W).
            data["labels"] = [
                (m > 0.5).to(torch.float32) for m in processed["pixel_values"]
            ]

        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process_alpha_matting(
        self,
        outputs,
        target_sizes: list[tuple[int, int]] | None = None,
    ) -> list[torch.Tensor]:
        """Convert raw BiRefNet logits into per-image alpha mattes in ``[0, 1]``.

        Args:
            outputs: The model output; a dict with a ``"logits"`` key or an object
                exposing ``.logits`` of shape ``(B, 1, H, W)``.
            target_sizes: Optional list of ``(height, width)`` tuples, one per image;
                each matte is bilinearly resized to its target size.

        Returns:
            A list of ``(H, W)`` tensors with values in ``[0, 1]``.
        """
        return post_process_alpha_matting(outputs, target_sizes)

    @staticmethod
    @set_doc(refine_foreground.__doc__)
    def refine_foreground(
        image: Union[torch.Tensor, "Image"],
        alpha: Union[torch.Tensor, "Image"],
        r: int = 90,
    ) -> Union[torch.Tensor, "Image"]:
        return refine_foreground(image, alpha, r)

    @staticmethod
    @set_doc(cutout.__doc__)
    def cutout(
        image: "Image",
        alpha: Union[torch.Tensor, "Image"],
        refine: bool = False,
        r: int = 90,
    ) -> "Image":
        return cutout(image, alpha, refine, r)

    def push_to_hub(self, repo_id: str, **kwargs) -> str:
        if "/" not in repo_id:
            from huggingface_hub import whoami

            repo_id = f"{whoami()['name']}/{repo_id}"
        return super().push_to_hub(repo_id, **kwargs)
