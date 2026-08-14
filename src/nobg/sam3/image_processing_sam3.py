from typing import TYPE_CHECKING, Union

import torch
from transformers.models.sam3.processing_sam3 import (
    Sam3Processor as TransformersSam3Processor,
)

from ..utils import cutout, post_process_alpha_matting, refine_foreground, set_doc

if TYPE_CHECKING:
    from PIL.Image import Image

DEFAULT_PROMPT = "the main foreground subject"


class Sam3Processor(TransformersSam3Processor):
    """Processor for nobg's SAM3 matting wrapper.

    Subclasses ``transformers``' ``Sam3Processor`` — the image processor and CLIP
    tokenizer wiring, box handling and instance post-processing are all
    inherited unchanged — and adds the two methods nobg's background-removal
    contract needs: ``post_process_alpha_matting``, ``refine_foreground`` and
    ``cutout``.

    The one behavioural change is that ``text`` defaults to ``default_prompt``
    instead of ``None``. SAM3 will not run without a prompt, so this is what
    makes ``processor(images=image)`` work like ``BiRefNetImageProcessor``,
    while any explicit ``text=`` still gives open-vocabulary cutouts.
    """

    def __init__(
        self,
        image_processor,
        tokenizer,
        target_size: int | None = None,
        point_pad_value: int = -10,
        default_prompt: str = DEFAULT_PROMPT,
        **kwargs,
    ):
        r"""
        default_prompt (`str`, *optional*, defaults to `"the main foreground subject"`):
            Text prompt used when ``__call__`` receives no ``text``. Should match
            the ``default_prompt`` on the model's ``Sam3Config``.
        """
        super().__init__(
            image_processor,
            tokenizer,
            target_size=target_size,
            point_pad_value=point_pad_value,
            **kwargs,
        )
        self.default_prompt = default_prompt

    def _resolve_text_prompts(self, text, input_boxes):
        # Upstream returns None when there is no text and no boxes, which makes
        # Sam3Model.forward raise. Substitute the prompt-free default instead.
        resolved = super()._resolve_text_prompts(text, input_boxes)
        if resolved is None:
            return self.default_prompt
        return resolved

    def post_process_alpha_matting(
        self,
        outputs,
        target_sizes: list[tuple[int, int]] | None = None,
    ) -> list[torch.Tensor]:
        """Convert raw ``Sam3`` matte logits into per-image alpha mattes in ``[0, 1]``.

        Takes the wrapper's aggregated ``logits`` (not the per-instance
        ``pred_masks``); use the inherited ``post_process_instance_segmentation``
        on the image processor for per-object outputs.

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
