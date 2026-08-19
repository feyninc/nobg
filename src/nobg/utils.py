from collections.abc import Mapping
from functools import wraps
from typing import TYPE_CHECKING, Union, cast

import torch

if TYPE_CHECKING:
    from PIL.Image import Image


# shared by every nobg image processor's `cutout` staticmethod
def cutout(
    image: "Image",
    alpha: Union[torch.Tensor, "Image"],
    refine: bool = False,
    r: int = 90,
) -> "Image":
    """Composite an alpha matte onto ``image``, returning an RGBA cutout.

    ``alpha`` may be a ``[0, 1]`` tensor of shape ``(H, W)`` or a PIL image; it
    is resized to ``image.size`` if needed.

    ``refine`` replaces the image's colors with ``refine_foreground``'s estimate
    of the unmixed foreground first, which removes the halo of the old
    background around soft edges (hair, fur, motion blur) at the cost of an
    extra pass over the image. ``r`` is forwarded to it.
    """
    from PIL import Image as PILImage

    if refine:
        # PIL in, PIL out.
        image = cast("Image", refine_foreground(image, alpha, r=r))
    if isinstance(alpha, torch.Tensor):
        arr = (alpha.detach().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        alpha = PILImage.fromarray(arr, mode="L")
    if alpha.size != image.size:
        alpha = alpha.resize(image.size, PILImage.Resampling.BILINEAR)
    cutout = image.convert("RGBA")
    cutout.putalpha(alpha)
    return cutout


def _box_blur(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Mean blur over a ``kernel_size`` square window, equivalent to ``cv2.blur``.

    Runs as two 1-D average pools rather than one ``kernel_size ** 2`` pool. A
    box filter is separable and replicate padding is a per-axis index clamp, so
    the result is identical to the 2-D pool while costing ``O(kernel_size)`` per
    pixel instead of ``O(kernel_size ** 2)`` -- an order of magnitude at the
    default ``r=90``.

    Padding is split ``kernel_size // 2`` before the pixel and the remainder
    after, matching OpenCV's anchor; an even kernel is otherwise offset by a
    pixel relative to the ``cv2.blur`` implementation this replaces.
    """
    import torch.nn.functional as F

    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    # `replicate` padding cannot exceed the extent it replicates from, and a
    # window wider than the image averages the whole axis either way.
    kernel_size = min(kernel_size, *x.shape[-2:])
    if kernel_size == 1:
        return x
    lo = kernel_size // 2
    hi = kernel_size - 1 - lo
    x = F.pad(x, (lo, hi, 0, 0), mode="replicate")
    x = F.avg_pool2d(x, (1, kernel_size), stride=1)
    x = F.pad(x, (0, 0, lo, hi), mode="replicate")
    return F.avg_pool2d(x, (kernel_size, 1), stride=1)


def _blur_fusion_foreground_estimator(
    image: torch.Tensor,
    foreground: torch.Tensor,
    background: torch.Tensor,
    alpha: torch.Tensor,
    r: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One blur-fusion pass. All inputs are ``(B, C, H, W)`` float in ``[0, 1]``."""
    blurred_alpha = _box_blur(alpha, r)
    blurred_fg = _box_blur(foreground * alpha, r) / (blurred_alpha + 1e-5)
    blurred_bg = _box_blur(background * (1 - alpha), r) / (1 - blurred_alpha + 1e-5)
    estimate = blurred_fg + alpha * (
        image - alpha * blurred_fg - (1 - alpha) * blurred_bg
    )
    return estimate.clamp(0, 1), blurred_bg


# shared by every nobg image processor's `refine_foreground` staticmethod
def refine_foreground(
    image: Union[torch.Tensor, "Image"],
    alpha: Union[torch.Tensor, "Image"],
    r: int = 90,
) -> Union[torch.Tensor, "Image"]:
    """Estimate unmixed foreground colors for a matte, killing halo fringing.

    A soft matte leaves background color mixed into every partially transparent
    pixel, so compositing the *original* pixels onto a new background shows a
    halo of the old one. This solves for the foreground color those pixels would
    have had, via the blur-fusion estimator of `Approximate Fast Foreground
    Colour Estimation <https://arxiv.org/abs/2006.14970>`_ (two passes, coarse
    then fine).

    Pure torch, so it runs wherever the input tensors live -- passing CUDA
    tensors keeps the whole thing on the GPU.

    Args:
        image: The original image, either a PIL image or a float tensor in
            ``[0, 1]`` of shape ``(3, H, W)`` or ``(B, 3, H, W)``.
        alpha: The matte, either a PIL image or a tensor in ``[0, 1]`` of shape
            ``(H, W)``, ``(1, H, W)`` or ``(B, 1, H, W)``. Resized to the image
            if the two disagree.
        r: Radius of the coarse blur. Larger reaches further for a color to
            borrow, at a roughly linear cost.

    Returns:
        The estimated foreground, matching ``image``'s type, shape and dtype: a
        PIL image in, a PIL image out.
    """
    from PIL.Image import Image as PILImageType

    return_pil = isinstance(image, PILImageType)
    if return_pil:
        image = _pil_to_tensor(image.convert("RGB"))
    if isinstance(alpha, PILImageType):
        alpha = _pil_to_tensor(alpha.convert("L"))

    if not isinstance(image, torch.Tensor) or not isinstance(alpha, torch.Tensor):
        raise TypeError("image and alpha must each be a torch.Tensor or a PIL image")
    if image.ndim not in (3, 4):
        raise ValueError(
            f"image must be (3, H, W) or (B, 3, H, W), got shape {tuple(image.shape)}"
        )

    batched = image.ndim == 4
    img = image if batched else image.unsqueeze(0)
    # (H, W) -> (1, 1, H, W), (C, H, W) -> (1, C, H, W).
    a = alpha
    while a.ndim < 4:
        a = a.unsqueeze(0)
    if a.shape[0] != img.shape[0] and a.shape[0] == 1:
        a = a.expand(img.shape[0], *a.shape[1:])

    input_dtype = img.dtype
    # Divisions below overflow in half precision, which shows up as black
    # patches in the refined output, so estimate in float32 regardless.
    compute_dtype = img.dtype if img.dtype == torch.float64 else torch.float32
    img = img.to(compute_dtype)
    a = a.to(device=img.device, dtype=compute_dtype)

    if a.shape[-2:] != img.shape[-2:]:
        import torch.nn.functional as F

        a = F.interpolate(a, size=img.shape[-2:], mode="bilinear", align_corners=False)

    foreground, background = _blur_fusion_foreground_estimator(img, img, img, a, r)
    # Second pass refines the estimate over a small window; `r=6` is the
    # constant from the reference implementation.
    refined = _blur_fusion_foreground_estimator(img, foreground, background, a, r=6)[0]

    refined = refined.to(input_dtype)
    if not batched:
        refined = refined[0]
    if return_pil:
        return _tensor_to_pil(refined)
    return refined


def _pil_to_tensor(image: "Image") -> torch.Tensor:
    """PIL image -> float tensor in ``[0, 1]``, shape ``(C, H, W)``."""
    from torchvision.transforms.functional import pil_to_tensor

    return pil_to_tensor(image).to(torch.float32).div_(255)


def _tensor_to_pil(image: torch.Tensor) -> "Image":
    """Float tensor in ``[0, 1]``, shape ``(C, H, W)`` -> PIL image."""
    from torchvision.transforms.functional import to_pil_image

    return to_pil_image(image.detach().float().clamp(0, 1).cpu())


# shared by every nobg model's `predict` method
def predict(
    model,
    processor,
    image,
    prompt=None,
    boxes=None,
    *,
    forward_keys: tuple[str, ...],
    processor_kwargs: dict | None = None,
    forward_kwargs: dict | None = None,
    batch_size: int = 1,
    return_type: str = "cutout",
):
    """Run the full preprocess → forward → post-process → composite pipeline.

    ``processor`` comes first: it is the fixed piece of the call, and the inputs
    that vary follow it in decreasing order of how often they are supplied —
    ``image``, then ``prompt``, then ``boxes``.

    ``image`` is anything ``loadimg.load_img`` accepts (a path, URL, base64
    string, numpy array or PIL image), or a list of them. A single image returns
    a single result; a list returns a list.

    ``prompt`` is passed to the processor as ``text=`` — transformers' convention
    for a text input — and is only meaningful for models whose processor takes
    one. It applies to every image in the call.

    ``boxes`` is passed as ``input_boxes=``, and unlike ``prompt`` it is
    **per-image**: a list of ``[x1, y1, x2, y2]`` boxes for a single image, or a
    list of those, one entry per image. It is sliced alongside the images so
    ``batch_size`` cannot misalign it.

    ``forward_keys`` names the batch entries this model's ``forward`` accepts, so
    extras the processor emits (SAM3's ``original_sizes``) are dropped rather than
    forwarded.
    """
    from loadimg import load_img

    if return_type not in ("cutout", "alpha"):
        raise ValueError(
            f"return_type must be 'cutout' or 'alpha', got {return_type!r}"
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    # `BatchFeature`/`BatchEncoding` are UserDicts, so check the ABC, not `dict`.
    if isinstance(image, Mapping):
        raise TypeError(
            "predict() takes raw images, not an already-preprocessed batch. Either "
            "pass the images themselves, or call the model directly with **inputs "
            "and post-process via the processor."
        )

    processor_kwargs = dict(processor_kwargs or {})
    if prompt is not None:
        processor_kwargs["text"] = prompt

    single = not isinstance(image, list | tuple)
    items = [image] if single else list(image)
    if not items:
        return []
    images = [load_img(item).convert("RGB") for item in items]

    # Boxes are per-image, so normalize to one entry per image up front and slice
    # them with each chunk below. A bare list of boxes is taken as this call's
    # single image; nesting one level deeper gives one box list per image.
    per_image_boxes = None
    if boxes is not None:
        if not isinstance(boxes, list | tuple) or not boxes:
            raise TypeError(
                "boxes must be a non-empty list of [x1, y1, x2, y2] boxes, or a "
                f"list of those (one per image), got {boxes!r}"
            )
        depth = 0
        probe = boxes
        while isinstance(probe, list | tuple) and probe:
            depth += 1
            probe = probe[0]
        if depth == 2:
            per_image_boxes = [list(boxes)]
        elif depth == 3:
            per_image_boxes = [list(entry) for entry in boxes]
        else:
            raise ValueError(
                "boxes must be nested [box level, coordinates] for one image or "
                f"[image level, box level, coordinates] for several, got {depth} "
                "levels of nesting"
            )
        if len(per_image_boxes) != len(images):
            raise ValueError(
                f"Got boxes for {len(per_image_boxes)} images but {len(images)} "
                "image(s) to predict on"
            )

    # Match the model's placement, and its dtype for half-precision models.
    param = next((p for p in model.parameters() if p.is_floating_point()), None)
    device = param.device if param is not None else torch.device("cpu")
    dtype = param.dtype if param is not None else torch.float32

    was_training = model.training
    model.eval()
    results = []
    try:
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                chunk = images[start : start + batch_size]
                chunk_kwargs = dict(processor_kwargs)
                if per_image_boxes is not None:
                    chunk_kwargs["input_boxes"] = per_image_boxes[
                        start : start + batch_size
                    ]
                batch = processor(images=chunk, return_tensors="pt", **chunk_kwargs)
                forward_inputs = {}
                for key in forward_keys:
                    value = batch.get(key)
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor):
                        value = (
                            value.to(device=device, dtype=dtype)
                            if value.is_floating_point()
                            else value.to(device)
                        )
                    forward_inputs[key] = value
                # A single text prompt for N images comes back as batch 1 while
                # `pixel_values` is batch N, which fails inside attention. Repeat
                # the shared prompt so every entry has the image batch size. Boxes
                # are excluded: they are per-image, already sliced per chunk, so a
                # batch-1 box tensor means one image, never a value to broadcast.
                size = len(forward_inputs["pixel_values"])
                shared = {"pixel_values", "input_boxes", "input_boxes_labels"}
                for key, value in forward_inputs.items():
                    if (
                        key not in shared
                        and isinstance(value, torch.Tensor)
                        and value.shape[0] == 1
                        and size > 1
                    ):
                        forward_inputs[key] = value.expand(
                            size, *value.shape[1:]
                        ).contiguous()
                outputs = model(**forward_inputs, **(forward_kwargs or {}))
                mattes = post_process_alpha_matting(
                    outputs, target_sizes=[(im.height, im.width) for im in chunk]
                )
                if return_type == "alpha":
                    results.extend(matte.float().cpu() for matte in mattes)
                else:
                    results.extend(
                        cutout(im, matte.float().cpu())
                        for im, matte in zip(chunk, mattes, strict=True)
                    )
    finally:
        if was_training:
            model.train()

    return results[0] if single else results


# turn raw alpha-matte logits into per-image mattes in [0, 1]
def post_process_alpha_matting(
    outputs,
    target_sizes: list[tuple[int, int]] | None = None,
) -> list[torch.Tensor]:
    """Convert raw alpha-matte logits into per-image alpha mattes in ``[0, 1]``.

    Args:
        outputs: The model output; a dict with a ``"logits"`` key or an object
            exposing ``.logits`` of shape ``(B, 1, H, W)``.
        target_sizes: Optional list of ``(height, width)`` tuples, one per image;
            each matte is bilinearly resized to its target size.

    Returns:
        A list of ``(H, W)`` tensors with values in ``[0, 1]``.
    """
    import torch.nn.functional as F

    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    if target_sizes is not None and len(logits) != len(target_sizes):
        raise ValueError(
            f"Got {len(target_sizes)} target sizes for a batch of {len(logits)} images"
        )
    # Sigmoid before resizing, matching the eval/benchmark scripts.
    probs = logits.sigmoid()
    mattes = []
    for idx in range(len(probs)):
        alpha = probs[idx].unsqueeze(0)  # (1, 1, H, W)
        if target_sizes is not None:
            alpha = F.interpolate(
                alpha, size=target_sizes[idx], mode="bilinear", align_corners=False
            )
        mattes.append(alpha[0, 0])
    return mattes


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
from nobg import AutoModel, AutoProcessor
model = AutoModel.from_pretrained("{{{{ repo_id | default("{default_repo}", true) }}}}")
processor = AutoProcessor.from_pretrained("{{{{ repo_id | default("{default_repo}", true) }}}}")
model.predict(processor, image)
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
