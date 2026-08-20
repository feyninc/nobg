import logging
import os
from dataclasses import dataclass, field, fields
from importlib.metadata import PackageNotFoundError, version

import torch
import torch.nn.functional as F
from torch import nn
from transformers import CLIPTextConfig
from transformers.models.sam3.configuration_sam3 import (
    Sam3Config as HFSam3Config,
)
from transformers.models.sam3.configuration_sam3 import (
    Sam3DETRDecoderConfig,
    Sam3DETREncoderConfig,
    Sam3GeometryEncoderConfig,
    Sam3MaskDecoderConfig,
    Sam3VisionConfig,
    Sam3ViTConfig,
)
from transformers.models.sam3.image_processing_sam3 import Sam3ImageProcessor
from transformers.models.sam3.modeling_sam3 import Sam3Model

from ..loss import sam3_loss
from ..mixin import Revised_Mixin
from ..utils import model_card_template, predict
from .image_processing_sam3 import Sam3Processor

logger = logging.getLogger(__name__)

# Fallback tokenizer for `default_processor`. SAM3's text tower *is* CLIP's, so
# CLIP's BPE vocabulary (49408 entries, matching `text_vocab_size`) is the right
# one. `openai/clip-vit-large-patch14` is preferred over `facebook/sam3` because
# it is ungated: the tokenizer is loadable without accepting a license.
DEFAULT_TOKENIZER = "openai/clip-vit-large-patch14"

try:
    NOBG_VERSION = version("nobg")
except PackageNotFoundError:  # running from source without an installed dist
    NOBG_VERSION = "0.0.0+unknown"

SAM3_CITATION = """@article{sam3,
  title={SAM 3: Segment Anything with Concepts},
  author={Carion, Nicolas and Gustafson, Laura and Hu, Yuan-Ting and Debnath, Shoubhik and Hu, Ronghang and Suris, Didac and Ryali, Chaitanya and Alwala, Kalyan Vasudev and Khedr, Haitham and Huang, Andrew and Lei, Jie and Ma, Tengyu and Guo, Baishan and Marks, Markus and Greer, Joseph and Wang, Meng and Sun, Peize and R{\\"a}dle, Roman and Afouras, Triantafyllos and Mavroudi, Effrosyni and Dollar, Piotr and Ravi, Nikhila and Saenko, Kate and Zhang, Pengchuan and Feichtenhofer, Christoph},
  journal={arXiv preprint},
  year={2025},
  url={https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/},
}"""


@dataclass
class Sam3Config:
    """Configuration for the nobg SAM3 matting wrapper.

    SAM3 is natively a promptable, open-vocabulary detector: it needs a text
    prompt and emits one mask per object query. ``default_prompt``,
    ``score_threshold`` and ``aggregate`` are the nobg-specific fields that turn
    that into a single background-removal alpha matte; the remaining fields are
    forwarded to the ``transformers`` SAM3 sub-configs.

    ``aggregate`` selects which SAM3 head produces the matte:

    - ``"semantic"`` (default) uses ``semantic_seg``, SAM3's own prompt-conditioned
      semantic head. Measured on ``facebook/sam3`` against FeyNobg on two photos:
      MAE 0.035/0.039 and IoU 0.976/0.983, with 19/29 % of pixels at intermediate
      alpha.
    - ``"max"`` / ``"mean"`` union the per-query instance masks (see
      ``Sam3._aggregate``). Same subject, but far mushier alpha — MAE 0.144/0.150
      and 71/79 % soft pixels on the same two photos, because a query's mask logits
      are calibrated for *binarizing* at 0.5, not for use as a matte. Use these
      when you want the matte to follow ``score_threshold`` and the detected
      instance set; ``score_threshold`` has no effect under ``"semantic"``.
    """

    image_size: int = 1008
    default_prompt: str = "the main foreground subject"
    score_threshold: float = 0.3
    aggregate: str = "semantic"
    # vision tower (Sam3ViTConfig / Sam3VisionConfig)
    vision_hidden_size: int = 1024
    vision_intermediate_size: int = 4736
    vision_num_hidden_layers: int = 32
    vision_num_attention_heads: int = 16
    vision_patch_size: int = 14
    vision_window_size: int = 24
    vision_global_attn_indexes: list = field(default_factory=lambda: [7, 15, 23, 31])
    vision_pretrain_image_size: int = 336
    fpn_hidden_size: int = 256
    # detector stack (geometry encoder / DETR encoder+decoder / mask decoder)
    hidden_size: int = 256
    intermediate_size: int = 2048
    num_attention_heads: int = 8
    geometry_num_layers: int = 3
    detr_encoder_num_layers: int = 6
    detr_decoder_num_layers: int = 6
    num_queries: int = 200
    num_upsampling_stages: int = 3
    # text tower (CLIPTextConfig)
    text_vocab_size: int = 49408
    text_hidden_size: int = 1024
    text_intermediate_size: int = 4096
    text_projection_dim: int = 512
    text_num_hidden_layers: int = 24
    text_num_attention_heads: int = 16
    text_max_position_embeddings: int = 32
    nobg_version: str = NOBG_VERSION

    def __post_init__(self):
        if self.aggregate not in ("semantic", "max", "mean"):
            raise ValueError(
                f"aggregate must be 'semantic', 'max' or 'mean', got {self.aggregate!r}"
            )
        if self.image_size % self.vision_patch_size != 0:
            raise ValueError(
                f"image_size={self.image_size} is not divisible by "
                f"vision_patch_size={self.vision_patch_size}"
            )


class Sam3(
    nn.Module,
    Revised_Mixin,
    library_name="nobg",
    repo_url="https://github.com/feyninc/nobg",
    paper_url="https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/",
    license="apache-2.0",
    tags=["nobg", "nobg-sam3"],
    model_card_template=model_card_template(
        class_name="Sam3",
        default_repo="nobg/sam3",
        citation=SAM3_CITATION,
    ),
):
    """SAM3 adapted to nobg's background-removal contract.

    Wraps ``transformers``' ``Sam3Model`` (the Apache-2.0 reimplementation) and
    collapses its per-query instance masks into a single alpha matte, so the
    output matches ``BiRefNet``: ``logits`` of shape ``(B, 1, H, W)``, raw (not
    sigmoided). The instance-level outputs are passed through untouched, so the
    open-vocabulary detector is still fully usable.
    """

    def __init__(self, config: Sam3Config | None = None):
        super().__init__()
        self.config = config or Sam3Config()

        grid = self.config.image_size // self.config.vision_patch_size
        vit_config = Sam3ViTConfig(
            hidden_size=self.config.vision_hidden_size,
            intermediate_size=self.config.vision_intermediate_size,
            num_hidden_layers=self.config.vision_num_hidden_layers,
            num_attention_heads=self.config.vision_num_attention_heads,
            image_size=self.config.image_size,
            patch_size=self.config.vision_patch_size,
            window_size=self.config.vision_window_size,
            global_attn_indexes=list(self.config.vision_global_attn_indexes),
            pretrain_image_size=self.config.vision_pretrain_image_size,
        )
        # `scale_factors` are relative to the ViT grid; the last level is dropped
        # by Sam3Model.forward, which consumes fpn_hidden_states[:-1].
        vision_config = Sam3VisionConfig(
            backbone_config=vit_config,
            fpn_hidden_size=self.config.fpn_hidden_size,
            backbone_feature_sizes=[
                [int(grid * s), int(grid * s)] for s in (4.0, 2.0, 1.0)
            ],
        )
        text_config = CLIPTextConfig(
            vocab_size=self.config.text_vocab_size,
            hidden_size=self.config.text_hidden_size,
            intermediate_size=self.config.text_intermediate_size,
            projection_dim=self.config.text_projection_dim,
            num_hidden_layers=self.config.text_num_hidden_layers,
            num_attention_heads=self.config.text_num_attention_heads,
            max_position_embeddings=self.config.text_max_position_embeddings,
            hidden_act="gelu",
        )
        hf_config = HFSam3Config(
            vision_config=vision_config,
            text_config=text_config,
            geometry_encoder_config=Sam3GeometryEncoderConfig(
                hidden_size=self.config.hidden_size,
                num_layers=self.config.geometry_num_layers,
                num_attention_heads=self.config.num_attention_heads,
                intermediate_size=self.config.intermediate_size,
            ),
            detr_encoder_config=Sam3DETREncoderConfig(
                hidden_size=self.config.hidden_size,
                num_layers=self.config.detr_encoder_num_layers,
                num_attention_heads=self.config.num_attention_heads,
                intermediate_size=self.config.intermediate_size,
            ),
            detr_decoder_config=Sam3DETRDecoderConfig(
                hidden_size=self.config.hidden_size,
                num_layers=self.config.detr_decoder_num_layers,
                num_queries=self.config.num_queries,
                num_attention_heads=self.config.num_attention_heads,
                intermediate_size=self.config.intermediate_size,
            ),
            mask_decoder_config=Sam3MaskDecoderConfig(
                hidden_size=self.config.hidden_size,
                num_upsampling_stages=self.config.num_upsampling_stages,
                num_attention_heads=self.config.num_attention_heads,
            ),
        )
        self.model = Sam3Model(hf_config)

        # Loss used when `labels` is passed to forward(). SAM 3's own
        # semantic-segmentation objective (weighted focal + dice) rather than
        # BiRefNet's BCE+IoU+SSIM, so fine-tuning matches how the checkpoint was
        # trained. Plain function attribute (not a submodule), so it never enters
        # the state dict and can be hot-swapped: `model.criterion = my_loss`.
        self.criterion = sam3_loss

        # Filled in by `from_origin`/`process`: where a tokenizer for `process` can
        # be found, and the processor built from it. Plain attributes, so neither
        # enters the state dict.
        self._tokenizer_source: str | None = None
        self._cached_processor: Sam3Processor | None = None

    def _aggregate(
        self,
        pred_masks: torch.Tensor,
        pred_logits: torch.Tensor,
        presence_logits: torch.Tensor | None,
        threshold: float,
    ) -> torch.Tensor:
        """Collapse per-query mask logits into a single ``(B, 1, h, w)`` matte.

        Used for ``aggregate="max"``/``"mean"``. The default is ``"semantic"``,
        which reads ``semantic_seg`` directly instead — per-query mask logits are
        trained to be *binarized* at 0.5, so as soft alpha they come out much
        mushier (see ``Sam3Config`` for the measured gap).

        Queries are scored the way SAM3's own instance post-processing scores
        them (``pred_logits * presence``), then the surviving mask logits are
        combined. Aggregation stays in logit space: ``sigmoid`` is monotonic, so
        a max over logits is exactly the logit of the max probability, and the
        raw-logits output contract is preserved.
        """
        scores = pred_logits.sigmoid()
        if presence_logits is not None:
            scores = scores * presence_logits.sigmoid()

        keep = scores > threshold
        # Never emit an empty matte: if nothing clears the threshold, fall back
        # to the single best-scoring query.
        best = scores.argmax(dim=1, keepdim=True)
        keep = keep | torch.zeros_like(keep).scatter_(1, best, True)

        mask = keep[..., None, None]
        if self.config.aggregate == "mean":
            kept = (pred_masks * mask).sum(dim=1)
            matte = kept / mask.sum(dim=1).clamp(min=1)
        else:
            neg_inf = torch.finfo(pred_masks.dtype).min
            matte = pred_masks.masked_fill(~mask, neg_inf).max(dim=1).values
        return matte.unsqueeze(1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_boxes: torch.Tensor | None = None,
        input_boxes_labels: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        score_threshold: float | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        """Run SAM3 and return a nobg-style alpha-matte prediction.

        Args:
            pixel_values: ``(B, 3, H, W)`` preprocessed images. ``H`` and ``W``
                must both equal ``config.image_size`` — the vision tower's
                rotary tables are fixed-size buffers built from it.
            input_ids: Tokenized text prompt. SAM3 requires a prompt; pass the
                one produced by ``Sam3Processor`` (which defaults to
                ``config.default_prompt``).
            attention_mask: Attention mask for ``input_ids``.
            input_boxes: Optional ``(B, num_boxes, 4)`` visual prompt in SAM3's
                normalized ``cxcywh`` form — pass the tensor ``Sam3Processor``
                produces from pixel-space ``input_boxes``, not raw coordinates.
            input_boxes_labels: Optional ``(B, num_boxes)`` labels for
                ``input_boxes``; the processor generates them when omitted.
            labels: Optional ``(B, 1, H, W)`` binary masks; when given, ``loss``
                is added to the output.
            score_threshold: Overrides ``config.score_threshold`` for this call.
                Only used when ``config.aggregate`` is ``"max"``/``"mean"``;
                the default ``"semantic"`` head has no per-query scores.

        Returns:
            A dict with ``logits`` ``(B, 1, H, W)`` raw alpha-matte logits, the
            passthrough instance outputs (``pred_masks``, ``pred_boxes``,
            ``pred_logits``, ``presence_logits``, ``semantic_seg``) and ``loss``
            when ``labels`` was provided.
        """
        if input_ids is None:
            raise ValueError(
                "SAM3 is prompt-driven: `input_ids` is required. Use "
                "Sam3Processor, which supplies config.default_prompt by default."
            )
        height, width = pixel_values.shape[-2:]
        if (height, width) != (self.config.image_size, self.config.image_size):
            # The ViT registers its rotary tables as fixed-size buffers sized from
            # config.image_size, so a mismatch would fail deep inside attention.
            raise ValueError(
                f"pixel_values is {height}x{width} but this model is built for "
                f"{self.config.image_size}x{self.config.image_size}. Preprocess "
                "with Sam3Processor, or rebuild with "
                f"Sam3Config(image_size={height})."
            )

        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            **kwargs,
        )

        if self.config.aggregate == "semantic":
            # SAM3's own prompt-conditioned semantic head. Sharper and far better
            # calibrated as an alpha matte than the union of instance masks; see
            # Sam3Config for the measured comparison. Sam3MaskDecoder always emits
            # it, but the field is typed Optional upstream, so fail loudly rather
            # than passing None into interpolate.
            matte = outputs.semantic_seg
            if matte is None:
                raise RuntimeError(
                    "SAM3 returned no semantic_seg; use "
                    "Sam3Config(aggregate='max') to build the matte from the "
                    "instance masks instead."
                )
        else:
            threshold = (
                self.config.score_threshold
                if score_threshold is None
                else score_threshold
            )
            matte = self._aggregate(
                outputs.pred_masks,
                outputs.pred_logits,
                outputs.presence_logits,
                threshold,
            )
        logits = F.interpolate(
            matte, size=(height, width), mode="bilinear", align_corners=False
        )

        result: dict[str, torch.Tensor | None] = {
            "logits": logits,
            "pred_masks": outputs.pred_masks,
            "pred_boxes": outputs.pred_boxes,
            "pred_logits": outputs.pred_logits,
            "presence_logits": outputs.presence_logits,
            "semantic_seg": outputs.semantic_seg,
        }
        if labels is not None:
            result["loss"] = self.criterion([logits], labels)
        return result

    def predict(
        self,
        processor,
        image,
        prompt: str | list[str] | None = None,
        boxes: list | None = None,
        *,
        score_threshold: float | None = None,
        batch_size: int = 1,
        return_type: str = "cutout",
        **processor_kwargs,
    ):
        """Cut out one or more images, end to end.

        Wraps the preprocess → forward → post-process → composite sequence in a
        single call, running under ``no_grad`` in eval mode on the model's own
        device and dtype.

        The positional order is the fixed argument first, then the varying ones by
        how often they are supplied: ``predict(processor, image, prompt, boxes)``.

        Args:
            processor: A ``Sam3Processor``.
            image: Anything ``loadimg.load_img`` accepts — a path, URL, base64
                string, numpy array or PIL image — or a list of them.
            prompt: Optional text prompt, applied to every image in the call.
                Omit it for prompt-free background removal (the processor
                supplies its ``default_prompt``); pass one for an open-vocabulary
                cutout, e.g. ``predict(processor, image, "the dog")``.
            boxes: Optional visual prompt — ``[[x1, y1, x2, y2], ...]`` in the
                original image's pixel coordinates for a single image, or one
                such list per image. Unlike ``prompt`` these are **per-image**.
                With boxes and no ``prompt``, the processor's own default text is
                ``"visual"``, i.e. segment what the boxes point at.
            score_threshold: Overrides ``config.score_threshold`` for this call.
                Only used when ``config.aggregate`` is ``"max"``/``"mean"``;
                the default ``"semantic"`` head has no per-query scores.
            batch_size: Number of images per forward pass. The default of 1 keeps
                peak memory flat; raise it for throughput.
            return_type: ``"cutout"`` for RGBA images, ``"alpha"`` for the raw
                ``(H, W)`` mattes in ``[0, 1]``.
            **processor_kwargs: Forwarded to the processor.

        Returns:
            A single result for a single image, or a list for a list of images.
            Either RGBA ``PIL.Image``s or ``(H, W)`` matte tensors at each image's
            original resolution, per ``return_type``.

        Note:
            The matte is never empty, so a prompt for something absent from the
            image still returns one. Call ``forward`` directly and read
            ``presence_logits`` when you need to know whether the concept is
            actually present.
        """
        return predict(
            self,
            processor,
            image,
            prompt,
            boxes,
            forward_keys=(
                "pixel_values",
                "input_ids",
                "attention_mask",
                "input_boxes",
                "input_boxes_labels",
            ),
            processor_kwargs=processor_kwargs,
            forward_kwargs=(
                {} if score_threshold is None else {"score_threshold": score_threshold}
            ),
            batch_size=batch_size,
            return_type=return_type,
        )

    def default_processor(self, tokenizer=None) -> Sam3Processor:
        """Build the processor this model's own config implies.

        The image side follows ``config.image_size`` and the prompt-free default
        follows ``config.default_prompt``, but SAM3 also needs a text tokenizer,
        which no ``config.json`` describes. It is resolved in order: the
        ``tokenizer`` argument, the origin ``from_origin`` loaded this model from,
        then ``DEFAULT_TOKENIZER``.

        Args:
            tokenizer: A tokenizer instance, or a repo id / local directory to load
                one from. ``None`` resolves as described above.

        Returns:
            A ``Sam3Processor``. The no-argument result is cached on the model, so
            repeated ``process`` calls don't re-load the tokenizer.

        Raises:
            OSError: If no tokenizer could be loaded from any candidate source.
            ValueError: If the resolved tokenizer's vocabulary is larger than
                ``config.text_vocab_size``, i.e. it can emit token ids the text
                embedding has no row for.
        """
        if tokenizer is None and self._cached_processor is not None:
            return self._cached_processor

        if tokenizer is None or isinstance(tokenizer, (str, os.PathLike)):
            from transformers import AutoTokenizer

            if tokenizer is not None:
                sources = [str(tokenizer)]
            else:
                candidates = (self._tokenizer_source, DEFAULT_TOKENIZER)
                sources = list(dict.fromkeys(s for s in candidates if s))
            tok, last_error = None, None
            for source in sources:
                try:
                    tok = AutoTokenizer.from_pretrained(source)
                    break
                except OSError as err:  # missing, gated or offline
                    logger.info("no tokenizer at %r (%s)", source, err)
                    last_error = err
            if tok is None:
                raise OSError(
                    f"could not load a tokenizer from any of {sources}; pass "
                    "`tokenizer=` a repo id, a local directory or an instance"
                ) from last_error
        else:
            tok = tokenizer

        if tok.vocab_size > self.config.text_vocab_size:
            raise ValueError(
                f"tokenizer vocabulary ({tok.vocab_size}) is larger than "
                f"config.text_vocab_size ({self.config.text_vocab_size}); it can "
                "emit token ids the text embedding has no row for"
            )

        size = {"height": self.config.image_size, "width": self.config.image_size}
        processor = Sam3Processor(
            Sam3ImageProcessor(size=size),
            tok,
            default_prompt=self.config.default_prompt,
        )
        if tokenizer is None:
            self._cached_processor = processor
        return processor

    def process(
        self,
        image,
        prompt: str | list[str] | None = None,
        boxes: list | None = None,
        *,
        tokenizer=None,
        score_threshold: float | None = None,
        batch_size: int = 1,
        return_type: str = "cutout",
        **processor_kwargs,
    ):
        """Cut out one or more images, without a processor.

        ``predict`` without the ``processor`` argument: it builds the one
        ``default_processor`` describes and forwards everything else unchanged, so
        a cutout — prompted or not — is a single call on a freshly loaded model.

        Args:
            image: Anything ``loadimg.load_img`` accepts — a path, URL, base64
                string, numpy array or PIL image — or a list of them.
            prompt: Optional text prompt, applied to every image in the call.
            boxes: Optional per-image visual prompt, in the original image's pixel
                coordinates.
            tokenizer: Passed to ``default_processor``; only needed when the
                tokenizer cannot be resolved automatically.
            score_threshold: Overrides ``config.score_threshold`` for this call.
            batch_size: Number of images per forward pass. The default of 1 keeps
                peak memory flat; raise it for throughput.
            return_type: ``"cutout"`` for RGBA images, ``"alpha"`` for the raw
                ``(H, W)`` mattes in ``[0, 1]``.
            **processor_kwargs: Forwarded to the processor.

        Returns:
            Exactly what ``predict`` returns for the same inputs.

        Note:
            Pass a processor to ``predict`` instead when you have one already, or
            when a checkpoint's ``preprocessor_config.json`` differs from its
            ``config.image_size`` — this method trusts the model config.
        """
        return self.predict(
            self.default_processor(tokenizer),
            image,
            prompt,
            boxes,
            score_threshold=score_threshold,
            batch_size=batch_size,
            return_type=return_type,
            **processor_kwargs,
        )

    # The TorchScript exporter, not torch.export: SAM3's decoder sizes tensors
    # from data, which torch.export refuses to guard on
    # (`GuardOnDataDependentSymNode: Eq(u0*u1, 64)`). Nothing is lost — the
    # traced graph matches torch to ~5e-7 for any prompt, not just the traced
    # one. BiRefNet needs the opposite; see `Onnx_Mixin.onnx_dynamo`.
    onnx_dynamo = False

    def onnx_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        """Add the tokenized prompt ``forward`` requires to the image input.

        SAM3 will not run without ``input_ids``, so the graph has to be traced
        with one. Only its shape is baked in, and ``Sam3Processor`` always pads
        text to 32 tokens, so a synthetic prompt of that length gives a graph
        every real prompt fits.

        ``input_boxes`` is left out: the visual-prompt path is optional, and
        including it would make boxes mandatory for every call. Pass
        ``dummy_inputs=`` explicitly to export that variant instead.
        """
        inputs = super().onnx_dummy_inputs(batch_size)
        length = self.config.text_max_position_embeddings
        # CLIP's specials are the last two ids of its vocabulary (49406/49407 of
        # 49408), and its pad id is 0.
        input_ids = torch.zeros(batch_size, length, dtype=torch.long)
        input_ids[:, 0] = self.config.text_vocab_size - 2  # <|startoftext|>
        input_ids[:, 1] = self.config.text_vocab_size - 1  # <|endoftext|>
        attention_mask = torch.zeros(batch_size, length, dtype=torch.long)
        attention_mask[:, :2] = 1
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask
        return inputs

    @classmethod
    def from_origin(
        cls,
        origin: str | os.PathLike | nn.Module,
        config: Sam3Config | None = None,
        *,
        token: str | None = None,
        **overrides,
    ) -> "Sam3":
        """Build a (possibly re-parameterized) Sam3 from a previous model.

        ``origin`` is a Hub repo id, a local directory containing
        ``config.json`` + ``model.safetensors``, or an ``nn.Module`` instance.
        Every weight whose key and shape still match is injected; anything new
        or reshaped keeps its fresh initialization.

        This is also the way to load an upstream ``transformers`` SAM3 release
        such as ``facebook/sam3``, whose layout differs twice over: the config is
        a ``sam3_video`` composite (so nobg's fields are read out of the nested
        ``detector_config``) and the weights are keyed ``detector_model.*``
        alongside video-tracker weights this detector-only wrapper never builds
        (so they are remapped and the tracker dropped).
        """
        if isinstance(origin, nn.Module):
            state_dict = origin.state_dict()
            # A live model has no tokenizer of its own; carry over whatever it was
            # itself loaded from, so `process` keeps working across a re-parameterization.
            tokenizer_source = getattr(origin, "_tokenizer_source", None)
            origin_config = getattr(origin, "config", None)
            origin_fields = (
                {f.name: getattr(origin_config, f.name) for f in fields(origin_config)}
                if origin_config is not None
                and hasattr(origin_config, "__dataclass_fields__")
                else {}
            )
        else:
            import json

            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file

            origin = str(origin)
            # Wherever the weights came from is also the best guess for a tokenizer,
            # which `default_processor` needs and no config.json records.
            tokenizer_source = origin
            if os.path.isdir(origin):
                config_path = os.path.join(origin, "config.json")
                weights_path = os.path.join(origin, "model.safetensors")
            else:
                config_path = hf_hub_download(origin, "config.json", token=token)
                weights_path = hf_hub_download(origin, "model.safetensors", token=token)
            with open(config_path) as f:
                raw = json.load(f)
            # A nobg config.json is the flat dataclass; a transformers one is
            # nested and always carries `model_type` (`sam3` or `sam3_video`).
            origin_fields = _fields_from_hf_config(raw) if "model_type" in raw else raw
            state_dict = load_file(weights_path)
        is_upstream = any(k.startswith("detector_model.") for k in state_dict)

        if config is None:
            known = {f.name for f in fields(Sam3Config)}
            merged = {k: v for k, v in origin_fields.items() if k in known}
            merged.update(overrides)
            merged["nobg_version"] = overrides.get("nobg_version", NOBG_VERSION)
            config = Sam3Config(**merged)
        elif overrides:
            raise ValueError(
                "pass either an explicit `config` or field overrides, not both"
            )

        model = cls(config)
        model._tokenizer_source = tokenizer_source

        if is_upstream:
            state_dict = _remap_upstream_state_dict(state_dict)

        target = model.state_dict()
        compatible = {}
        skipped_shape = []
        for k, v in state_dict.items():
            if k in target and target[k].shape == v.shape:
                compatible[k] = v
            elif k in target:
                skipped_shape.append(k)
        missing = [k for k in target if k not in compatible]
        model.load_state_dict(compatible, strict=False)

        logger.info(
            "from_origin: injected %d/%d tensors (%d shape-mismatched, "
            "%d fresh-initialized)%s",
            len(compatible),
            len(target),
            len(skipped_shape),
            len(missing),
            " [upstream layout remapped]" if is_upstream else "",
        )
        return model


def _remap_upstream_state_dict(state_dict: dict) -> dict:
    """Translate an upstream ``transformers`` SAM3 checkpoint to nobg's key layout.

    ``facebook/sam3`` is a ``Sam3VideoModel`` checkpoint: the image detector lives
    under ``detector_model.``, and it is followed by ``tracker_model.`` /
    ``tracker_neck.`` weights for video propagation. nobg wraps only the detector
    (as ``self.model``), so the prefix is rewritten and the tracker dropped.
    """
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("detector_model."):
            remapped[f"model.{key[len('detector_model.') :]}"] = value
    return remapped


def _fields_from_hf_config(raw: dict) -> dict:
    """Map an upstream transformers SAM3 ``config.json`` onto nobg's fields.

    Handles both a plain ``sam3`` config and the ``sam3_video`` composite that
    ``facebook/sam3`` ships (detector nested under ``detector_config``).
    """
    detector = raw.get("detector_config", raw)
    vision = detector.get("vision_config", {})
    backbone = vision.get("backbone_config", {})
    text = detector.get("text_config", {})
    geometry = detector.get("geometry_encoder_config", {})
    encoder = detector.get("detr_encoder_config", {})
    decoder = detector.get("detr_decoder_config", {})
    mask = detector.get("mask_decoder_config", {})

    out: dict = {}

    def put(name, source, key):
        if key in source:
            out[name] = source[key]

    put("image_size", backbone, "image_size")
    put("vision_hidden_size", backbone, "hidden_size")
    put("vision_intermediate_size", backbone, "intermediate_size")
    put("vision_num_hidden_layers", backbone, "num_hidden_layers")
    put("vision_num_attention_heads", backbone, "num_attention_heads")
    put("vision_patch_size", backbone, "patch_size")
    put("vision_window_size", backbone, "window_size")
    put("vision_global_attn_indexes", backbone, "global_attn_indexes")
    put("vision_pretrain_image_size", backbone, "pretrain_image_size")
    put("fpn_hidden_size", vision, "fpn_hidden_size")
    put("hidden_size", encoder, "hidden_size")
    put("intermediate_size", encoder, "intermediate_size")
    put("num_attention_heads", encoder, "num_attention_heads")
    put("geometry_num_layers", geometry, "num_layers")
    put("detr_encoder_num_layers", encoder, "num_layers")
    put("detr_decoder_num_layers", decoder, "num_layers")
    put("num_queries", decoder, "num_queries")
    put("num_upsampling_stages", mask, "num_upsampling_stages")
    put("text_vocab_size", text, "vocab_size")
    put("text_hidden_size", text, "hidden_size")
    put("text_intermediate_size", text, "intermediate_size")
    put("text_projection_dim", text, "projection_dim")
    put("text_num_hidden_layers", text, "num_hidden_layers")
    put("text_num_attention_heads", text, "num_attention_heads")
    put("text_max_position_embeddings", text, "max_position_embeddings")
    return out
