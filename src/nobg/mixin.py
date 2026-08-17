from __future__ import annotations

import copy
import inspect
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import torch
from huggingface_hub import PyTorchModelHubMixin, whoami
from torch import nn

from .utils import predict, set_doc

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

ONNX_FILE_NAME = "model.onnx"

# `DeformConv`, which BiRefNet's deformable convolutions lower to, is an
# opset-19 operator. A lower opset exports but fails to load in onnxruntime
# ("No Op registered for DeformConv"), so 19 is the floor for this library.
DEFAULT_ONNX_OPSET = 19

# Weight formats an ONNX repo never needs, skipped when downloading one.
# Everything else *is* downloaded: an export whose initializers exceed the 2 GB
# protobuf limit spills them into sidecar files that only the graph knows the
# names of (`model.onnx.data` from the torch.export exporter, one file per
# tensor from the TorchScript one).
_TORCH_WEIGHT_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.h5",
    "*.msgpack",
]

# Appended to the shared model card for an export, since the card's own "how to
# load" covers the torch checkpoint and `from_pretrained` cannot read a graph.
_ONNX_USAGE_SECTION = """
## how to load the ONNX export
```
pip install nobg[onnx]
```
```python
from nobg import {class_name}
model = {class_name}.onnx_from_pretrained("{repo_id}")
cutout = model.process("image.png")
```
This repo holds an ONNX export -- `model.onnx`, plus sidecar data files for a
large model -- traced at fixed input shapes, including the batch size.
"""

# onnxruntime reports input types as strings; map the ones nobg models take onto
# the torch dtype the session expects, so a float64 or int32 tensor coming out of
# a processor is cast rather than rejected.
_ONNX_TO_TORCH_DTYPE = {
    "tensor(float)": torch.float32,
    "tensor(float16)": torch.float16,
    "tensor(double)": torch.float64,
    "tensor(int64)": torch.int64,
    "tensor(int32)": torch.int32,
    "tensor(uint8)": torch.uint8,
    "tensor(bool)": torch.bool,
}


def _to_model_placement(
    model: nn.Module, inputs: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Move tracing inputs onto the model's own device and dtype.

    Mirrors ``utils.predict``: floating tensors follow both, integer tensors
    (token ids, box labels) follow only the device.
    """
    param = next((p for p in model.parameters() if p.is_floating_point()), None)
    if param is None:
        return inputs
    return {
        name: (
            value.to(device=param.device, dtype=param.dtype)
            if value.is_floating_point()
            else value.to(param.device)
        )
        for name, value in inputs.items()
    }


class _OnnxMatteWrapper(nn.Module):
    """Presents a nobg model to the ONNX exporter as tensors in, one tensor out.

    The exporter traces positional arguments and cannot return a dict holding a
    list, which is what every nobg ``forward`` produces (BiRefNet's
    ``intermediate_logits``, SAM3's instance heads). This maps the traced
    positional arguments back onto keyword names and keeps ``logits`` alone, so
    the graph is exactly the alpha-matte computation.
    """

    def __init__(self, model: nn.Module, input_names: tuple[str, ...]):
        super().__init__()
        self.model = model
        self.input_names = input_names

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        outputs = self.model(**dict(zip(self.input_names, args, strict=True)))
        return outputs["logits"] if isinstance(outputs, dict) else outputs.logits


class Onnx_Mixin:
    """ONNX counterparts to the Hub mixin's save / push / load.

    ``Revised_Mixin`` mixes this in, so every nobg model has:

    - ``onnx_save_pretrained(directory)`` — trace the model and write
      ``model.onnx`` beside the same ``config.json`` and ``README.md``
      ``save_pretrained`` writes.
    - ``onnx_push_to_hub(repo_id)`` — that folder, uploaded.
    - ``onnx_from_pretrained(repo_or_directory)`` — an [`OnnxModel`], the graph
      running under onnxruntime with nobg's ``predict``/``process`` on top.

    The exported graph is the matte alone: the traced wrapper returns
    ``forward(**inputs)["logits"]``, so the auxiliary outputs are pruned. Its
    input shapes are those of the tracing inputs, batch dimension included —
    transformers' Swin windowing reshapes with Python ints, which pins the batch
    size no matter what ``dynamic_axes``/``dynamic_shapes`` claim, so nothing is
    marked dynamic by default. Export at ``batch_size=n`` to run at ``n``.

    Two hooks tailor the export per model:

    - ``onnx_dummy_inputs()`` names, shapes and orders the graph's inputs. The
      default is ``pixel_values`` from ``config.image_size``; a model whose
      ``forward`` needs more (SAM3's tokenized prompt) extends it.
    - ``onnx_dynamo`` picks the exporter. ``True`` is the ``torch.export`` path,
      which BiRefNet requires (torchvision no longer registers an ONNX symbolic
      for ``deform_conv2d``, so the TorchScript exporter cannot lower it).
      ``False`` is the TorchScript tracer, which SAM3 requires (its decoder
      sizes tensors from data, which ``torch.export`` refuses to guard on).
      ``None`` leaves the choice to torch. Ignored on torch < 2.5, whose
      ``torch.onnx.export`` has no ``dynamo`` parameter.

    Both the export and the runtime need extras: ``pip install nobg[onnx]``.
    """

    # The torch.export path. Overridden to False by models it cannot handle.
    onnx_dynamo: bool | None = True

    if TYPE_CHECKING:
        # Mixed into `nn.Module` + `PyTorchModelHubMixin` models; these are the
        # pieces of that interface the methods below use.
        config: Any
        training: bool
        _hub_mixin_config: Any
        _hub_mixin_init_parameters: dict[str, inspect.Parameter]

        def eval(self) -> Any: ...
        def train(self, mode: bool = True) -> Any: ...
        def parameters(self) -> Any: ...
        def generate_model_card(self, *args: Any, **kwargs: Any) -> Any: ...
        @classmethod
        def _decode_arg(cls, expected_type: Any, value: Any) -> Any: ...

    def onnx_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        """Build the inputs the graph is traced with.

        The keys are the ONNX input names and the order is the order ``forward``
        receives them in; the shapes become the graph's shapes. The default
        covers a model whose ``forward`` takes only images, sized from
        ``config.image_size``.

        Args:
            batch_size: Batch size to trace at, and therefore the batch size the
                exported graph accepts.

        Returns:
            A dict of input name to tensor.
        """
        image_size = getattr(getattr(self, "config", None), "image_size", None)
        if image_size is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no config.image_size, so the default "
                "tracing inputs cannot be built; override onnx_dummy_inputs()"
            )
        return {"pixel_values": torch.randn(batch_size, 3, image_size, image_size)}

    def onnx_save_pretrained(
        self,
        save_directory: str | Path,
        *,
        config: dict | DataclassInstance | None = None,
        file_name: str = ONNX_FILE_NAME,
        batch_size: int = 1,
        opset_version: int = DEFAULT_ONNX_OPSET,
        dummy_inputs: dict[str, torch.Tensor] | None = None,
        model_card_kwargs: dict[str, Any] | None = None,
        **export_kwargs: Any,
    ) -> str:
        """Export to ONNX in a local directory.

        The ONNX counterpart of ``save_pretrained``: the same ``config.json`` and
        ``README.md`` (tagged ``onnx``), with ``model.onnx`` in place of
        ``model.safetensors``. A model whose initializers exceed 2 GB also gets
        sidecar data files, which are part of the export — keep the directory
        together.

        Args:
            save_directory: Directory to write to; created if needed.
            config: Config to record, as a dict or dataclass. Defaults to the
                model's own.
            file_name: Name of the ONNX file.
            batch_size: Batch size to trace at, and the only one the graph
                accepts. Ignored when ``dummy_inputs`` is given.
            opset_version: ONNX opset to target. The default is the floor for
                nobg's operators; see [`DEFAULT_ONNX_OPSET`].
            dummy_inputs: Tracing inputs, overriding ``onnx_dummy_inputs()``.
                Use this to trace a variant of the graph — a visual-prompt SAM3
                (adding ``input_boxes``), or images at a non-default size.
            model_card_kwargs: Extra arguments for the model card template.
            **export_kwargs: Forwarded to ``torch.onnx.export``, so its own
                switches (``dynamo``, ``dynamic_shapes``, ``external_data``,
                ``optimize``, ``verify``) remain reachable.

        Returns:
            The path of the written ONNX file.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        inputs = (
            self.onnx_dummy_inputs(batch_size)
            if dummy_inputs is None
            else dict(dummy_inputs)
        )
        if not inputs:
            raise ValueError(
                "no tracing inputs: onnx_dummy_inputs() returned nothing to export"
            )
        # A mixin cannot name its host in its own bases, so spell out what every
        # class this is mixed into is: the `nn.Module` the exporter traces.
        model = cast("nn.Module", self)
        inputs = _to_model_placement(model, inputs)

        if (
            self.onnx_dynamo is not None
            and "dynamo" not in export_kwargs
            and "dynamo" in inspect.signature(torch.onnx.export).parameters
        ):
            export_kwargs["dynamo"] = self.onnx_dynamo

        onnx_path = save_directory / file_name
        # Note the flag before building the wrapper: the model is a submodule of
        # it, so `wrapper.eval()` is what puts the model in eval mode, and the
        # original mode is restored below rather than silently dropped.
        was_training = self.training
        wrapper = _OnnxMatteWrapper(model, tuple(inputs)).eval()
        try:
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    tuple(inputs.values()),
                    str(onnx_path),
                    input_names=list(inputs),
                    output_names=["logits"],
                    opset_version=opset_version,
                    **export_kwargs,
                )
        finally:
            if was_training:
                self.train()

        self._onnx_write_metadata(save_directory, config, model_card_kwargs)
        return str(onnx_path)

    def onnx_push_to_hub(
        self,
        repo_id: str,
        *,
        config: dict | DataclassInstance | None = None,
        commit_message: str = "Push ONNX model using huggingface_hub.",
        private: bool | None = None,
        token: str | None = None,
        branch: str | None = None,
        create_pr: bool | None = None,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        delete_patterns: list[str] | str | None = None,
        file_name: str = ONNX_FILE_NAME,
        batch_size: int = 1,
        opset_version: int = DEFAULT_ONNX_OPSET,
        dummy_inputs: dict[str, torch.Tensor] | None = None,
        model_card_kwargs: dict[str, Any] | None = None,
        **export_kwargs: Any,
    ) -> str:
        """Export to ONNX and upload the result to the Hub.

        ``push_to_hub`` for the exported graph: ``repo_id`` is auto-prefixed with
        your username when it has no owner, and the whole export — ONNX file, any
        sidecar data, ``config.json``, ``README.md`` — goes up in one commit.
        Because the ONNX weights are their own artifact, prefer a separate repo
        (or at least ``file_name``/``subfolder`` discipline) over mixing them into
        a torch checkpoint's repo.

        Args:
            repo_id: Target repo, e.g. ``"nobg/birefnet-onnx"``. Without a ``/``,
                your username is prepended.
            config: Config to record, as a dict or dataclass. Defaults to the
                model's own.
            commit_message: Commit message for the upload.
            private: Whether a newly created repo is private.
            token: Hub token; defaults to the cached login.
            branch: Branch to push to. Defaults to ``main``.
            create_pr: Open a pull request instead of committing directly.
            allow_patterns: Only upload files matching these patterns.
            ignore_patterns: Skip files matching these patterns.
            delete_patterns: Delete matching remote files in the same commit.
            file_name: Name of the ONNX file.
            batch_size: Batch size to trace at. Ignored with ``dummy_inputs``.
            opset_version: ONNX opset to target.
            dummy_inputs: Tracing inputs, overriding ``onnx_dummy_inputs()``.
            model_card_kwargs: Extra arguments for the model card template.
            **export_kwargs: Forwarded to ``torch.onnx.export``.

        Returns:
            The URL of the resulting commit.
        """
        from huggingface_hub import HfApi
        from huggingface_hub.utils import SoftTemporaryDirectory

        if model_card_kwargs is None:
            model_card_kwargs = {}
        if "/" not in repo_id:
            repo_id = f"{whoami(token=token)['name']}/{repo_id}"
        api = HfApi(token=token)
        repo_id = api.create_repo(
            repo_id=repo_id, private=private, exist_ok=True
        ).repo_id
        model_card_kwargs["repo_id"] = repo_id

        with SoftTemporaryDirectory() as tmp:
            saved_path = Path(tmp) / repo_id
            self.onnx_save_pretrained(
                saved_path,
                config=config,
                file_name=file_name,
                batch_size=batch_size,
                opset_version=opset_version,
                dummy_inputs=dummy_inputs,
                model_card_kwargs=model_card_kwargs,
                **export_kwargs,
            )
            return api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message=commit_message,
                revision=branch,
                create_pr=create_pr,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete_patterns,
            )

    @classmethod
    def onnx_from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        file_name: str = ONNX_FILE_NAME,
        providers: list | None = None,
        provider_options: list[dict] | None = None,
        session_options: Any | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
    ) -> OnnxModel:
        """Load an exported ONNX model from the Hub or a local directory.

        The ONNX counterpart of ``from_pretrained``. Nothing is instantiated in
        torch: the graph is handed to onnxruntime and wrapped in an
        [`OnnxModel`], which carries this class's ``config`` (read back from
        ``config.json``) so ``predict``/``process`` work as they do on the torch
        model.

        Args:
            pretrained_model_name_or_path: A Hub repo id, or a local directory
                holding the export.
            file_name: Name of the ONNX file within it.
            providers: onnxruntime execution providers, highest priority first.
                Defaults to every provider installed, so a GPU build uses the
                GPU. Pass ``["CPUExecutionProvider"]`` to pin to CPU.
            provider_options: Per-provider option dicts, aligned with
                ``providers``.
            session_options: An ``onnxruntime.SessionOptions``, for thread counts
                and graph-optimization level.
            revision: Hub revision — branch, tag or commit.
            token: Hub token; defaults to the cached login.
            cache_dir: Where downloads are cached.
            force_download: Re-download even when cached.
            local_files_only: Never hit the network.

        Returns:
            An [`OnnxModel`] wrapping the onnxruntime session.
        """
        model_id = str(pretrained_model_name_or_path)
        if os.path.isdir(model_id):
            directory = Path(model_id)
        else:
            from huggingface_hub import snapshot_download

            # Whole snapshot minus torch weights: sidecar data files from a large
            # export have no predictable names, and the ONNX file is unusable
            # without them.
            directory = Path(
                snapshot_download(
                    model_id,
                    revision=revision,
                    token=token,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    ignore_patterns=_TORCH_WEIGHT_PATTERNS,
                )
            )

        onnx_path = directory / file_name
        if not onnx_path.is_file():
            raise FileNotFoundError(
                f"no {file_name!r} in {directory}; pass `file_name=` if the export "
                "is named differently, or use from_pretrained for a torch checkpoint"
            )

        config = None
        config_path = directory / "config.json"
        if config_path.is_file():
            with open(config_path) as f:
                config = cls._onnx_decode_config(json.load(f))

        return OnnxModel(
            _inference_session(
                onnx_path,
                providers=providers,
                provider_options=provider_options,
                session_options=session_options,
            ),
            config=config,
            model_class=cls,
            path=onnx_path,
            # Wherever the graph came from is also the best guess for a tokenizer,
            # which SAM3's `default_processor` needs and no config.json records.
            tokenizer_source=model_id,
        )

    def _onnx_write_metadata(
        self,
        save_directory: Path,
        config: dict | DataclassInstance | None,
        model_card_kwargs: dict[str, Any] | None,
    ) -> None:
        """Write the ``config.json`` and ``README.md`` that accompany the graph.

        Mirrors ``save_pretrained``'s handling of both, except that the card is
        tagged ``onnx`` so an exported repo is recognizable as one on the Hub.
        """
        if config is None:
            config = getattr(self, "_hub_mixin_config", None)
        if config is None:
            config = getattr(self, "config", None)
        if config is not None:
            if is_dataclass(config) and not isinstance(config, type):
                config = asdict(config)
            (save_directory / "config.json").write_text(
                json.dumps(config, sort_keys=True, indent=2)
            )

        model_card_path = save_directory / "README.md"
        if not model_card_path.exists():  # do not overwrite if already exists
            model_card_kwargs = model_card_kwargs or {}
            card = self.generate_model_card(**model_card_kwargs)
            # `generate_model_card` hands back the class-level card data, so copy
            # it before tagging: the tag must not leak into later torch pushes.
            card.data = copy.deepcopy(card.data)
            tags = list(card.data.tags or [])
            if "onnx" not in tags:
                card.data.tags = [*tags, "onnx"]
            card.text = (
                card.text.rstrip()
                + "\n"
                + _ONNX_USAGE_SECTION.format(
                    class_name=type(self).__name__,
                    repo_id=model_card_kwargs.get("repo_id") or "<repo id>",
                )
            )
            card.save(model_card_path)

    @classmethod
    def _onnx_decode_config(cls, config_dict: dict) -> Any:
        """Rebuild this model's config dataclass from a ``config.json`` dict.

        Uses the Hub mixin's own decoder against the annotation of ``__init__``'s
        ``config`` parameter — the dataclass every nobg model is configured by —
        so unknown fields are dropped exactly as ``from_pretrained`` drops them.
        Falls back to a namespace when there is nothing to decode against, since
        attribute access is all the config is used for here.
        """
        parameter = cls._hub_mixin_init_parameters.get("config")
        if (
            parameter is not None
            and parameter.annotation is not inspect.Parameter.empty
        ):
            decoded = cls._decode_arg(parameter.annotation, config_dict)
            if is_dataclass(decoded):
                return decoded
        return SimpleNamespace(**config_dict)


def _inference_session(
    path: str | Path,
    *,
    providers: list | None = None,
    provider_options: list[dict] | None = None,
    session_options: Any | None = None,
):
    """Open an onnxruntime session on an exported graph."""
    try:
        import onnxruntime as ort
    except ImportError as err:
        raise ImportError(
            "onnxruntime is required to run an exported nobg model: "
            "pip install nobg[onnx]"
        ) from err

    return ort.InferenceSession(
        str(path),
        sess_options=session_options,
        providers=ort.get_available_providers() if providers is None else providers,
        provider_options=provider_options,
    )


class OnnxModel:
    """An exported nobg model, running under onnxruntime instead of torch.

    Returned by ``onnx_from_pretrained``. It quacks like the torch model where it
    matters: callable with the same keyword inputs, returning a dict of torch
    tensors with the same ``logits``, carrying the same ``config`` — which is
    what lets the shared ``predict`` (and so ``predict``/``process`` here) run
    against it unchanged.

    What it is not is a ``nn.Module``: there are no parameters, no autograd and
    no training. ``parameters()`` is deliberately empty, which is how ``predict``
    reads it as a CPU float32 model, matching what the session takes.

    The graph's shapes are fixed at export time (see [`Onnx_Mixin`]), so
    ``batch_size`` is not a free choice at inference: it defaults to whatever the
    graph was traced at, exposed as ``batch_size``.
    """

    def __init__(
        self,
        session: Any,
        *,
        config: Any | None = None,
        model_class: type | None = None,
        path: str | Path | None = None,
        tokenizer_source: str | None = None,
    ):
        self.session = session
        self.config = config
        self.path = None if path is None else str(path)
        self.training = False
        self._model_class = model_class
        # Read by `Sam3.default_processor` through `default_processor` below.
        self._tokenizer_source = tokenizer_source
        self._cached_processor = None

    @property
    def input_names(self) -> tuple[str, ...]:
        """The graph's input names, in order."""
        return tuple(node.name for node in self.session.get_inputs())

    @property
    def output_names(self) -> tuple[str, ...]:
        """The graph's output names, in order."""
        return tuple(node.name for node in self.session.get_outputs())

    @property
    def providers(self) -> list[str]:
        """The execution providers this session is running on."""
        return self.session.get_providers()

    @property
    def batch_size(self) -> int:
        """The batch size the graph was exported at.

        Falls back to 1 when the first input's batch dimension is symbolic, i.e.
        the export declared it dynamic.
        """
        shape = self.session.get_inputs()[0].shape
        return shape[0] if shape and isinstance(shape[0], int) else 1

    def parameters(self):
        """An empty iterator — the weights are initializers inside the graph.

        Present because ``utils.predict`` reads a device and dtype off the model;
        no parameters means CPU float32, which is what the session accepts.
        """
        return iter(())

    def eval(self) -> OnnxModel:
        """No-op: an exported graph is always in inference mode."""
        return self

    def train(self, mode: bool = True) -> OnnxModel:
        """No-op for ``mode=False``; an error otherwise.

        An ONNX graph has no trainable parameters. Load the torch model with
        ``from_pretrained`` to fine-tune.
        """
        if mode:
            raise RuntimeError(
                "an exported ONNX graph cannot be trained; load the torch model "
                "with from_pretrained() instead"
            )
        return self

    def forward(self, **inputs: Any) -> dict[str, torch.Tensor]:
        """Run the graph.

        Args:
            **inputs: One entry per graph input, named as ``input_names``. Torch
                tensors and numpy arrays are both accepted; tensors are moved to
                CPU and cast to the dtype the graph declares.

        Returns:
            A dict of output name to torch tensor, always including ``logits`` —
            the raw ``(B, 1, H, W)`` alpha-matte logits, the same contract as the
            torch model's output.
        """
        expected = {node.name: node.type for node in self.session.get_inputs()}
        unexpected = [name for name in inputs if name not in expected]
        if unexpected:
            raise TypeError(
                f"{type(self).__name__} got unexpected input(s) {unexpected}; this "
                f"graph takes {tuple(expected)}"
            )
        missing = [name for name in expected if name not in inputs]
        if missing:
            raise ValueError(
                f"missing input(s) {missing}; this graph takes {tuple(expected)}"
            )

        feed = {}
        for name, value in inputs.items():
            if isinstance(value, torch.Tensor):
                dtype = _ONNX_TO_TORCH_DTYPE.get(expected[name])
                value = value.detach().cpu()
                if dtype is not None and value.dtype != dtype:
                    value = value.to(dtype)
                value = value.numpy()
            feed[name] = value

        outputs = self.session.run(None, feed)
        return {
            name: torch.from_numpy(value)
            for name, value in zip(self.output_names, outputs, strict=True)
        }

    def __call__(self, **inputs: Any) -> dict[str, torch.Tensor]:
        return self.forward(**inputs)

    def default_processor(self, *args: Any, **kwargs: Any):
        """Build the processor this checkpoint implies.

        Delegates to the torch class's own ``default_processor`` with this object
        standing in for the model. That method reads nothing but the config (and,
        for SAM3, the tokenizer source and processor cache), all of which an
        exported model carries, so the processor is identical to the one the
        torch model would have built.
        """
        method = getattr(self._model_class, "default_processor", None)
        if method is None or self.config is None:
            raise RuntimeError(
                "no config or model class for this graph, so its processor cannot "
                "be inferred; pass a processor to predict() instead"
            )
        return method(self, *args, **kwargs)

    def predict(
        self,
        processor,
        image,
        prompt=None,
        boxes=None,
        *,
        batch_size: int | None = None,
        return_type: str = "cutout",
        **processor_kwargs,
    ):
        """Remove the background from one or more images, end to end.

        The exported-graph twin of the torch models' ``predict``: same
        preprocess → run → post-process → composite sequence, same arguments,
        same returns. ``prompt`` and ``boxes`` only mean anything for a graph
        whose inputs cover them (SAM3's).

        Args:
            processor: The processor for this model, e.g. from
                ``default_processor()`` or ``AutoProcessor``.
            image: Anything ``loadimg.load_img`` accepts — a path, URL, base64
                string, numpy array or PIL image — or a list of them.
            prompt: Optional text prompt, applied to every image in the call.
            boxes: Optional per-image visual prompt, in the original image's
                pixel coordinates.
            batch_size: Images per run. Defaults to the graph's own
                ``batch_size``; a different value needs a graph exported at it.
            return_type: ``"cutout"`` for RGBA images, ``"alpha"`` for the raw
                ``(H, W)`` mattes in ``[0, 1]``.
            **processor_kwargs: Forwarded to the processor.

        Returns:
            A single result for a single image, or a list for a list of images.
        """
        return predict(
            self,
            processor,
            image,
            prompt,
            boxes,
            forward_keys=self.input_names,
            processor_kwargs=processor_kwargs,
            batch_size=self.batch_size if batch_size is None else batch_size,
            return_type=return_type,
        )

    def process(
        self,
        image,
        prompt=None,
        boxes=None,
        *,
        tokenizer=None,
        batch_size: int | None = None,
        return_type: str = "cutout",
        **processor_kwargs,
    ):
        """Remove the background from one or more images, without a processor.

        ``predict`` without the ``processor`` argument: it builds the one
        ``default_processor`` describes and forwards everything else unchanged.

        Args:
            image: Anything ``loadimg.load_img`` accepts, or a list of them.
            prompt: Optional text prompt, applied to every image in the call.
            boxes: Optional per-image visual prompt, in pixel coordinates.
            tokenizer: Passed to ``default_processor`` for models that need one
                (SAM3); leave unset otherwise.
            batch_size: Images per run. Defaults to the graph's ``batch_size``.
            return_type: ``"cutout"`` for RGBA images, ``"alpha"`` for mattes.
            **processor_kwargs: Forwarded to the processor.

        Returns:
            Exactly what ``predict`` returns for the same inputs.
        """
        processor = (
            self.default_processor()
            if tokenizer is None
            else self.default_processor(tokenizer)
        )
        return self.predict(
            processor,
            image,
            prompt,
            boxes,
            batch_size=batch_size,
            return_type=return_type,
            **processor_kwargs,
        )

    def __repr__(self) -> str:
        name = getattr(self._model_class, "__name__", "?")
        return (
            f"{type(self).__name__}({name}, inputs={list(self.input_names)}, "
            f"batch_size={self.batch_size}, providers={self.providers})"
        )


class Revised_Mixin(PyTorchModelHubMixin, Onnx_Mixin):
    @set_doc(PyTorchModelHubMixin.push_to_hub.__doc__)
    def push_to_hub(
        self,
        repo_id: str,
        *,
        config: dict | DataclassInstance | None = None,
        commit_message: str = "Push model using huggingface_hub.",
        private: bool | None = None,
        token: str | None = None,
        branch: str | None = None,
        create_pr: bool | None = None,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        delete_patterns: list[str] | str | None = None,
        model_card_kwargs: dict[str, Any] | None = None,
    ) -> str:
        if model_card_kwargs is None:
            model_card_kwargs = {}
        if "/" not in repo_id:
            username = whoami()["name"]
            repo_id = f"{username}/{repo_id}"
        model_card_kwargs["repo_id"] = repo_id
        return super().push_to_hub(
            repo_id,
            config=config,
            commit_message=commit_message,
            private=private,
            token=token,
            branch=branch,
            create_pr=create_pr,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            delete_patterns=delete_patterns,
            model_card_kwargs=model_card_kwargs,
        )
