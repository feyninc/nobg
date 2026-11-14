"""ComfyUI nodes for nobg background removal and foreground refinement."""

import json
import logging
import os
import re
import shutil
import tempfile
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import comfy.model_management as mm  # ty: ignore[unresolved-import]
import comfy.utils  # ty: ignore[unresolved-import]
import folder_paths  # ty: ignore[unresolved-import]
import torch

logger = logging.getLogger("ComfyUI-NoBG")

NOBG_DIR = os.path.join(folder_paths.models_dir, "nobg")
folder_paths.add_model_folder_path("nobg", NOBG_DIR)

PRESETS = {
    "feyninc/FeyNobg": "birefnet",
    "facebook/sam3": "sam3",
}
LOCAL_PREFIX = "local: "
DOWNLOAD_IGNORE = [
    "*.pt",
    "*.pth",
    "*.bin",
    "*.ckpt",
    "*.onnx",
    "*.onnx_data",
    "*.h5",
    "*.msgpack",
]
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_REPO_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
)


@dataclass(frozen=True)
class NoBGCheckpoint:
    """A pinned, locally validated checkpoint (not a workflow path selector)."""

    path: str
    architecture: str
    repo_id: str
    commit: str
    revision: str = ""


def _registered_checkpoint(path: str) -> str:
    """Confine a checkpoint to one immediate folder under a registered root."""
    candidate = Path(path)
    if not candidate.is_absolute() or candidate.is_symlink():
        raise ValueError("nobg checkpoint must be a real registered model directory")
    for root_name in folder_paths.get_folder_paths("nobg"):
        root = Path(root_name)
        if root.is_symlink():
            continue
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) != 1 or relative.parts[0] in (".", ".."):
            continue
        if not root.is_dir() or not candidate.is_dir():
            continue
        if candidate.resolve() != root.resolve() / relative:
            continue
        for filename in ("config.json", "model.safetensors"):
            item = candidate / filename
            if (
                item.is_symlink()
                or not item.is_file()
                or item.resolve() != candidate / filename
            ):
                raise ValueError(f"nobg checkpoint has an unsafe {filename}")
        return str(candidate)
    raise ValueError(
        "nobg checkpoint is not directly under a registered models/nobg root"
    )


def _validated_checkpoint(checkpoint: NoBGCheckpoint) -> tuple[str, str]:
    if type(checkpoint) is not NoBGCheckpoint:
        raise ValueError("checkpoint must originate from NoBG Download Model")
    source = _registered_checkpoint(checkpoint.path)
    if _local_architecture(source) != checkpoint.architecture:
        raise ValueError("nobg checkpoint architecture no longer matches its config")
    metadata = _download_metadata(source)
    if metadata != (
        checkpoint.repo_id,
        checkpoint.commit,
        checkpoint.architecture,
        checkpoint.revision,
    ):
        raise ValueError("nobg checkpoint provenance no longer matches its download")
    return source, checkpoint.architecture


def _download_metadata(source: str) -> tuple[str, str, str, str]:
    metadata_path = Path(source) / ".nobg-download.json"
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ValueError("nobg checkpoint has no download provenance")
    with metadata_path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise TypeError("nobg checkpoint has invalid download provenance")
    repo, commit, architecture, revision = (
        data.get("repo_id"),
        data.get("commit"),
        data.get("architecture"),
        data.get("revision", ""),
    )
    if (
        not isinstance(repo, str)
        or not _REPO_RE.fullmatch(repo)
        or not isinstance(commit, str)
        or not _COMMIT_RE.fullmatch(commit)
        or architecture not in ("birefnet", "sam3")
        or not isinstance(revision, str)
    ):
        raise ValueError("nobg checkpoint has invalid download provenance")
    return repo, commit, architecture, revision


@contextmanager
def _download_lock(path: Path):
    # flock serializes competing downloads to the same final directory on Linux.
    import fcntl

    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _check_snapshot_files(path: Path) -> None:
    """Reject symlinks and external hardlinks in a Hub snapshot before publish."""
    for base, directories, files in os.walk(path, followlinks=False):
        for name in directories + files:
            item = Path(base) / name
            if item.is_symlink():
                raise ValueError(f"download contains a symlink: {item.name}")
            if item.is_file() and item.stat().st_nlink != 1:
                raise ValueError(f"download contains a hardlinked file: {item.name}")


def _download_checkpoint(repo_id: str, revision: str = "") -> NoBGCheckpoint:
    from huggingface_hub import HfApi, snapshot_download

    repo_id = repo_id.strip()
    revision = revision.strip()
    if not _REPO_RE.fullmatch(repo_id) or any(
        part.startswith(".") or part.endswith(".") or ".." in part
        for part in repo_id.split("/")
    ):
        raise ValueError("repo_id must be an owner-qualified Hugging Face model ID")
    if revision and (
        revision in (".", "..")
        or any(char in revision for char in ("\x00", "\n", "\r"))
    ):
        raise ValueError("invalid model revision")
    root = Path(NOBG_DIR)
    if root.is_symlink() or (root.parent.exists() and root.parent.is_symlink()):
        raise ValueError("models/nobg must not be a symlink")
    if root.is_dir():
        # Do not write through an owner-named symlink left by another integration.
        owner = repo_id.split("/")[0]
        if (root / owner).is_symlink():
            raise ValueError("model owner path must not be a symlink")
        # Explicit revisions are stable and reused without a network request.
        # An unpinned branch name intentionally remains pinned until the user
        # removes its downloaded folder to request a fresh resolution.
        for candidate in root.iterdir():
            if (
                candidate.is_dir()
                and not candidate.is_symlink()
                and candidate.name.startswith(
                    f"{urllib.parse.quote(repo_id.split('/')[0], safe='')}"
                    f"--{urllib.parse.quote(repo_id.split('/')[1], safe='')}--"
                )
            ):
                try:
                    source = _registered_checkpoint(str(candidate))
                    existing = _download_metadata(source)
                except (ValueError, FileNotFoundError, OSError):
                    continue
                if existing[0] == repo_id and existing[3] == revision:
                    return NoBGCheckpoint(
                        source, existing[2], repo_id, existing[1], revision
                    )

    info = HfApi().model_info(repo_id, revision=revision or None)
    commit = info.sha
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        raise ValueError(f"Hub returned an invalid commit for {repo_id!r}")

    # Owner and repository are escaped independently; the commit pins the
    # destination, so a later branch update cannot mutate a prior download.
    owner, repository = repo_id.split("/")
    folder = (
        f"{urllib.parse.quote(owner, safe='')}"
        f"--{urllib.parse.quote(repository, safe='')}"
        f"--{revision.encode().hex() if revision else 'default'}--{commit}"
    )
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink():
        raise ValueError("models/nobg must not be a symlink")
    destination = root / folder
    with _download_lock(root / f".{folder}.lock"):
        if destination.exists() or destination.is_symlink():
            source = _registered_checkpoint(str(destination))
            if _download_metadata(source) != (
                repo_id,
                commit,
                _local_architecture(source),
                revision,
            ):
                raise ValueError(
                    f"refusing to overwrite existing checkpoint {destination}"
                )
            return NoBGCheckpoint(
                source, _local_architecture(source), repo_id, commit, revision
            )

        staging = Path(tempfile.mkdtemp(prefix=".nobg-download-", dir=root))
        staging_identity = staging.stat()
        try:
            snapshot_download(
                repo_id=repo_id,
                revision=commit,
                local_dir=str(staging),
                allow_patterns=["config.json", "model.safetensors", *_TOKENIZER_FILES],
            )
            _check_snapshot_files(staging)
            _require_local_weights(str(staging))
            architecture = _local_architecture(str(staging))
            if architecture == "sam3":
                tokenizer = staging / "tokenizer.json"
                vocabulary = staging / "vocab.json"
                merges = staging / "merges.txt"
                if not tokenizer.is_file() and not (
                    vocabulary.is_file() and merges.is_file()
                ):
                    logger.info(
                        "%s has no SAM3 tokenizer; default_processor uses the ungated "
                        "openai/clip-vit-large-patch14 CLIP tokenizer fallback",
                        repo_id,
                    )
            (staging / ".nobg-download.json").write_text(
                json.dumps(
                    {
                        "repo_id": repo_id,
                        "commit": commit,
                        "architecture": architecture,
                        "revision": revision,
                    }
                ),
                encoding="utf-8",
            )
            # Same-filesystem rename publishes only fully validated snapshots.
            # The per-destination lock ensures other downloads cannot race this.
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(
                    f"checkpoint destination already exists: {destination}"
                )
            staging.rename(destination)
        finally:
            if (
                staging.exists()
                and not staging.is_symlink()
                and staging.stat().st_ino == staging_identity.st_ino
                and staging.stat().st_dev == staging_identity.st_dev
            ):
                shutil.rmtree(staging)
    return NoBGCheckpoint(str(destination), architecture, repo_id, commit, revision)


def _download_is_changed(
    repo_id: str,
    revision: str = "",
    **_: Any,
) -> tuple[str, str]:
    # Avoid rerunning a completed download just because a downstream node changed.
    # A changed revision or repo creates a distinct checkpoint.
    return repo_id, revision


@dataclass
class NoBGModel:
    """Model and placement state passed between ComfyUI nodes."""

    model: Any
    architecture: str
    source: str
    device: torch.device
    _processor: Any = field(default=None, repr=False)

    @property
    def processor(self):
        if self._processor is None:
            self._processor = self.model.default_processor()
        return self._processor

    def param_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    def current_device(self) -> torch.device:
        param = next(iter(self.model.parameters()), None)
        return param.device if param is not None else torch.device("cpu")

    def to_device(self):
        if self.current_device() != self.device:
            if self.device.type != "cpu":
                # Include a modest inference buffer, not just model parameters.
                mm.free_memory(
                    int(self.param_bytes() * 1.2) + 256 * 1024 * 1024,
                    self.device,
                )
            self.model.to(self.device)

    def offload(self):
        offload_device = mm.unet_offload_device()
        if self.current_device() != offload_device:
            self.model.to(offload_device)
            mm.soft_empty_cache()


def _local_models() -> list[str]:
    names = []
    roots = folder_paths.get_folder_paths("nobg")
    discovered = []
    for index, root in enumerate(roots):
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if _has_local_checkpoint(path):
                discovered.append((name, index, path))
    counts: dict[str, int] = {}
    for name, _, _ in discovered:
        counts[name] = counts.get(name, 0) + 1
    for name, index, _ in discovered:
        # Keep legacy simple labels for ordinary unique folders. Encoded labels
        # cannot collide with literal names because every "@"-prefixed name is
        # encoded, including names that merely resemble an encoded selector.
        label = f"@{index + 1}:{urllib.parse.quote(name, safe='')}"
        if counts[name] == 1 and not name.startswith("@"):
            label = name
        names.append(LOCAL_PREFIX + label)
    return names


def _has_local_checkpoint(path: str) -> bool:
    return os.path.isdir(path) and all(
        os.path.isfile(os.path.join(path, filename))
        for filename in ("config.json", "model.safetensors")
    )


def _encoded_label(label: str) -> tuple[str, int] | None:
    if label.startswith("@") and ":" in label:
        number, encoded_name = label[1:].split(":", 1)
        if number.isdigit() and encoded_name:
            name = urllib.parse.unquote(encoded_name)
            if urllib.parse.quote(name, safe="") == encoded_name:
                return name, int(number) - 1
    return None


def _resolve_local(name: str) -> str:
    roots = folder_paths.get_folder_paths("nobg")
    if name != os.path.basename(name) or name in (".", ".."):
        raise ValueError(f"invalid nobg model folder name {name!r}")
    encoded = _encoded_label(name)
    if encoded is not None:
        base, index = encoded
        if base != os.path.basename(base) or base in (".", ".."):
            raise ValueError(f"invalid nobg model folder name {name!r}")
        if 0 <= index < len(roots):
            path = os.path.join(roots[index], base)
            if os.path.isfile(os.path.join(path, "config.json")):
                return path
        raise FileNotFoundError(f"nobg model folder {name!r} not found")
    matches = [
        os.path.join(root, name)
        for root in roots
        if os.path.isfile(os.path.join(root, name, "config.json"))
    ]
    if len(matches) > 1:
        raise ValueError(
            f"nobg model folder {name!r} is ambiguous; select a numbered local model"
        )
    if matches:
        return matches[0]
    raise FileNotFoundError(f"nobg model folder {name!r} not found in models/nobg")


def _local_config(source: str) -> dict[str, Any]:
    with open(os.path.join(source, "config.json")) as config_file:
        return json.load(config_file)


def _local_architecture(source: str, config: dict[str, Any] | None = None) -> str:
    if config is None:
        config = _local_config(source)
    if not isinstance(config, dict):
        raise TypeError(
            f"unsupported local nobg model at {source!r}: invalid config.json"
        )
    model_type = config.get("model_type")
    # Extend this registry when another supported architecture is added.
    architecture_signatures = {
        "sam3": (("sam3", "sam3_video"), ("vision_hidden_size", "default_prompt")),
        "birefnet": ((), ("embed_dim", "depths")),
    }
    if model_type is not None:
        if model_type in architecture_signatures["sam3"][0]:
            return "sam3"
        raise ValueError(
            f"unsupported local nobg model at {source!r}: "
            f"model_type={model_type!r} is not a supported SAM3 checkpoint"
        )
    # nobg Hub mixin writes the flat dataclass without a model_type. These
    # fields distinguish its SAM3 and BiRefNet configs without folder-name guesses.
    for architecture, (_, fields) in architecture_signatures.items():
        if all(field in config for field in fields):
            return architecture
    raise ValueError(
        f"unsupported local nobg model at {source!r}: config.json has no "
        "recognizable SAM3 or BiRefNet architecture fields"
    )


def _require_local_weights(source: str) -> None:
    if not os.path.isfile(os.path.join(source, "model.safetensors")):
        raise FileNotFoundError(
            f"nobg checkpoint at {source!r} needs model.safetensors"
        )


def _guess_architecture(name: str) -> str:
    if name in PRESETS:
        return PRESETS[name]
    raise ValueError(
        f"unsupported nobg model {name!r}; select a preset or a local checkpoint"
    )


def _load(
    architecture: str,
    source: str,
    sam3_aggregate: str,
    *,
    upstream_sam3: bool | None = None,
):
    from nobg import BiRefNet, Sam3

    if architecture == "birefnet":
        return BiRefNet.from_pretrained(source)
    if architecture == "sam3":
        if upstream_sam3 is None:
            upstream_sam3 = source == "facebook/sam3" or (
                os.path.isfile(os.path.join(source, "config.json"))
                and _local_config(source).get("model_type") in ("sam3", "sam3_video")
            )
        if upstream_sam3:
            return Sam3.from_origin(source, aggregate=sam3_aggregate)
        return Sam3.from_pretrained(source)
    raise ValueError(f"unknown nobg architecture {architecture!r}")


def _parse_boxes(text: str) -> list[list[float]] | None:
    boxes = []
    for chunk in text.replace(";", "\n").splitlines():
        chunk = chunk.strip().strip("[]()")
        if not chunk:
            continue
        values = [float(v) for v in chunk.replace(" ", ",").split(",") if v]
        if len(values) != 4:
            raise ValueError(f"box {chunk!r} must have 4 values: x1, y1, x2, y2")
        boxes.append(values)
    return boxes or None


class NoBGDownloadModel:
    IS_CHANGED = staticmethod(_download_is_changed)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": (
                    "STRING",
                    {"default": "feyninc/FeyNobg", "multiline": False},
                ),
            },
            "optional": {
                "revision": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("NOBG_CHECKPOINT",)
    RETURN_NAMES = ("checkpoint",)
    FUNCTION = "download"
    CATEGORY = "NoBG"

    def download(self, repo_id: str, revision: str = ""):
        return (_download_checkpoint(repo_id, revision),)


class NoBGModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    list(PRESETS) + _local_models(),
                    {"default": "feyninc/FeyNobg"},
                ),
            },
            "optional": {"checkpoint": ("NOBG_CHECKPOINT",)},
        }

    RETURN_TYPES = ("NOBG_MODEL",)
    RETURN_NAMES = ("nobg_model",)
    FUNCTION = "load"
    CATEGORY = "NoBG"

    # Keep the old Python arguments for direct callers, but not as UI widgets.
    def load(
        self,
        model,
        architecture="auto",
        device="auto",
        dtype="auto",
        download_to_models_dir=False,
        sam3_aggregate="semantic",
        checkpoint: NoBGCheckpoint | None = None,
    ):
        if checkpoint is not None:
            source, checkpoint_architecture = _validated_checkpoint(checkpoint)
            name = checkpoint.repo_id
            local = True
            architecture = checkpoint_architecture
        elif model.startswith(LOCAL_PREFIX):
            local = True
            name = model[len(LOCAL_PREFIX) :]
            source = _resolve_local(name)
        else:
            local = False
            name = model
            source = model
            if download_to_models_dir:
                from huggingface_hub import snapshot_download

                source = snapshot_download(
                    model,
                    local_dir=os.path.join(NOBG_DIR, model.split("/")[-1]),
                    ignore_patterns=DOWNLOAD_IGNORE,
                )
        if local or (download_to_models_dir and os.path.isdir(source)):
            _require_local_weights(source)
        config = (
            _local_config(source)
            if local or os.path.isfile(os.path.join(source, "config.json"))
            else None
        )
        if architecture == "auto":
            architecture = (
                _local_architecture(source, config)
                if config is not None
                else _guess_architecture(name)
            )

        logger.info("loading nobg %s model from %s", architecture, source)
        upstream_sam3 = name == "facebook/sam3" or (
            config is not None and config.get("model_type") in ("sam3", "sam3_video")
        )
        net = _load(architecture, source, sam3_aggregate, upstream_sam3=upstream_sam3)
        if dtype != "auto":
            net = net.to(DTYPES[dtype])
        net.eval()

        target = mm.get_torch_device() if device == "auto" else torch.device(device)
        return (NoBGModel(net, architecture, source, target),)


class NoBGRemoveBackground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nobg_model": ("NOBG_MODEL",),
                "image": ("IMAGE",),
                "refine_foreground": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Unmix the old background from soft edges (hair, fur) to remove halos.",
                    },
                ),
            },
            "optional": {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "defaultInput": False,
                        "multiline": True,
                        "tooltip": "SAM3 only: what to cut out. Empty uses the model's default prompt.",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                        "tooltip": "Images per inference pass. Larger batches are faster but use more VRAM.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("cutout", "foreground", "mask")
    OUTPUT_TOOLTIPS = (
        "RGBA cutout; save it with Save Image for a transparent PNG.",
        "RGB foreground colors (refined when refine_foreground is on).",
        "Alpha matte in [0, 1].",
    )
    FUNCTION = "remove"
    CATEGORY = "NoBG"

    def remove(
        self,
        nobg_model: NoBGModel,
        image,
        batch_size=1,
        refine_foreground=False,
        refine_radius=90,
        keep_model_loaded=False,
        prompt="",
        boxes="",
        score_threshold=-1.0,
    ):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if image.ndim != 4 or image.shape[-1] < 3 or image.shape[0] == 0:
            raise ValueError("image must be a non-empty [B, H, W, C] batch with C >= 3")
        arch = nobg_model.architecture
        prompt = prompt.strip() or None
        parsed_boxes = _parse_boxes(boxes) if boxes else None

        kwargs: dict[str, Any] = {"return_type": "alpha"}
        if arch == "birefnet":
            if prompt or parsed_boxes:
                logger.warning("BiRefNet takes no prompt or boxes; ignoring them")
        elif score_threshold >= 0:
            kwargs["score_threshold"] = score_threshold

        rgb = image[..., :3]
        mattes = []
        try:
            nobg_model.to_device()
            processor = nobg_model.processor
            pbar = comfy.utils.ProgressBar(len(rgb))
            for start in range(0, len(rgb), batch_size):
                mm.throw_exception_if_processing_interrupted()
                chunk = [
                    (frame.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
                    for frame in rgb[start : start + batch_size]
                ]
                call_kwargs = dict(kwargs)
                if arch == "birefnet":
                    out = nobg_model.model.predict(
                        processor, chunk, batch_size=len(chunk), **call_kwargs
                    )
                else:
                    chunk_boxes = [parsed_boxes] * len(chunk) if parsed_boxes else None
                    out = nobg_model.model.predict(
                        processor,
                        chunk,
                        prompt,
                        chunk_boxes,
                        batch_size=len(chunk),
                        **call_kwargs,
                    )
                # predict returns CPU mattes, but custom implementations may not.
                # Keep only the final batch of output mattes in device memory.
                mattes.extend(m.detach().float().clamp(0, 1).cpu() for m in out)
                pbar.update(len(chunk))
        finally:
            if not keep_model_loaded:
                nobg_model.offload()

        # Keep inference mattes on CPU until needed; match the image device for
        # both the MASK output and RGBA compositing.
        mask = torch.stack(mattes).to(rgb.device)
        foreground = rgb
        if refine_foreground:
            from nobg.utils import refine_foreground as refine

            device = mm.get_torch_device()
            refined = []
            for frame, matte in zip(rgb, mask):
                mm.throw_exception_if_processing_interrupted()
                out = cast(
                    torch.Tensor,
                    refine(
                        frame.permute(2, 0, 1).to(device),
                        matte.to(device),
                        r=refine_radius,
                    ),
                )
                refined.append(out.permute(1, 2, 0).to(rgb.device, rgb.dtype))
            foreground = torch.stack(refined)

        cutout = torch.cat([foreground, mask.unsqueeze(-1).to(foreground)], dim=-1)
        return (cutout, foreground, mask)


class NoBGRefineForeground:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "radius": ("INT", {"default": 90, "min": 1, "max": 512}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("foreground", "cutout")
    DESCRIPTION = "Estimate unmixed foreground colors for a soft mask, removing background halos on hair and fur."
    FUNCTION = "refine"
    CATEGORY = "NoBG"

    def refine(self, image, mask, radius):
        from nobg.utils import refine_foreground as refine

        rgb = image[..., :3]
        if rgb.ndim != 4 or rgb.shape[0] == 0:
            raise ValueError("image must be a non-empty [B, H, W, C] batch")
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and rgb.shape[0] > 1:
            mask = mask.expand(rgb.shape[0], -1, -1)
        if mask.shape[0] != rgb.shape[0]:
            raise ValueError(f"got {mask.shape[0]} masks for {rgb.shape[0]} images")

        device = mm.get_torch_device()
        refined, mattes = [], []
        for frame, matte in zip(rgb, mask):
            mm.throw_exception_if_processing_interrupted()
            if matte.shape != frame.shape[:2]:
                matte = torch.nn.functional.interpolate(
                    matte[None, None].float(),
                    size=frame.shape[:2],
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            matte = matte.clamp(0, 1)
            out = cast(
                torch.Tensor,
                refine(frame.permute(2, 0, 1).to(device), matte.to(device), r=radius),
            )
            refined.append(out.permute(1, 2, 0).to(rgb.device, rgb.dtype))
            mattes.append(matte.to(rgb.device, rgb.dtype))
        foreground = torch.stack(refined)
        cutout = torch.cat([foreground, torch.stack(mattes).unsqueeze(-1)], dim=-1)
        return (foreground, cutout)


NODE_CLASS_MAPPINGS = {
    "NoBGDownloadModel": NoBGDownloadModel,
    "NoBGModelLoader": NoBGModelLoader,
    "NoBGRemoveBackground": NoBGRemoveBackground,
    "NoBGRefineForeground": NoBGRefineForeground,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NoBGDownloadModel": "NoBG Download Model",
    "NoBGModelLoader": "NoBG Model Loader",
    "NoBGRemoveBackground": "NoBG Remove Background",
    "NoBGRefineForeground": "NoBG Refine Foreground",
}
