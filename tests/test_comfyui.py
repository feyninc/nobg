"""ComfyUI integration tests; no ComfyUI installation or network access required."""

import importlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import types
import warnings
from pathlib import Path
from unittest import mock
from zipfile import ZipFile, ZipInfo

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "comfyui"


def _release_module():
    pytest.importorskip("tomllib")
    spec = importlib.util.spec_from_file_location(
        "nobg_registry_release", PACK / "release.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registry_metadata_and_release_notes(tmp_path):
    release = _release_module()
    project = release.metadata(PACK)
    assert project["name"] == "nobg"
    version = project["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
    assert project["requires-python"] == ">=3.10"
    minimums = [
        dependency.removeprefix("nobg>=")
        for dependency in project["dependencies"]
        if dependency.startswith("nobg>=")
    ]
    assert len(minimums) == 1
    assert re.fullmatch(r"\d+\.\d+\.\d+", minimums[0])
    assert tuple(map(int, minimums[0].split("."))) >= (0, 3, 2)
    assert release.release_notes(PACK).strip()
    assert release.PUBLISHER == "chonk-lain"
    assert release.NODE == "nobg"
    assert project["name"] == release.NODE
    comfy = release.tomllib.loads((PACK / "pyproject.toml").read_text())["tool"][
        "comfy"
    ]
    assert comfy["PublisherId"] == release.PUBLISHER
    assert comfy["DisplayName"] == "NoBG"
    assert (PACK / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    assert (PACK / ".comfyignore").exists()
    # ComfyUI-Manager reads only requirements.txt, never [project] dependencies.
    requirements = (PACK / "requirements.txt").read_text().split()
    assert requirements == project["dependencies"]
    for name in ("pyproject.toml", "requirements.txt", "CHANGELOG.md"):
        (tmp_path / name).write_bytes((PACK / name).read_bytes())
    (tmp_path / "CHANGELOG.md").write_text(f"## {version}\n \n")
    with pytest.raises(ValueError, match="nonempty"):
        release.release_notes(tmp_path)


@pytest.mark.parametrize(
    "workflow",
    ("NoBG_remove_background.json", "NoBG_sam3_text_prompt.json"),
)
def test_example_workflow_registry_identity(workflow):
    data = json.loads((PACK / "example_workflows" / workflow).read_text())
    nobg_nodes = [node for node in data["nodes"] if node["type"].startswith("NoBG")]
    assert nobg_nodes
    assert all(node["properties"]["cnr_id"] == "nobg" for node in nobg_nodes)


@pytest.mark.parametrize(
    ("workflow", "prompt"),
    (
        ("NoBG_remove_background.json", ""),
        ("NoBG_sam3_text_prompt.json", "the dog"),
    ),
)
def test_example_remove_widget_order(workflow, prompt):
    data = json.loads((PACK / "example_workflows" / workflow).read_text())
    (node,) = (node for node in data["nodes"] if node["type"] == "NoBGRemoveBackground")
    assert node["widgets_values"] == [False, prompt, 1]
    assert [(item["name"], item["link"]) for item in node["inputs"]] == [
        ("nobg_model", 1),
        ("image", 2),
    ]


@pytest.mark.parametrize(
    ("workflow", "model"),
    (
        ("NoBG_remove_background.json", "feyninc/FeyNobg"),
        ("NoBG_sam3_text_prompt.json", "facebook/sam3"),
    ),
)
def test_example_loader_remains_widget_driven(workflow, model):
    data = json.loads((PACK / "example_workflows" / workflow).read_text())
    (loader,) = (node for node in data["nodes"] if node["type"] == "NoBGModelLoader")
    assert loader["inputs"] == []
    assert loader["widgets_values"] == [model]
    assert loader["outputs"][0]["type"] == "NOBG_MODEL"
    (remove,) = (
        node for node in data["nodes"] if node["type"] == "NoBGRemoveBackground"
    )
    assert [loader["id"], 0, remove["id"], 0, "NOBG_MODEL"] in [
        [link[1], link[2], link[3], link[4], link[5]] for link in data["links"]
    ]


@pytest.fixture
def registry_snapshot(tmp_path):
    release = _release_module()
    for name in release.SNAPSHOT_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes((PACK / name).read_bytes())
    with ZipFile(tmp_path / "node.zip", "w") as archive:
        for name in release.ALLOWED_ARCHIVE:
            archive.write(tmp_path / name, name)
    return tmp_path


def test_registry_archive_and_entrypoint_smoke(registry_snapshot):
    tmp_path = registry_snapshot
    release = _release_module()
    release.inspect_archive(tmp_path)
    with ZipFile(tmp_path / "node.zip", "a") as archive:
        archive.writestr("secret.env", "not for publication")
    with pytest.raises(ValueError, match="Unsafe"):
        release.inspect_archive(tmp_path)


def test_registry_archive_requires_examples(registry_snapshot):
    release = _release_module()
    with ZipFile(registry_snapshot / "node.zip", "w") as archive:
        for name in release.ALLOWED_ARCHIVE - release.EXAMPLES:
            archive.write(registry_snapshot / name, name)
    with pytest.raises(ValueError, match="Unsafe or incomplete"):
        release.inspect_archive(registry_snapshot)


def test_registry_archive_rejects_tampered_member(registry_snapshot):
    release = _release_module()
    with ZipFile(registry_snapshot / "node.zip", "w") as archive:
        for name in release.ALLOWED_ARCHIVE:
            if name == "__init__.py":
                archive.writestr(name, "tampered")
            else:
                archive.write(registry_snapshot / name, name)
    with pytest.raises(ValueError, match="differs from snapshot"):
        release.inspect_archive(registry_snapshot)


def test_registry_archive_rejects_duplicate_members(registry_snapshot):
    release = _release_module()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with ZipFile(registry_snapshot / "node.zip", "a") as archive:
            archive.writestr("__init__.py", "replaced")
    with pytest.raises(ValueError, match="Duplicate"):
        release.inspect_archive(registry_snapshot)


@pytest.mark.parametrize("mode", [stat.S_IFLNK, stat.S_IFDIR])
def test_registry_archive_rejects_nonregular_members(registry_snapshot, mode):
    release = _release_module()
    with ZipFile(registry_snapshot / "node.zip", "w") as archive:
        for name in release.ALLOWED_ARCHIVE:
            if name == "__init__.py":
                info = ZipInfo(name)
                info.create_system = 3
                info.external_attr = (mode | 0o755) << 16
                archive.writestr(info, "target")
            else:
                archive.write(registry_snapshot / name, name)
    with pytest.raises(ValueError, match="Nonregular"):
        release.inspect_archive(registry_snapshot)


def test_registry_snapshot_rejects_extra_and_symlink(registry_snapshot):
    release = _release_module()
    extra = registry_snapshot / "secret.env"
    extra.write_text("secret")
    with pytest.raises(ValueError, match="Unsafe or incomplete Registry snapshot"):
        release.inspect_archive(registry_snapshot)
    extra.unlink()
    path = registry_snapshot / "__init__.py"
    path.unlink()
    path.symlink_to("README.md")
    with pytest.raises(ValueError, match="Nonregular Registry snapshot"):
        release.inspect_archive(registry_snapshot)


def test_registry_snapshot_ignores_only_tool_caches(registry_snapshot):
    release = _release_module()
    cache = registry_snapshot / ".ruff_cache" / "0.14.0" / "artifact"
    cache.parent.mkdir(parents=True)
    cache.write_text("ruff cache")
    pycache = registry_snapshot / "__pycache__" / "release.cpython-313.pyc"
    pycache.parent.mkdir()
    pycache.write_bytes(b"compiled")
    release.inspect_archive(registry_snapshot)
    other = registry_snapshot / "other_cache" / "artifact"
    other.parent.mkdir()
    other.write_text("unexpected")
    with pytest.raises(ValueError, match="Unsafe or incomplete Registry snapshot"):
        release.inspect_archive(registry_snapshot)


def test_registry_actual_cli_pack_from_tracked_snapshot(registry_snapshot):
    release = _release_module()
    if not shutil.which("git") or not shutil.which("comfy"):
        pytest.skip("git and comfy-cli are needed for the optional pack smoke")
    subprocess.run(["git", "init", "-q"], cwd=registry_snapshot, check=True)
    (registry_snapshot / "node.zip").unlink()
    subprocess.run(["git", "add", "--all"], cwd=registry_snapshot, check=True)
    env = {**os.environ, "RUFF_NO_CACHE": "true"}
    for command in ("validate", "pack"):
        subprocess.run(
            ["comfy", "--skip-prompt", "--no-enable-telemetry", "node", command],
            cwd=registry_snapshot,
            env=env,
            capture_output=True,
            check=True,
        )
    release.inspect_archive(registry_snapshot)


def test_registry_rejects_tracked_cache_even_when_ignored_on_disk(registry_snapshot):
    release = _release_module()
    if not shutil.which("git"):
        pytest.skip("git is needed to inspect the staged snapshot")
    cache = registry_snapshot / ".ruff_cache" / "cache"
    cache.parent.mkdir()
    cache.write_text("staged cache")
    subprocess.run(["git", "init", "-q"], cwd=registry_snapshot, check=True)
    (registry_snapshot / "node.zip").unlink()
    subprocess.run(["git", "add", "--all"], cwd=registry_snapshot, check=True)
    with ZipFile(registry_snapshot / "node.zip", "w") as archive:
        for name in release.ALLOWED_ARCHIVE:
            archive.write(registry_snapshot / name, name)
    with pytest.raises(ValueError, match="Unsafe tracked Registry snapshot"):
        release.inspect_archive(registry_snapshot)


def test_registry_workflow_gates_and_scopes_publish():
    workflow = (ROOT / ".github/workflows/publish-comfy.yml").read_text()
    assert "workflow_dispatch:" in workflow
    assert "git archive HEAD:comfyui" in workflow
    assert "working-directory: ${{ runner.temp }}/nobg-registry" in workflow
    assert "python release.py preflight" in workflow
    assert "python release.py inspect" in workflow
    assert "REGISTRY_ACCESS_TOKEN: ${{ secrets.REGISTRY_ACCESS_TOKEN }}" in workflow
    assert "python release.py publish" in workflow
    assert "python release.py verify" in workflow
    assert "python -m pip install comfy-cli==1.26.0" in workflow


def test_registry_version_changelog_and_status(monkeypatch):
    release = _release_module()
    notes = release.release_notes(PACK)
    response = {"changelog": notes, "status": "NodeVersionStatusPending"}
    read = mock.Mock(return_value=response)
    monkeypatch.setattr(release, "read_version", read)
    release.verify(PACK)
    read.assert_called_once_with(
        f"{release.PUBLIC_API}/{release.metadata(PACK)['version']}"
    )
    with pytest.raises(RuntimeError, match="still pending"):
        release.verify(PACK, require_active=True)
    response["status"] = "NodeVersionStatusFlagged"
    with pytest.raises(RuntimeError, match="needs investigation"):
        release.verify(PACK)
    response["status"] = "NodeVersionStatusActive"
    response["changelog"] = "wrong release notes"
    with pytest.raises(RuntimeError, match="did not match"):
        release.verify(PACK)


def test_registry_publish_passes_notes_to_cli_and_reads_back(
    monkeypatch, registry_snapshot
):
    release = _release_module()
    notes = release.release_notes(registry_snapshot)
    token = "private-registry-token"
    monkeypatch.setenv("REGISTRY_ACCESS_TOKEN", token)
    run = mock.Mock(return_value=types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(release.subprocess, "run", run)
    response = {"changelog": notes, "status": "NodeVersionStatusActive"}
    read = mock.Mock(return_value=response)
    monkeypatch.setattr(release, "read_version", read)

    release.publish(registry_snapshot)

    run.assert_called_once()
    args, kwargs = run.call_args
    assert args == (
        [
            "comfy",
            "--skip-prompt",
            "--no-enable-telemetry",
            "node",
            "publish",
        ],
    )
    assert kwargs["env"]["COMFY_NODE_CHANGELOG"] == notes
    assert kwargs["input"] == token + "\n"
    assert token not in str(args)
    assert kwargs["cwd"] == registry_snapshot
    assert kwargs["capture_output"] is True
    read.assert_called_once_with(
        f"{release.PUBLIC_API}/{release.metadata(registry_snapshot)['version']}"
    )
    assert token not in str(read.call_args)

    response["changelog"] = "incorrect notes"
    with pytest.raises(RuntimeError, match="did not match"):
        release.publish(registry_snapshot)


def test_registry_publish_failure_withholds_token_and_output(
    monkeypatch, registry_snapshot
):
    release = _release_module()
    token = "private-registry-token"
    monkeypatch.setenv("REGISTRY_ACCESS_TOKEN", token)
    monkeypatch.setattr(
        release.subprocess,
        "run",
        mock.Mock(return_value=types.SimpleNamespace(returncode=1, stderr=token)),
    )
    read = mock.Mock()
    monkeypatch.setattr(release, "read_version", read)
    with pytest.raises(RuntimeError, match="CLI output withheld") as exc:
        release.publish(registry_snapshot)
    assert token not in str(exc.value)
    read.assert_not_called()


def test_registry_publish_rejects_changed_snapshot_after_cli(
    monkeypatch, registry_snapshot
):
    release = _release_module()
    monkeypatch.setenv("REGISTRY_ACCESS_TOKEN", "secret")

    def mutate(*args, **kwargs):
        (registry_snapshot / "README.md").write_text("changed after inspection")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(release.subprocess, "run", mutate)
    read = mock.Mock()
    monkeypatch.setattr(release, "read_version", read)
    with pytest.raises(RuntimeError, match="snapshot changed"):
        release.publish(registry_snapshot)
    read.assert_not_called()


def test_registry_publish_rejects_repacked_archive_tamper(
    monkeypatch, registry_snapshot
):
    release = _release_module()
    monkeypatch.setenv("REGISTRY_ACCESS_TOKEN", "secret")

    def mutate(*args, **kwargs):
        with ZipFile(registry_snapshot / "node.zip", "w") as archive:
            for name in release.ALLOWED_ARCHIVE:
                if name == "README.md":
                    archive.writestr(name, "tampered by CLI")
                else:
                    archive.write(registry_snapshot / name, name)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(release.subprocess, "run", mutate)
    read = mock.Mock()
    monkeypatch.setattr(release, "read_version", read)
    with pytest.raises(ValueError, match="differs from snapshot"):
        release.publish(registry_snapshot)
    read.assert_not_called()


@pytest.fixture
def comfy_nodes(monkeypatch, tmp_path):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.__dict__["models_dir"] = str(tmp_path / "models")
    paths = {}

    def add_model_folder_path(name, path):
        paths.setdefault(name, []).append(path)

    folder_paths.__dict__["add_model_folder_path"] = add_model_folder_path
    folder_paths.__dict__["get_folder_paths"] = lambda name: paths.get(name, [])

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    mm = types.ModuleType("comfy.model_management")
    utils = types.ModuleType("comfy.utils")
    comfy.__dict__.update(model_management=mm, utils=utils)
    mm.__dict__.update(
        get_torch_device=mock.Mock(return_value=torch.device("cpu")),
        unet_offload_device=mock.Mock(return_value=torch.device("cpu")),
        free_memory=mock.Mock(),
        soft_empty_cache=mock.Mock(),
        throw_exception_if_processing_interrupted=mock.Mock(),
    )
    utils.__dict__["ProgressBar"] = mock.MagicMock()

    for name, module in {
        "folder_paths": folder_paths,
        "comfy": comfy,
        "comfy.model_management": mm,
        "comfy.utils": utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.delitem(sys.modules, "nobg.comfyui.nodes", raising=False)
    nodes = importlib.import_module("nobg.comfyui.nodes")
    yield types.SimpleNamespace(nodes=nodes, paths=paths, mm=mm, utils=utils)
    sys.modules.pop("nobg.comfyui.nodes", None)


class FakeNet:
    def __init__(self):
        self.param = torch.nn.Parameter(torch.zeros(4))
        self.calls = []
        self.processor_calls = 0
        self.to_calls = []

    def parameters(self):
        return iter([self.param])

    def to(self, device):
        self.to_calls.append(device)
        return self

    def eval(self):
        return self

    def default_processor(self):
        self.processor_calls += 1
        return "processor"

    def predict(self, processor, images, *args, batch_size=1, **kwargs):
        self.calls.append((processor, images, args, batch_size, kwargs))
        return [torch.full(image.shape[:2], 0.25) for image in images]


def test_entrypoint_and_node_contract(comfy_nodes):
    nodes = comfy_nodes.nodes
    spec = importlib.util.spec_from_file_location(
        "nobg_comfyui_entrypoint",
        PACK / "__init__.py",
    )
    assert spec is not None and spec.loader is not None
    package = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package)
    assert package.NODE_CLASS_MAPPINGS == nodes.NODE_CLASS_MAPPINGS
    assert set(package.NODE_DISPLAY_NAME_MAPPINGS) == set(nodes.NODE_CLASS_MAPPINGS)
    assert nodes.NOBG_DIR in comfy_nodes.paths["nobg"]
    assert list(nodes.NoBGModelLoader.INPUT_TYPES()["required"]) == ["model"]
    download = nodes.NoBGDownloadModel
    download_inputs = download.INPUT_TYPES()
    assert list(download_inputs["required"]) == ["repo_id"]
    assert download_inputs["required"]["repo_id"][0] == "STRING"
    assert download_inputs["optional"]["revision"][0] == "STRING"
    assert download_inputs["optional"]["revision"][1]["default"] == ""
    assert download.RETURN_TYPES == ("NOBG_CHECKPOINT",)
    assert download.FUNCTION == "download"
    assert nodes.NODE_CLASS_MAPPINGS["NoBGDownloadModel"] is download
    assert nodes.NoBGModelLoader.INPUT_TYPES()["optional"]["checkpoint"][0] == (
        "NOBG_CHECKPOINT"
    )
    remove_inputs = nodes.NoBGRemoveBackground.INPUT_TYPES()
    assert list(remove_inputs["required"]) == [
        "nobg_model",
        "image",
        "refine_foreground",
    ]
    assert remove_inputs["optional"]["prompt"][0] == "STRING"
    assert remove_inputs["optional"]["prompt"][1]["default"] == ""
    assert remove_inputs["optional"]["prompt"][1]["defaultInput"] is False
    assert remove_inputs["optional"]["batch_size"][1]["default"] == 1
    assert nodes.NoBGRemoveBackground.RETURN_TYPES == ("IMAGE", "IMAGE", "MASK")
    assert nodes.NoBGRefineForeground.RETURN_TYPES == ("IMAGE", "IMAGE")


def test_local_discovery_and_loader(comfy_nodes, tmp_path, monkeypatch):
    nodes = comfy_nodes.nodes
    root = tmp_path / "models" / "nobg"
    local = root / "my-sam3"
    local.mkdir(parents=True)
    (local / "config.json").write_text(
        json.dumps({"vision_hidden_size": 1024, "default_prompt": "subject"})
    )
    (local / "model.safetensors").touch()
    (root / "incomplete").mkdir()
    missing_weights = root / "config-only"
    missing_weights.mkdir()
    (missing_weights / "config.json").write_text(
        json.dumps({"vision_hidden_size": 1024, "default_prompt": "subject"})
    )
    assert nodes._local_models() == ["local: my-sam3"]

    net = FakeNet()
    loader = mock.Mock(return_value=net)
    monkeypatch.setattr(nodes, "_load", loader)
    (loaded,) = nodes.NoBGModelLoader().load("local: my-sam3")
    loader.assert_called_once_with("sam3", str(local), "semantic", upstream_sam3=False)
    assert loaded.model is net
    assert loaded.device == torch.device("cpu")
    assert loaded.processor == loaded.processor == "processor"
    assert net.processor_calls == 1
    with pytest.raises(FileNotFoundError):
        nodes.NoBGModelLoader().load("local: missing")
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        nodes.NoBGModelLoader().load("local: config-only")


def test_local_metadata_and_duplicate_names(comfy_nodes, tmp_path, monkeypatch):
    nodes = comfy_nodes.nodes
    root = tmp_path / "models" / "nobg"
    second = tmp_path / "other"
    comfy_nodes.paths["nobg"].append(str(second))
    for path, config in (
        (root / "sam3-in-name", {"embed_dim": 192, "depths": [2, 2, 18, 2]}),
        (root / "duplicate", {"embed_dim": 192, "depths": [2, 2, 18, 2]}),
        (second / "duplicate", {"vision_hidden_size": 1024, "default_prompt": "x"}),
        (second / "literal (root 1)", {"embed_dim": 192, "depths": [2, 2, 18, 2]}),
        (second / "unknown", {"unexpected": True}),
        (second / "upstream", {"model_type": "sam3"}),
    ):
        path.mkdir(parents=True)
        (path / "config.json").write_text(json.dumps(config))
        (path / "model.safetensors").touch()
    choices = nodes._local_models()
    assert "local: @1:duplicate" in choices
    assert "local: @2:duplicate" in choices
    assert "local: sam3-in-name" in choices
    assert "local: literal (root 1)" in choices
    loader = mock.Mock(return_value=FakeNet())
    monkeypatch.setattr(nodes, "_load", loader)
    nodes.NoBGModelLoader().load("local: sam3-in-name")
    loader.assert_called_with(
        "birefnet", str(root / "sam3-in-name"), "semantic", upstream_sam3=False
    )
    nodes.NoBGModelLoader().load("local: @2:duplicate")
    loader.assert_called_with(
        "sam3", str(second / "duplicate"), "semantic", upstream_sam3=False
    )
    nodes.NoBGModelLoader().load("local: literal (root 1)")
    loader.assert_called_with(
        "birefnet", str(second / "literal (root 1)"), "semantic", upstream_sam3=False
    )
    with pytest.raises(ValueError, match="ambiguous"):
        nodes.NoBGModelLoader().load("local: duplicate")
    with pytest.raises(ValueError, match="unsupported local nobg model"):
        nodes.NoBGModelLoader().load("local: unknown")
    nodes.NoBGModelLoader().load("local: upstream")
    loader.assert_called_with(
        "sam3", str(second / "upstream"), "semantic", upstream_sam3=True
    )
    with pytest.raises(ValueError, match="unsupported nobg model"):
        nodes.NoBGModelLoader().load("unknown/repo")
    with pytest.raises(ValueError, match="invalid nobg model folder name"):
        nodes.NoBGModelLoader().load("local: ../other")
    with pytest.raises(ValueError, match="invalid nobg model folder name"):
        nodes.NoBGModelLoader().load("local: @1:..%2Fother")
    (second / "upstream" / "model.safetensors").unlink()
    assert "local: upstream" not in nodes._local_models()
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        nodes.NoBGModelLoader().load("local: upstream")


def test_every_local_choice_resolves_with_adversarial_names(comfy_nodes, tmp_path):
    nodes = comfy_nodes.nodes
    roots = [tmp_path / "models" / "nobg", tmp_path / "other"]
    comfy_nodes.paths["nobg"].append(str(roots[1]))
    names = (
        "base",
        "base (root 1)",
        "base (root 1) (root 1)",
        "@1:base",
        "@1:%40base",
        "percent%20name",
        "literal (root 1)",
    )
    expected_paths = set()
    for root in roots:
        for name in names:
            path = root / name
            path.mkdir(parents=True)
            (path / "config.json").write_text("{}")
            (path / "model.safetensors").touch()
            expected_paths.add(str(path))
    # Unique ordinary names keep the old user-facing label.
    unique = roots[0] / "ordinary"
    unique.mkdir()
    (unique / "config.json").write_text("{}")
    (unique / "model.safetensors").touch()
    expected_paths.add(str(unique))
    choices = nodes._local_models()
    assert "local: ordinary" in choices
    assert len(choices) == len(expected_paths) == len(set(choices))
    assert {
        nodes._resolve_local(choice[len(nodes.LOCAL_PREFIX) :]) for choice in choices
    } == expected_paths


def test_sam3_loader_uses_origin_only_for_upstream(comfy_nodes, monkeypatch):
    nodes = comfy_nodes.nodes
    import nobg

    origin = mock.Mock(return_value=FakeNet())
    pretrained = mock.Mock(return_value=FakeNet())
    monkeypatch.setattr(nobg.Sam3, "from_origin", origin)
    monkeypatch.setattr(nobg.Sam3, "from_pretrained", pretrained)
    nodes._load("sam3", "/local/sam3", "max")
    pretrained.assert_called_once_with("/local/sam3")
    origin.assert_not_called()
    nodes._load("sam3", "facebook/sam3", "semantic")
    origin.assert_called_once_with("facebook/sam3", aggregate="semantic")


def test_sam3_origin_for_upstream_snapshot_and_pretrained_for_nobg(
    comfy_nodes, monkeypatch, tmp_path
):
    nodes = comfy_nodes.nodes
    import nobg

    origin = mock.Mock(return_value=FakeNet())
    pretrained = mock.Mock(return_value=FakeNet())
    monkeypatch.setattr(nobg.Sam3, "from_origin", origin)
    monkeypatch.setattr(nobg.Sam3, "from_pretrained", pretrained)
    upstream = tmp_path / "facebook-snapshot"
    upstream.mkdir()
    (upstream / "config.json").write_text(json.dumps({"model_type": "sam3_video"}))
    (upstream / "model.safetensors").touch()
    nobg_checkpoint = tmp_path / "nobg-checkpoint"
    nobg_checkpoint.mkdir()
    (nobg_checkpoint / "config.json").write_text(
        json.dumps({"vision_hidden_size": 1024, "default_prompt": "subject"})
    )
    nodes._load("sam3", str(upstream), "max")
    origin.assert_called_once_with(str(upstream), aggregate="max")
    loaded = nodes._load("sam3", str(nobg_checkpoint), "semantic")
    pretrained.assert_called_once_with(str(nobg_checkpoint))
    assert loaded._tokenizer_source == str(nobg_checkpoint)

    snapshot = mock.Mock(return_value=str(upstream))
    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    nodes.NoBGModelLoader().load("facebook/sam3", download_to_models_dir=True)
    origin.assert_called_with(str(upstream), aggregate="semantic")
    snapshot.assert_called_once()


def test_downloaded_snapshot_requires_weights(comfy_nodes, monkeypatch, tmp_path):
    nodes = comfy_nodes.nodes
    snapshot_dir = tmp_path / "incomplete-snapshot"
    snapshot_dir.mkdir()
    (snapshot_dir / "config.json").write_text(json.dumps({"model_type": "sam3"}))
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        mock.Mock(return_value=str(snapshot_dir)),
    )
    load = mock.Mock(return_value=FakeNet())
    monkeypatch.setattr(nodes, "_load", load)
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        nodes.NoBGModelLoader().load("facebook/sam3", download_to_models_dir=True)
    load.assert_not_called()


@pytest.fixture
def mocked_hub(monkeypatch):
    """No requests: simulate metadata and an actual snapshot in its local_dir."""
    import huggingface_hub

    model_info = mock.Mock(
        side_effect=lambda repo_id, **kwargs: types.SimpleNamespace(
            id=repo_id,
            sha="0123456789abcdef0123456789abcdef01234567",
            siblings=[
                types.SimpleNamespace(rfilename="config.json"),
                types.SimpleNamespace(rfilename="model.safetensors"),
            ],
        )
    )

    def populate(repo_id, **kwargs):
        destination = Path(kwargs["local_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text(
            json.dumps(
                {"embed_dim": 192, "depths": [2, 2, 18, 2]}
                if repo_id != "org/sam3"
                else {"model_type": "sam3"}
            )
        )
        (destination / "model.safetensors").write_bytes(b"weights")
        return str(destination)

    snapshot = mock.Mock(side_effect=populate)
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", model_info)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    return types.SimpleNamespace(info=model_info, snapshot=snapshot)


def _checkpoint_path(checkpoint):
    """The typed handoff may be a dataclass, dict, or path-bearing wrapper."""
    if isinstance(checkpoint, (str, os.PathLike)):
        return Path(checkpoint)
    if isinstance(checkpoint, dict):
        for key in ("path", "source", "local_dir"):
            if key in checkpoint:
                return Path(checkpoint[key])
    for key in ("path", "source", "local_dir"):
        value = getattr(checkpoint, key, None)
        if value is not None:
            return Path(value)
    raise AssertionError("download must return a path-bearing checkpoint")


def test_download_identifies_remote_config_and_loader_handoff(
    comfy_nodes, mocked_hub, monkeypatch, tmp_path
):
    nodes = comfy_nodes.nodes
    net = FakeNet()
    load = mock.Mock(return_value=net)
    monkeypatch.setattr(nodes, "_load", load)
    (checkpoint,) = nodes.NoBGDownloadModel().download("org/sam3")
    assert isinstance(checkpoint, nodes.NoBGCheckpoint)
    path = _checkpoint_path(checkpoint)
    assert path.is_dir()
    assert path.is_relative_to(tmp_path / "models" / "nobg")
    assert (path / "config.json").is_file()
    assert (path / "model.safetensors").is_file()
    assert path.name != "sam3"  # Owner-qualified, not a bare basename.
    assert "org" in str(path)
    assert mocked_hub.info.called
    assert mocked_hub.snapshot.called
    assert mocked_hub.snapshot.call_args.kwargs["revision"] == (
        "0123456789abcdef0123456789abcdef01234567"
    )
    (model,) = nodes.NoBGModelLoader().load("feyninc/FeyNobg", checkpoint=checkpoint)
    assert model.architecture == "sam3"
    assert Path(model.source) == path
    load.assert_called_once_with("sam3", str(path), "semantic", upstream_sam3=True)
    assert model.model is net


def test_download_namespaces_owners_and_reuses_same_revision(
    comfy_nodes, mocked_hub, tmp_path
):
    nodes = comfy_nodes.nodes
    download = nodes.NoBGDownloadModel()
    (first,) = download.download("owner-one/shared")
    (second,) = download.download("owner-two/shared")
    first_path, second_path = _checkpoint_path(first), _checkpoint_path(second)
    assert first_path != second_path
    assert first_path.is_relative_to(tmp_path / "models" / "nobg")
    assert second_path.is_relative_to(tmp_path / "models" / "nobg")
    assert (first_path / "model.safetensors").read_bytes() == b"weights"
    assert (second_path / "model.safetensors").read_bytes() == b"weights"
    snapshot_count = mocked_hub.snapshot.call_count
    info_count = mocked_hub.info.call_count
    (again,) = download.download("owner-one/shared")
    assert _checkpoint_path(again) == first_path
    assert mocked_hub.snapshot.call_count == snapshot_count
    assert mocked_hub.info.call_count == info_count
    assert any(
        nodes._resolve_local(choice[len(nodes.LOCAL_PREFIX) :]) == str(first_path)
        for choice in nodes._local_models()
        if choice.startswith(nodes.LOCAL_PREFIX)
    )


def test_download_distinct_revisions_keep_distinct_snapshots(comfy_nodes, mocked_hub):
    download = comfy_nodes.nodes.NoBGDownloadModel()
    (main,) = download.download("org/model", revision="main")
    (branch,) = download.download("org/model", revision="feature/experiment")
    assert _checkpoint_path(main) != _checkpoint_path(branch)
    assert mocked_hub.snapshot.call_count == 2
    assert all(_checkpoint_path(item).is_dir() for item in (main, branch))
    (again,) = download.download("org/model", revision="feature/experiment")
    assert _checkpoint_path(again) == _checkpoint_path(branch)
    assert mocked_hub.snapshot.call_count == 2


@pytest.mark.parametrize(
    ("config", "weights", "error"),
    (
        ({"model_type": "unrelated"}, True, ValueError),
        ({"unexpected": True}, True, ValueError),
        ({"embed_dim": 192, "depths": [2, 2, 18, 2]}, False, FileNotFoundError),
    ),
)
def test_download_rejects_unsupported_or_incomplete_snapshots(
    comfy_nodes, mocked_hub, config, weights, error
):
    nodes = comfy_nodes.nodes

    def snapshot(repo_id, **kwargs):
        path = Path(kwargs["local_dir"])
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(config))
        if weights:
            (path / "model.safetensors").touch()
        return str(path)

    mocked_hub.snapshot.side_effect = snapshot
    with pytest.raises(error):
        nodes.NoBGDownloadModel().download("org/invalid")
    assert nodes._local_models() == []


def test_interrupted_download_does_not_expose_staging(
    comfy_nodes, mocked_hub, tmp_path
):
    nodes = comfy_nodes.nodes
    staged = []

    def interrupt(repo_id, **kwargs):
        path = Path(kwargs["local_dir"])
        staged.append(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps({"embed_dim": 192, "depths": [2, 2, 18, 2]})
        )
        (path / "model.safetensors").write_bytes(b"partial")
        raise InterruptedError("download interrupted")

    mocked_hub.snapshot.side_effect = interrupt
    with pytest.raises(InterruptedError, match="interrupted"):
        nodes.NoBGDownloadModel().download("org/interrupted")
    assert nodes._local_models() == []
    assert not any(
        (path / "model.safetensors").exists()
        for path in (tmp_path / "models" / "nobg").iterdir()
        if path.is_dir()
    )
    assert staged


def test_download_refuses_destination_symlink_escape(comfy_nodes, mocked_hub, tmp_path):
    nodes = comfy_nodes.nodes
    root = tmp_path / "models" / "nobg"
    external = tmp_path / "outside"
    external.mkdir()
    # Pre-existing symlink at the model root must never become a write destination.
    root.mkdir(parents=True, exist_ok=True)
    (root / "org").symlink_to(external, target_is_directory=True)
    with pytest.raises((ValueError, OSError, FileExistsError)):
        nodes.NoBGDownloadModel().download("org/escape")
    assert list(external.iterdir()) == []
    mocked_hub.snapshot.assert_not_called()


@pytest.mark.parametrize(
    "repo_id",
    ("", "not-qualified", "../escape", "owner/../escape", "owner/model/extra"),
)
def test_download_rejects_malformed_repo_ids_without_network(
    comfy_nodes, mocked_hub, repo_id
):
    with pytest.raises(ValueError):
        comfy_nodes.nodes.NoBGDownloadModel().download(repo_id)
    mocked_hub.info.assert_not_called()
    mocked_hub.snapshot.assert_not_called()


def test_download_propagates_gated_authorization_without_indexing(
    comfy_nodes, mocked_hub
):
    from huggingface_hub.errors import GatedRepoError

    response = mock.Mock(
        status_code=403, headers={}, url="https://huggingface.co/org/gated"
    )
    mocked_hub.info.side_effect = GatedRepoError(
        "Access to gated repository denied", response=response
    )
    with pytest.raises((ValueError, GatedRepoError), match="gated|sign in") as exc:
        comfy_nodes.nodes.NoBGDownloadModel().download("org/gated")
    if not isinstance(exc.value, GatedRepoError):
        assert isinstance(exc.value.__cause__, GatedRepoError)
    mocked_hub.snapshot.assert_not_called()
    assert comfy_nodes.nodes._local_models() == []


def test_loader_trusts_only_exact_local_checkpoint_reference(
    comfy_nodes, mocked_hub, monkeypatch, tmp_path
):
    nodes = comfy_nodes.nodes
    (checkpoint,) = nodes.NoBGDownloadModel().download("org/trusted")
    source = _checkpoint_path(checkpoint)
    load = mock.Mock(return_value=FakeNet())
    monkeypatch.setattr(nodes, "_load", load)
    (model,) = nodes.NoBGModelLoader().load(
        "unknown/not-a-preset", checkpoint=checkpoint
    )
    assert Path(model.source) == source
    assert model.architecture == "birefnet"
    load.assert_called_once_with(
        "birefnet", str(source), "semantic", upstream_sam3=False
    )
    for untrusted in ("org/trusted", "local: trusted", str(source)):
        with pytest.raises((TypeError, ValueError), match="NoBG Download Model"):
            nodes.NoBGModelLoader().load("unknown/not-a-preset", checkpoint=untrusted)
    forged = nodes.NoBGCheckpoint(
        str(source), checkpoint.architecture, "different/owner", checkpoint.commit
    )
    with pytest.raises(ValueError, match="provenance"):
        nodes.NoBGModelLoader().load("unknown/not-a-preset", checkpoint=forged)
    assert load.call_count == 1
    assert source.is_relative_to(tmp_path / "models" / "nobg")


def test_model_placement_and_offload(comfy_nodes):
    nodes = comfy_nodes.nodes
    net = FakeNet()
    model = nodes.NoBGModel(net, "birefnet", "local", torch.device("meta"))
    model.to_device()
    comfy_nodes.mm.free_memory.assert_called_once_with(
        int(16 * 1.2) + 256 * 1024 * 1024, torch.device("meta")
    )
    assert net.to_calls == [torch.device("meta")]
    model.offload()
    comfy_nodes.mm.soft_empty_cache.assert_not_called()  # fake net remains on CPU


def test_remove_birefnet_batch_and_rgba(comfy_nodes):
    nodes = comfy_nodes.nodes
    net = FakeNet()
    model = nodes.NoBGModel(net, "birefnet", "local", torch.device("cpu"))
    images = torch.rand(3, 8, 6, 4)
    cutout, foreground, mask = nodes.NoBGRemoveBackground().remove(
        model, images, batch_size=2, keep_model_loaded=True
    )
    assert cutout.shape == (3, 8, 6, 4)
    assert torch.equal(foreground, images[..., :3])
    assert torch.allclose(mask, torch.full((3, 8, 6), 0.25))
    assert mask.dtype == torch.float32
    assert mask.device == images.device == cutout.device == foreground.device
    assert torch.equal(cutout[..., 3], mask)
    assert [call[3] for call in net.calls] == [2, 1]
    assert all(call[4] == {"return_type": "alpha"} for call in net.calls)
    assert net.calls[0][1][0].dtype.name == "uint8"
    comfy_nodes.mm.throw_exception_if_processing_interrupted.assert_called()


def test_remove_rejects_invalid_batch_before_model_use(comfy_nodes):
    nodes = comfy_nodes.nodes
    model = nodes.NoBGModel(FakeNet(), "birefnet", "local", torch.device("cpu"))
    for images, batch_size, message in (
        (torch.rand(1, 8, 6, 3), 0, "batch_size"),
        (torch.rand(1, 8, 6, 3), -1, "batch_size"),
        (torch.empty(0, 8, 6, 3), 1, "non-empty"),
        (torch.rand(1, 8, 6, 2), 1, "C >= 3"),
    ):
        with pytest.raises(ValueError, match=message):
            nodes.NoBGRemoveBackground().remove(model, images, batch_size=batch_size)
    assert not model.model.calls


def test_remove_streams_chunk_conversion(comfy_nodes, monkeypatch):
    nodes = comfy_nodes.nodes
    model = nodes.NoBGModel(FakeNet(), "birefnet", "local", torch.device("cpu"))
    original = model.model.predict
    conversion_count = 0
    original_numpy = torch.Tensor.numpy

    def count_numpy(tensor):
        nonlocal conversion_count
        conversion_count += 1
        return original_numpy(tensor)

    def predict(*args, **kwargs):
        # The later frames must not have been converted before the first chunk.
        assert conversion_count == sum(
            len(call[1]) for call in model.model.calls
        ) + len(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "numpy", count_numpy)
    monkeypatch.setattr(model.model, "predict", predict)
    nodes.NoBGRemoveBackground().remove(model, torch.rand(5, 8, 6, 3), batch_size=2)
    assert conversion_count == 5


def test_remove_sam3_prompt_boxes_and_failure_offload(comfy_nodes, monkeypatch):
    nodes = comfy_nodes.nodes
    net = FakeNet()
    model = nodes.NoBGModel(net, "sam3", "local", torch.device("cpu"))
    monkeypatch.setattr(model, "offload", mock.Mock())
    nodes.NoBGRemoveBackground().remove(
        model,
        torch.rand(2, 8, 6, 3),
        prompt=" dog ",
        boxes="1,2,3,4",
        score_threshold=0.4,
    )
    assert net.calls[0][2] == ("dog", [[[1.0, 2.0, 3.0, 4.0]]])
    assert net.calls[0][4] == {"return_type": "alpha", "score_threshold": 0.4}
    assert model.offload.call_count == 1
    with pytest.raises(ValueError, match="must have 4 values"):
        nodes.NoBGRemoveBackground().remove(
            model, torch.rand(1, 8, 6, 3), boxes="1,2,3"
        )

    net.predict = mock.Mock(side_effect=RuntimeError("inference failed"))
    with pytest.raises(RuntimeError, match="inference failed"):
        nodes.NoBGRemoveBackground().remove(model, torch.rand(1, 8, 6, 3))
    assert model.offload.call_count == 2


def test_refine_node_broadcasts_mask_and_rejects_mismatch(comfy_nodes, monkeypatch):
    nodes = comfy_nodes.nodes
    refine = mock.Mock(side_effect=lambda image, alpha, r: image * 0.5)
    monkeypatch.setattr("nobg.utils.refine_foreground", refine)
    images = torch.rand(2, 8, 6, 4)
    foreground, cutout = nodes.NoBGRefineForeground().refine(
        images, torch.ones(4, 3), 12
    )
    assert torch.allclose(foreground, images[..., :3] * 0.5)
    assert cutout.shape == (2, 8, 6, 4)
    assert torch.allclose(cutout[..., 3], torch.ones(2, 8, 6))
    assert refine.call_count == 2
    for call in refine.call_args_list:
        assert call.args[1].shape == (8, 6)
        assert torch.allclose(call.args[1], torch.ones(8, 6))
    with pytest.raises(ValueError, match="got 3 masks for 2 images"):
        nodes.NoBGRefineForeground().refine(images, torch.ones(3, 8, 6), 12)


def test_refine_clamps_and_resizes_matte_before_refining(comfy_nodes, monkeypatch):
    nodes = comfy_nodes.nodes
    seen = []

    def refine(image, alpha, r):
        seen.append(alpha.clone())
        assert alpha.shape == image.shape[-2:]
        assert alpha.min() >= 0 and alpha.max() <= 1
        return image

    monkeypatch.setattr("nobg.utils.refine_foreground", refine)
    image = torch.rand(1, 8, 6, 3)
    foreground, cutout = nodes.NoBGRefineForeground().refine(
        image, torch.tensor([[-1.0, 3.0], [3.0, -1.0]]), 12
    )
    assert torch.equal(foreground, image)
    assert torch.allclose(cutout[0, ..., 3], seen[0])
