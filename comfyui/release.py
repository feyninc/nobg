"""Checks and Registry publication for the separate ComfyUI node distribution.

Run from a clean, git-tracked snapshot of comfyui/, never from the repo root.
"""

import importlib.util
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import types
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import tomllib  # ty: ignore[unresolved-import] -- this distribution requires Python 3.11+

PUBLISHER = "chonk-lain"
NODE = "nobg"
PUBLIC_API = f"https://api.comfy.org/nodes/{NODE}/versions"
ALLOWED_ARCHIVE = {
    "__init__.py",
    "pyproject.toml",
    # ComfyUI-Manager installs dependencies only from requirements.txt; it
    # ignores [project] dependencies, so without this the nobg library is
    # never installed and the entrypoint fails with ModuleNotFoundError.
    "requirements.txt",
    "README.md",
    "LICENSE",
    ".comfyignore",
    "example_workflows/NoBG_remove_background.json",
    "example_workflows/NoBG_remove_background.jpg",
    "example_workflows/NoBG_sam3_text_prompt.json",
    "example_workflows/NoBG_sam3_text_prompt.jpg",
}
EXAMPLES = {name for name in ALLOWED_ARCHIVE if name.startswith("example_workflows/")}
SNAPSHOT_FILES = ALLOWED_ARCHIVE | {"CHANGELOG.md", "release.py"}


def metadata(directory: Path) -> dict:
    data = tomllib.loads((directory / "pyproject.toml").read_text())
    assert data["project"]["name"] == NODE
    assert data["tool"]["comfy"]["PublisherId"] == PUBLISHER
    assert data["tool"]["comfy"]["DisplayName"] == "NoBG"
    minimum = [
        dep.removeprefix("nobg>=")
        for dep in data["project"]["dependencies"]
        if dep.startswith("nobg>=")
    ]
    assert len(minimum) == 1
    assert re.fullmatch(r"\d+\.\d+\.\d+", minimum[0])
    assert tuple(map(int, minimum[0].split("."))) >= (0, 3, 2)
    requirements = (directory / "requirements.txt").read_text().split()
    assert requirements == data["project"]["dependencies"], (
        "requirements.txt must match [project] dependencies"
    )
    return data["project"]


def release_notes(directory: Path) -> str:
    version = metadata(directory)["version"]
    changelog = (directory / "CHANGELOG.md").read_text()
    match = re.search(
        rf"(?m)^## {re.escape(version)}\s*$\n(.*?)(?=^## |\Z)",
        changelog,
        re.DOTALL | re.MULTILINE,
    )
    if not match or not match[1].strip():
        raise ValueError(f"Missing nonempty Registry release notes for {version}")
    return match[1].strip()


def check_pypi_wheel():
    """Require an installable published wheel containing the ComfyUI integration."""
    dependencies = metadata(Path.cwd())["dependencies"]
    required = [item for item in dependencies if item.startswith("nobg>=")]
    if len(required) != 1:
        raise ValueError("Exactly one minimum nobg PyPI dependency is required")
    version = required[0].removeprefix("nobg>=")
    try:
        with urllib.request.urlopen(
            f"https://pypi.org/pypi/nobg/{version}/json"
        ) as response:
            release = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Publish nobg {version} to PyPI first") from exc
    wheels = [item for item in release["urls"] if item["filename"].endswith(".whl")]
    if not wheels:
        raise RuntimeError(f"nobg {version} has no published wheel")
    with urllib.request.urlopen(wheels[0]["url"]) as response:
        wheel = zipfile.ZipFile(io.BytesIO(response.read()))
    if "nobg/comfyui/nodes.py" not in wheel.namelist():
        raise RuntimeError(
            f"Published nobg {version} wheel lacks nobg/comfyui/nodes.py"
        )
    if "nobg/comfyui/__init__.py" not in wheel.namelist():
        raise RuntimeError(
            f"Published nobg {version} wheel lacks nobg/comfyui/__init__.py"
        )


def snapshot(directory: Path) -> dict[str, bytes]:
    """Capture the complete, regular-file-only source set used by comfy-cli."""
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if ".git" not in path.relative_to(directory).parts
        and path.name != "node.zip"
        and "__pycache__" not in path.relative_to(directory).parts
        and ".ruff_cache" not in path.relative_to(directory).parts
        and path.suffix != ".pyc"
        and (path.is_file() or path.is_symlink() or not path.is_dir())
    }
    if actual != SNAPSHOT_FILES:
        raise ValueError(f"Unsafe or incomplete Registry snapshot: {sorted(actual)}")
    files = {}
    for name in sorted(actual):
        path = directory / name
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"Nonregular Registry snapshot file: {name}")
        files[name] = path.read_bytes()
    return files


def inspect_archive(directory: Path, expected: dict[str, bytes] | None = None) -> bytes:
    """Validate ZIP members against the snapshot without extracting untrusted files."""
    if expected is None:
        expected = snapshot(directory)
    if (directory / ".git").exists():
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "--cached"],
            cwd=directory,
            capture_output=True,
            check=True,
        ).stdout
        staged = {os.fsdecode(name) for name in tracked.split(b"\0") if name}
        unexpected = staged - SNAPSHOT_FILES
        if unexpected:
            raise ValueError(
                f"Unsafe tracked Registry snapshot files: {sorted(unexpected)}"
            )
    archive_bytes = (directory / "node.zip").read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        members = set(names)
        if len(names) != len(members):
            raise ValueError("Duplicate Registry archive member")
        if any(
            info.is_dir()
            or (
                info.create_system == 3
                and stat.S_IFMT(info.external_attr >> 16) not in (0, stat.S_IFREG)
            )
            for info in infos
        ):
            raise ValueError("Nonregular Registry archive member")
        if members != ALLOWED_ARCHIVE:
            raise ValueError(
                f"Unsafe or incomplete Registry archive: {sorted(members)}"
            )
        for info in infos:
            if archive.read(info) != expected[info.filename]:
                raise ValueError(
                    f"Registry archive differs from snapshot: {info.filename}"
                )
        if tomllib.loads(archive.read("pyproject.toml").decode())[
            "project"
        ] != metadata(directory):
            raise ValueError("Registry archive metadata differs from release metadata")
        with tempfile.TemporaryDirectory() as temp:
            # Only known, byte-verified regular files are written for the smoke test.
            (Path(temp) / "__init__.py").write_bytes(expected["__init__.py"])
            fake_nodes = types.ModuleType("nobg.comfyui.nodes")
            fake_nodes.__dict__["NODE_CLASS_MAPPINGS"] = {"NoBG": object}
            fake_nodes.__dict__["NODE_DISPLAY_NAME_MAPPINGS"] = {"NoBG": "NoBG"}
            original = sys.modules.get("nobg.comfyui.nodes")
            try:
                sys.modules["nobg.comfyui.nodes"] = fake_nodes
                spec = importlib.util.spec_from_file_location(
                    "nobg_registry_smoke", Path(temp) / "__init__.py"
                )
                assert spec and spec.loader
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                assert (
                    module.NODE_CLASS_MAPPINGS
                    is fake_nodes.__dict__["NODE_CLASS_MAPPINGS"]
                )
            finally:
                if original is None:
                    sys.modules.pop("nobg.comfyui.nodes", None)
                else:
                    sys.modules["nobg.comfyui.nodes"] = original
    print(f"Inspected Registry archive: {', '.join(sorted(members))}")
    return archive_bytes


def read_version(url: str) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def publish(directory: Path):
    token = os.environ.get("REGISTRY_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("REGISTRY_ACCESS_TOKEN is required")
    notes = release_notes(directory)
    expected = snapshot(directory)
    packed = inspect_archive(directory, expected)
    # comfy-cli 1.26.0 prompts for the API key (hide_input=True) when --token
    # is absent; --skip-prompt does not disable this explicit Typer prompt.
    # stdin avoids exposing the token in argv / process listings.
    result = subprocess.run(
        [
            "comfy",
            "--skip-prompt",
            "--no-enable-telemetry",
            "node",
            "publish",
        ],
        cwd=directory,
        env={**os.environ, "COMFY_NODE_CHANGELOG": notes},
        input=token + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "Comfy Registry publish failed; CLI output withheld to protect the token"
        )
    # CLI repacks and overwrites node.zip during publish; compare content, not
    # ZIP bytes (entry timestamps can change). This check is after upload.
    if snapshot(directory) != expected:
        raise RuntimeError("Registry snapshot changed during publish")
    inspect_archive(directory, expected)
    if (directory / "node.zip").read_bytes() != packed:
        print(
            "::notice::CLI repacked Registry archive; verified member bytes, not ZIP bytes"
        )
    # comfy-cli publishes COMFY_NODE_CHANGELOG with the version itself.
    verify(directory)


def verify(directory: Path, require_active: bool = False):
    version = metadata(directory)["version"]
    notes = release_notes(directory)
    published = read_version(f"{PUBLIC_API}/{version}")
    if (published.get("changelog") or "").strip() != notes:
        raise RuntimeError(
            "Registry release changelog did not match the intended notes"
        )
    status = published.get("status")
    if status in {
        "NodeVersionStatusFlagged",
        "NodeVersionStatusBanned",
        "NodeVersionStatusDeleted",
    }:
        raise RuntimeError(f"Registry version {version} needs investigation: {status}")
    if status == "NodeVersionStatusPending":
        if require_active:
            raise RuntimeError(
                f"Registry version {version} is still pending; verify again later"
            )
        print(
            f"::warning::Registry version {version} is pending scan; rerun with verify_only later"
        )
    elif status == "NodeVersionStatusActive":
        print(f"Verified {PUBLISHER}/{NODE} {version}: active with release notes")
    else:
        raise RuntimeError(f"Unknown Registry version status for {version}: {status}")


if __name__ == "__main__":
    root = Path.cwd()
    try:
        if sys.argv[1:] == ["preflight"]:
            release_notes(root)
            check_pypi_wheel()
            print(
                "Published PyPI integration wheel and Registry release notes verified"
            )
        elif sys.argv[1:] == ["inspect"]:
            release_notes(root)
            inspect_archive(root)
        elif sys.argv[1:] == ["publish"]:
            publish(root)
        elif sys.argv[1:] == ["verify"]:
            verify(root, require_active=True)
        else:
            raise ValueError("Usage: release.py preflight|inspect|publish|verify")
    except (AssertionError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        sys.exit(str(exc))
