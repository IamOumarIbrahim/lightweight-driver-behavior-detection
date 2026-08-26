"""Host-side launcher for the isolated RTMDet and EfficientDet CUDA image."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tools.benchmark.paths import REPO_ROOT
from tools.benchmark.protocol import ProtocolError, load_backends


def backend_spec() -> dict[str, Any]:
    return load_backends()["additional_models"]


def image_tag() -> str:
    return str(backend_spec()["image"])


def container_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ProtocolError(f"Container path must be inside the repository: {resolved}") from exc
    return "/workspace/" + relative.as_posix()


def ensure_image() -> None:
    if shutil.which("docker") is None:
        raise ProtocolError("Docker Desktop is required for RTMDet-Tiny and EfficientDet-D1")
    result = subprocess.run(
        ["docker", "image", "inspect", image_tag()],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        raise ProtocolError(
            "The pinned additional-model image is unavailable; start Docker Desktop and "
            "run scripts/setup/02_setup_backends.bat"
        )


def command(*arguments: str, gpu: bool = True) -> list[str]:
    result = [
        "docker",
        "run",
        "--rm",
        "--shm-size",
        "8g",
        "--volume",
        f"{REPO_ROOT}:/workspace",
    ]
    if gpu:
        result.extend(["--gpus", "device=0"])
    result.extend([image_tag(), *arguments])
    return result


def run(*arguments: str, gpu: bool = True) -> None:
    ensure_image()
    subprocess.run(command(*arguments, gpu=gpu), cwd=REPO_ROOT, check=True)


def infer(
    *,
    model: str,
    config: Path,
    checkpoint: Path,
    ground_truth: Path,
    output: Path,
    profile: bool,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "infer",
        "--model",
        model,
        "--config",
        container_path(config),
        "--checkpoint",
        container_path(checkpoint),
        "--ground-truth",
        container_path(ground_truth),
        "--output",
        container_path(output),
    ]
    if profile:
        args.append("--profile")
    run(*args)
    return json.loads(output.read_text(encoding="utf-8"))
