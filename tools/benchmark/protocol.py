"""Load and validate the machine-readable RGB and NIR protocols."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIGS_ROOT, NIR_RATIOS, NIR_SEED, RGB_SEEDS, repo_path

TRAINING_SEEDS = RGB_SEEDS
BACKEND_CONFIG = CONFIGS_ROOT / "backends.yaml"


class ProtocolError(RuntimeError):
    """Raised when a frozen protocol invariant is violated."""


def resolve_repo_path(value: str | Path) -> Path:
    return repo_path(value)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_repo_path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ProtocolError(f"Expected a YAML mapping in {path}")
    return data


def load_protocol(track: str = "RGB") -> dict[str, Any]:
    normalized = track.upper()
    if normalized not in {"RGB", "NIR"}:
        raise ProtocolError(f"Unknown benchmark track: {track}")
    return load_yaml(CONFIGS_ROOT / normalized / "protocol.yaml")


def load_backends() -> dict[str, Any]:
    return load_yaml(BACKEND_CONFIG)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with resolve_repo_path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_authoritative_fingerprints(
    protocol: dict[str, Any] | None = None, track: str = "RGB"
) -> dict[str, str]:
    protocol = protocol or load_protocol(track)
    dataset = protocol["dataset"]
    actual: dict[str, str] = {}
    for name in ("annotations", "splits", "source_videos"):
        checksum_key = f"{name}_sha256"
        if name not in dataset or checksum_key not in dataset:
            continue
        actual[checksum_key] = sha256_file(dataset[name])
        expected = str(dataset[checksum_key]).lower()
        if actual[checksum_key] != expected:
            raise ProtocolError(
                f"{track} {name} fingerprint mismatch: expected {expected}, got {actual[checksum_key]}"
            )
    return actual


def model_spec(
    model_id: str, track: str = "RGB"
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_protocol(track)
    model = next((item for item in protocol["models"] if item["id"] == model_id), None)
    if model is None:
        raise ProtocolError(f"Unknown frozen model: {model_id}")
    backends = load_backends()
    backend = (
        backends["ultralytics"]["models"][model_id]
        if model["adapter"] == "ultralytics"
        else backends["dfine"]
    )
    return model, backend


def validate_protocol(
    track: str = "RGB", *, verify_files: bool = True
) -> dict[str, Any]:
    normalized = track.upper()
    protocol = load_protocol(normalized)
    if protocol.get("track") != normalized:
        raise ProtocolError(f"Protocol track must be {normalized}")
    if [item.get("id") for item in protocol.get("models", [])] != [
        "yolo11n",
        "yolo26n",
        "dfine_n",
    ]:
        raise ProtocolError("Model order must be yolo11n, yolo26n, dfine_n")

    dataset = protocol["dataset"]
    if (dataset.get("width"), dataset.get("height")) != (640, 640):
        raise ProtocolError("Both tracks require 640x640 inputs")
    training = protocol["training"]
    if (
        training.get("epochs"),
        training.get("physical_batch_size"),
        training.get("effective_batch_size"),
    ) != (220, 8, 32):
        raise ProtocolError(
            "Frozen training budget must be 220 epochs, physical batch 8, effective batch 32"
        )
    if (
        training.get("gradient_accumulation_steps") != 4
        or training.get("early_stopping") is not False
    ):
        raise ProtocolError(
            "Frozen accumulation must be 4 and early stopping must be disabled"
        )
    if training.get("checkpoint_retention") != {
        "best": True,
        "last": True,
        "periodic_every_epochs": 100,
    }:
        raise ProtocolError(
            "Checkpoint retention must keep best/last and 100-epoch milestones only"
        )

    if normalized == "RGB":
        classes = {int(key): value for key, value in dataset["classes"].items()}
        if classes != {
            1: "yawning",
            2: "hand_over_mouth",
            3: "drinking",
            4: "phone_use",
        }:
            raise ProtocolError("RGB ontology mismatch")
        if (
            tuple(training.get("run_seeds", ())) != RGB_SEEDS
            or training.get("runs_per_model") != 3
        ):
            raise ProtocolError("RGB requires seeds 13, 37, and 73")
    else:
        classes = {int(key): value for key, value in dataset["classes"].items()}
        if classes != {1: "drinking", 2: "phone_use"}:
            raise ProtocolError("NIR ontology mismatch")
        if (
            training.get("seed") != NIR_SEED
            or tuple(training.get("ratios", ())) != NIR_RATIOS
        ):
            raise ProtocolError("NIR requires seed 13 and ratios 1to2 and 1to6 only")
        if dataset.get("snippet_fps") != 10 or dataset.get("frames_per_snippet") != 10:
            raise ProtocolError(
                "NIR snippets must contain ten frames sampled at 10 FPS"
            )
        if dataset.get("evaluation_sets") != "identical_across_ratios":
            raise ProtocolError(
                "NIR validation and test sets must be identical across ratios"
            )
        if training.get("ratio_policy") != "nested_training_negatives_only":
            raise ProtocolError(
                "NIR 1:2 negatives must be nested inside the 1:6 training pool"
            )

    if verify_files:
        verify_authoritative_fingerprints(protocol, normalized)
    return protocol
