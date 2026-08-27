"""Build the path-safe NIR publication source from frozen protected results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from tools.benchmark.paths import (
    NIR_MODELS,
    NIR_RATIOS,
    NIR_SEED,
    REPO_ROOT,
    RESULTS_ROOT,
)
from tools.benchmark.protocol import ProtocolError, sha256_file

SUMMARY_PATH = (
    RESULTS_ROOT / "NIR" / "summary" / "training_negative_exposure_source.json"
)
RATIO_LABELS = {"1to2": "1:2", "1to6": "1:6"}


def require_number(value: Any, label: str, *, lower: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{label} must be finite")
    if lower is not None and result < lower:
        raise ProtocolError(f"{label} must be at least {lower}")
    return result


def require_hash(value: Any, label: str) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ProtocolError(f"{label} must be a SHA-256 digest")
    return result


def publication_row(path: Path, model: str, ratio: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_identity = {
        "track": "NIR",
        "model": model,
        "ratio": ratio,
        "training_seed": NIR_SEED,
    }
    for field, expected in expected_identity.items():
        if payload.get(field) != expected:
            raise ProtocolError(
                f"{path.relative_to(REPO_ROOT)} has {field}={payload.get(field)!r}; "
                f"expected {expected!r}"
            )
    for field in (
        "checkpoint_sha256",
        "validation_manifest_sha256",
        "test_ground_truth_sha256",
        "test_predictions_sha256",
    ):
        require_hash(payload.get(field), f"{path.name}:{field}")
    runtime = payload.get("runtime_profile", {})
    if runtime.get("environment", {}).get("cuda_available") is not True:
        raise ProtocolError(f"{path.relative_to(REPO_ROOT)} is not a CUDA result")
    if runtime.get("precision_mode") != "cuda_amp_fp16":
        raise ProtocolError(
            f"{path.relative_to(REPO_ROOT)} is not the frozen FP16 profile"
        )
    snippet = payload.get("snippet_operating_point", {})
    if snippet.get("frames_per_snippet") != 1:
        raise ProtocolError(
            f"{path.relative_to(REPO_ROOT)} is not the 1-FPS midpoint result"
        )

    relative_path = path.relative_to(REPO_ROOT).as_posix()
    return {
        "model_id": model,
        "ratio": RATIO_LABELS[ratio],
        "map_50_95": require_number(
            payload["coco_metrics"]["map_50_95"],
            f"{relative_path}:map_50_95",
            lower=0,
        ),
        "micro_f1": require_number(
            payload["operating_point"]["micro_f1"],
            f"{relative_path}:micro_f1",
            lower=0,
        ),
        "macro_f1": require_number(
            snippet["macro_f1"], f"{relative_path}:macro_f1", lower=0
        ),
        "false_detections_per_100_negative_frames": require_number(
            payload["operating_point"]["far_per_100_negative_frames"],
            f"{relative_path}:false_detections_per_100_negative_frames",
            lower=0,
        ),
        "tensor_to_final_p50_ms": require_number(
            runtime["tensor_to_final_detections"]["p50_ms"],
            f"{relative_path}:tensor_to_final_p50_ms",
            lower=0,
        ),
        "tensor_to_final_sustained_fps": require_number(
            runtime["tensor_to_final_detections"]["sustained_fps"],
            f"{relative_path}:tensor_to_final_sustained_fps",
            lower=0,
        ),
        "metrics_path": relative_path,
        "metrics_sha256": sha256_file(path),
    }


def build_source() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    completed_models: list[str] = []
    ground_truth_hashes: set[str] = set()
    for model in NIR_MODELS:
        result_paths = [
            RESULTS_ROOT / "NIR" / model / f"ratio_{ratio}" / "metrics.json"
            for ratio in NIR_RATIOS
        ]
        present = [path.is_file() for path in result_paths]
        if any(present) and not all(present):
            raise ProtocolError(
                f"NIR publication requires both ratios or neither for {model}"
            )
        if not all(present):
            continue
        completed_models.append(model)
        for ratio, path in zip(NIR_RATIOS, result_paths, strict=True):
            row = publication_row(path, model, ratio)
            result = json.loads(path.read_text(encoding="utf-8"))
            ground_truth_hashes.add(result["test_ground_truth_sha256"])
            rows.append(row)
    if len(ground_truth_hashes) > 1:
        raise ProtocolError(
            "Completed NIR results do not share one protected test split"
        )
    pending_models = [model for model in NIR_MODELS if model not in completed_models]
    return {
        "artifact": "nir_training_negative_exposure_figure_source",
        "status": "complete" if not pending_models else "partial",
        "seed": NIR_SEED,
        "ratios": [RATIO_LABELS[ratio] for ratio in NIR_RATIOS],
        "models": list(NIR_MODELS),
        "completed_models": completed_models,
        "pending_models": pending_models,
        "sampling": {
            "fps": 1,
            "sample_time_seconds": 0.5,
            "source_frame_offset": 14,
        },
        "metrics": [
            "map_50_95",
            "macro_f1",
            "false_detections_per_100_negative_frames",
        ],
        "rows": rows,
    }


def main() -> int:
    payload = build_source()
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    relative_summary = SUMMARY_PATH.relative_to(REPO_ROOT)
    print(f"Wrote {relative_summary} with {len(payload['rows'])} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
