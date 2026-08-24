"""Publish sanitized frozen RGB test predictions without rerunning inference."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from tools.benchmark.evaluation import (
    load_ground_truth,
    operating_point_metrics,
    validate_predictions,
)
from tools.benchmark.paths import REPO_ROOT, RESULTS_ROOT, RUNS_ROOT
from tools.benchmark.protocol import ProtocolError, validate_protocol

MODELS = ("yolo11n", "yolo26n")
SEEDS = (13, 37, 73)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ProtocolError(f"{label} mismatch: {actual} != {expected}")


def export_run(
    model_id: str,
    seed: int,
    ground_truth: dict,
    annotations_path: Path,
) -> Path:
    source = RUNS_ROOT / "RGB" / model_id / f"seed_{seed}" / "test" / "result.json"
    if not source.is_file():
        raise ProtocolError(f"Frozen protected-test result is missing: {source}")
    result = json.loads(source.read_text(encoding="utf-8"))
    if result.get("artifact") != "protected_test_result":
        raise ProtocolError(f"Unexpected source artifact: {source}")
    if result.get("model_id") != model_id or int(result.get("training_seed", -1)) != seed:
        raise ProtocolError(f"Run identity mismatch: {source}")

    valid_image_ids = {int(image["id"]) for image in ground_truth["images"]}
    predictions = validate_predictions(result.get("predictions", []), valid_image_ids)
    threshold = float(result["threshold"])
    recomputed = operating_point_metrics(ground_truth, predictions, threshold)
    published = result["operating_point"]
    for metric in (
        "precision",
        "recall",
        "micro_f1",
        "far_per_100_negative_frames",
    ):
        require_close(float(recomputed[metric]), float(published[metric]), metric)
    for metric in ("tp", "fp", "fn", "fp_detections_on_negative_frames"):
        if int(recomputed[metric]) != int(published[metric]):
            raise ProtocolError(f"{metric} mismatch for {model_id}/seed_{seed}")

    relative_source = source.relative_to(REPO_ROOT).as_posix()
    relative_annotations = annotations_path.relative_to(REPO_ROOT).as_posix()
    payload = {
        "artifact": "sanitized_protected_test_predictions",
        "schema_version": 1,
        "track": "RGB",
        "split": "test",
        "model_id": model_id,
        "training_seed": seed,
        "threshold": threshold,
        "threshold_source": "validation_only_micro_f1",
        "operating_iou": 0.5,
        "prediction_count": len(predictions),
        "predictions": predictions,
        "provenance": {
            "source_result": relative_source,
            "source_result_sha256": sha256(source),
            "ground_truth": relative_annotations,
            "ground_truth_sha256": sha256(annotations_path),
            "manifest_id": result["manifest_id"],
            "suite_id": result["suite_id"],
        },
    }
    destination = RESULTS_ROOT / "RGB" / model_id / f"seed_{seed}" / "test_predictions.json"
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def main() -> int:
    protocol = validate_protocol("RGB")
    annotations_path = REPO_ROOT / protocol["dataset"]["annotations"]
    ground_truth = load_ground_truth(
        protocol["dataset"]["annotations"],
        "test",
        protocol["dataset"]["splits"],
    )
    for model_id in MODELS:
        for seed in SEEDS:
            destination = export_run(model_id, seed, ground_truth, annotations_path)
            print(f"Published {destination.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
