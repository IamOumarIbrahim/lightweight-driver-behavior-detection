"""Build a public, path-safe validation operating-point sweep from frozen runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from tools.benchmark.evaluation import load_ground_truth, operating_point_metrics
from tools.benchmark.paths import REPO_ROOT, RESULTS_ROOT
from tools.benchmark.protocol import ProtocolError

MODELS = ("yolo11n", "yolo26n")
SEEDS = (13, 37, 73)
THRESHOLDS = tuple(index / 100 for index in range(1, 100))
OUTPUT_PATH = RESULTS_ROOT / "RGB" / "summary" / "validation_operating_point_sweep.csv"


def filtered_category(
    ground_truth: dict[str, Any], category_id: int
) -> dict[str, Any]:
    """Keep all frames but only one category's annotations."""
    result = {
        key: value
        for key, value in ground_truth.items()
        if key not in {"annotations", "categories"}
    }
    result["annotations"] = [
        item
        for item in ground_truth["annotations"]
        if int(item["category_id"]) == category_id
    ]
    result["categories"] = [
        item
        for item in ground_truth.get("categories", [])
        if int(item["id"]) == category_id
    ]
    return result


def load_run(model_id: str, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_root = REPO_ROOT / "runs" / "RGB" / model_id / f"seed_{seed}" / "validation"
    prediction_path = run_root / "predictions.json"
    calibration_path = run_root / "calibration.json"
    if not prediction_path.is_file() or not calibration_path.is_file():
        raise ProtocolError(
            f"Frozen validation artifacts are missing for {model_id}/seed_{seed}"
        )
    envelope = json.loads(prediction_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if envelope.get("artifact") != "validation_predictions":
        raise ProtocolError(f"Unexpected prediction artifact for {model_id}/seed_{seed}")
    if envelope.get("model_id") != model_id or calibration.get("model_id") != model_id:
        raise ProtocolError(f"Model identity mismatch for {model_id}/seed_{seed}")
    if int(calibration.get("training_seed", -1)) != seed:
        raise ProtocolError(f"Seed identity mismatch for {model_id}/seed_{seed}")
    return envelope["predictions"], calibration


def main() -> None:
    ground_truth = load_ground_truth(
        "data/annotations/RGB/annotations.json",
        "val",
        "data/annotations/RGB/splits.json",
    )
    category_ids = tuple(int(item["id"]) for item in ground_truth["categories"])
    category_ground_truth = {
        category_id: filtered_category(ground_truth, category_id)
        for category_id in category_ids
    }
    rows: list[dict[str, str | int | float]] = []

    for model_id in MODELS:
        for seed in SEEDS:
            predictions, calibration = load_run(model_id, seed)
            selected = calibration["selected"]
            selected_threshold = float(selected["threshold"])
            for threshold in THRESHOLDS:
                overall = operating_point_metrics(ground_truth, predictions, threshold)
                class_f1 = []
                for category_id in category_ids:
                    category_predictions = [
                        item
                        for item in predictions
                        if int(item["category_id"]) == category_id
                    ]
                    category_metrics = operating_point_metrics(
                        category_ground_truth[category_id],
                        category_predictions,
                        threshold,
                    )
                    class_f1.append(float(category_metrics["micro_f1"]))
                selected_row = math.isclose(
                    threshold, selected_threshold, rel_tol=0.0, abs_tol=1e-12
                )
                if selected_row and not math.isclose(
                    float(overall["micro_f1"]),
                    float(selected["micro_f1"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ProtocolError(
                        f"Calibration mismatch for {model_id}/seed_{seed}"
                    )
                rows.append(
                    {
                        "model_id": model_id,
                        "training_seed": seed,
                        "threshold": threshold,
                        "micro_f1": float(overall["micro_f1"]),
                        "macro_f1": sum(class_f1) / len(class_f1),
                        "false_detections_per_100_negative_frames": float(
                            overall["far_per_100_negative_frames"]
                        ),
                        "selected_primary": str(selected_row).lower(),
                    }
                )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} validation sweep rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
