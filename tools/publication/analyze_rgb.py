"""Derive annotation, class, subject, and paired-seed evidence from frozen predictions."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.benchmark.evaluation import (
    coco_metrics,
    load_ground_truth,
    operating_point_metrics,
    validate_predictions,
)
from tools.benchmark.paths import REPO_ROOT, RESULTS_ROOT
from tools.benchmark.protocol import ProtocolError, validate_protocol

MODELS = ("yolo11n", "yolo26n")
SEEDS = (13, 37, 73)
SUMMARY_ROOT = RESULTS_ROOT / "RGB" / "summary"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subject_name(file_name: str) -> str:
    parts = Path(file_name).parts
    if len(parts) < 2 or parts[0] != "images":
        raise ProtocolError(f"Unexpected RGB image path: {file_name}")
    return parts[1]


def filter_ground_truth(
    ground_truth: dict[str, Any],
    *,
    image_ids: set[int] | None = None,
    category_id: int | None = None,
) -> dict[str, Any]:
    selected_images = [
        image
        for image in ground_truth["images"]
        if image_ids is None or int(image["id"]) in image_ids
    ]
    selected_ids = {int(image["id"]) for image in selected_images}
    selected_annotations = [
        annotation
        for annotation in ground_truth["annotations"]
        if int(annotation["image_id"]) in selected_ids
        and (category_id is None or int(annotation["category_id"]) == category_id)
    ]
    categories = ground_truth.get("categories", [])
    if category_id is not None:
        categories = [item for item in categories if int(item["id"]) == category_id]
    result = {
        key: value
        for key, value in ground_truth.items()
        if key not in {"images", "annotations", "categories"}
    }
    result.update(
        images=selected_images,
        annotations=selected_annotations,
        categories=categories,
    )
    return result


def annotation_audit(master: dict[str, Any]) -> dict[str, Any]:
    images = master["images"]
    annotations = master["annotations"]
    image_by_id = {int(image["id"]): image for image in images}
    category_names = {
        int(category["id"]): category["name"] for category in master["categories"]
    }
    annotation_counts = Counter(int(item["image_id"]) for item in annotations)
    problems = {
        "duplicate_image_ids": len(images) - len({int(item["id"]) for item in images}),
        "duplicate_file_names": len(images) - len({item["file_name"] for item in images}),
        "duplicate_annotation_ids": len(annotations)
        - len({int(item["id"]) for item in annotations}),
        "orphan_annotations": 0,
        "invalid_categories": 0,
        "nonfinite_boxes": 0,
        "nonpositive_boxes": 0,
        "out_of_bounds_boxes": 0,
        "images_over_one_annotation": sum(count > 1 for count in annotation_counts.values()),
    }
    areas: dict[str, list[float]] = defaultdict(list)
    truncated = Counter()
    for annotation in annotations:
        image = image_by_id.get(int(annotation["image_id"]))
        category_name = category_names.get(int(annotation["category_id"]))
        if image is None:
            problems["orphan_annotations"] += 1
            continue
        if category_name is None:
            problems["invalid_categories"] += 1
            continue
        x, y, width, height = map(float, annotation["bbox"])
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            problems["nonfinite_boxes"] += 1
        if width <= 0.0 or height <= 0.0:
            problems["nonpositive_boxes"] += 1
        image_width = float(image["width"])
        image_height = float(image["height"])
        if (
            x < 0.0
            or y < 0.0
            or x + width > image_width + 1e-6
            or y + height > image_height + 1e-6
        ):
            problems["out_of_bounds_boxes"] += 1
        areas[category_name].append(100.0 * width * height / (image_width * image_height))
        if (
            x <= 1e-6
            or y <= 1e-6
            or x + width >= image_width - 1e-6
            or y + height >= image_height - 1e-6
        ):
            truncated[category_name] += 1

    area_summary = {}
    for category_name in sorted(areas):
        values = np.asarray(areas[category_name], dtype=np.float64)
        area_summary[category_name] = {
            "annotations": int(values.size),
            "min_percent": float(np.min(values)),
            "median_percent": float(np.median(values)),
            "p95_percent": float(np.quantile(values, 0.95)),
            "max_percent": float(np.max(values)),
            "below_1_percent": int(np.sum(values < 1.0)),
            "below_2_percent": int(np.sum(values < 2.0)),
            "touching_image_boundary": int(truncated[category_name]),
        }
    return {
        "scope": "automated_structural_audit_not_semantic_agreement",
        "primary_human_annotators": 1,
        "independent_second_person_review": False,
        "images": len(images),
        "annotations": len(annotations),
        "negative_images": len(images) - len(annotations),
        "problems": problems,
        "box_area_by_class": area_summary,
    }


def class_operating_metrics(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    threshold: float,
) -> tuple[list[dict[str, Any]], float]:
    rows = []
    for category in ground_truth["categories"]:
        category_id = int(category["id"])
        category_name = category["name"]
        category_ground_truth = filter_ground_truth(
            ground_truth, category_id=category_id
        )
        category_predictions = [
            item for item in predictions if int(item["category_id"]) == category_id
        ]
        metrics = operating_point_metrics(
            category_ground_truth, category_predictions, threshold
        )
        rows.append(
            {
                "class_name": category_name,
                "support": len(category_ground_truth["annotations"]),
                **{
                    key: metrics[key]
                    for key in ("tp", "fp", "fn", "precision", "recall", "micro_f1")
                },
            }
        )
    return rows, statistics.fmean(float(row["micro_f1"]) for row in rows)


def subject_operating_metrics(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    subject_ids: dict[str, set[int]] = defaultdict(set)
    for image in ground_truth["images"]:
        subject_ids[subject_name(image["file_name"])].add(int(image["id"]))
    rows = []
    for subject in sorted(subject_ids):
        image_ids = subject_ids[subject]
        subject_ground_truth = filter_ground_truth(ground_truth, image_ids=image_ids)
        subject_predictions = [
            item for item in predictions if int(item["image_id"]) in image_ids
        ]
        metrics = operating_point_metrics(
            subject_ground_truth, subject_predictions, threshold
        )
        rows.append(
            {
                "subject": subject,
                "images": len(subject_ground_truth["images"]),
                "support": len(subject_ground_truth["annotations"]),
                **{
                    key: metrics[key]
                    for key in (
                        "tp",
                        "fp",
                        "fn",
                        "precision",
                        "recall",
                        "micro_f1",
                        "far_per_100_negative_frames",
                    )
                },
            }
        )
    return rows


def verify_published_metrics(
    run: dict[str, Any], published: dict[str, Any], model_id: str, seed: int
) -> None:
    checks = {
        "map_50_95": run["coco_metrics"]["map_50_95"],
        "map_50": run["coco_metrics"]["map_50"],
        "precision": run["operating_point"]["precision"],
        "recall": run["operating_point"]["recall"],
        "micro_f1": run["operating_point"]["micro_f1"],
        "far_per_100_negative_frames": run["operating_point"][
            "far_per_100_negative_frames"
        ],
    }
    for key, actual in checks.items():
        if not math.isclose(
            float(actual), float(published[key]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ProtocolError(f"{key} drift for {model_id}/seed_{seed}")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    protocol = validate_protocol("RGB")
    annotations_path = REPO_ROOT / protocol["dataset"]["annotations"]
    splits_path = REPO_ROOT / protocol["dataset"]["splits"]
    master = json.loads(annotations_path.read_text(encoding="utf-8"))
    ground_truth = load_ground_truth(
        protocol["dataset"]["annotations"],
        "test",
        protocol["dataset"]["splits"],
    )
    aggregate_path = SUMMARY_ROOT / "final_benchmark_aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    published_runs = {
        (item["model_id"], int(item["training_seed"])): item
        for item in aggregate["runs"]
    }

    runs: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    for model_id in MODELS:
        for seed in SEEDS:
            prediction_path = (
                RESULTS_ROOT
                / "RGB"
                / model_id
                / f"seed_{seed}"
                / "test_predictions.json"
            )
            envelope = json.loads(prediction_path.read_text(encoding="utf-8"))
            if (
                envelope.get("artifact") != "sanitized_protected_test_predictions"
                or envelope.get("model_id") != model_id
                or int(envelope.get("training_seed", -1)) != seed
            ):
                raise ProtocolError(f"Prediction identity mismatch: {prediction_path}")
            predictions = validate_predictions(
                envelope["predictions"],
                {int(image["id"]) for image in ground_truth["images"]},
            )
            threshold = float(envelope["threshold"])
            operating_point = operating_point_metrics(
                ground_truth, predictions, threshold
            )
            detection_metrics = coco_metrics(ground_truth, predictions)
            per_class, macro_f1 = class_operating_metrics(
                ground_truth, predictions, threshold
            )
            per_subject = subject_operating_metrics(
                ground_truth, predictions, threshold
            )
            run = {
                "model_id": model_id,
                "training_seed": seed,
                "threshold": threshold,
                "prediction_artifact": prediction_path.relative_to(REPO_ROOT).as_posix(),
                "prediction_sha256": sha256(prediction_path),
                "coco_metrics": detection_metrics,
                "operating_point": operating_point,
                "macro_f1": macro_f1,
                "per_class": per_class,
                "per_subject": per_subject,
            }
            verify_published_metrics(run, published_runs[(model_id, seed)], model_id, seed)
            runs.append(run)
            run_rows.append(
                {
                    "model_id": model_id,
                    "training_seed": seed,
                    "threshold": threshold,
                    "map_50_95": detection_metrics["map_50_95"],
                    "map_50": detection_metrics["map_50"],
                    "precision": operating_point["precision"],
                    "recall": operating_point["recall"],
                    "micro_f1": operating_point["micro_f1"],
                    "macro_f1": macro_f1,
                    "far_per_100_negative_frames": operating_point[
                        "far_per_100_negative_frames"
                    ],
                }
            )
            for row in per_class:
                class_rows.append(
                    {"model_id": model_id, "training_seed": seed, **row}
                )
            for row in per_subject:
                subject_rows.append(
                    {"model_id": model_id, "training_seed": seed, **row}
                )

    run_lookup = {
        (item["model_id"], int(item["training_seed"])): item for item in runs
    }
    published_lookup = published_runs
    paired_rows: list[dict[str, Any]] = []
    paired_specs = {
        "map_50_95": lambda model, seed: run_lookup[(model, seed)]["coco_metrics"][
            "map_50_95"
        ],
        "map_50": lambda model, seed: run_lookup[(model, seed)]["coco_metrics"][
            "map_50"
        ],
        "micro_f1": lambda model, seed: run_lookup[(model, seed)]["operating_point"][
            "micro_f1"
        ],
        "macro_f1": lambda model, seed: run_lookup[(model, seed)]["macro_f1"],
        "far_per_100_negative_frames": lambda model, seed: published_lookup[
            (model, seed)
        ]["far_per_100_negative_frames"],
        "tensor_to_detections_p50_ms": lambda model, seed: published_lookup[
            (model, seed)
        ]["tensor_to_final_detections_p50_ms"],
        "sustained_fps": lambda model, seed: published_lookup[(model, seed)][
            "tensor_to_final_detections_sustained_fps"
        ],
    }
    category_names = [item["name"] for item in ground_truth["categories"]]
    for category_name in category_names:
        paired_specs[f"ap_50_95_{category_name}"] = (
            lambda model, seed, name=category_name: run_lookup[(model, seed)][
                "coco_metrics"
            ]["per_class_ap_50_95"][name]
        )
    for metric, getter in paired_specs.items():
        differences = []
        records = []
        for seed in SEEDS:
            yolo11n = float(getter("yolo11n", seed))
            yolo26n = float(getter("yolo26n", seed))
            difference = yolo11n - yolo26n
            differences.append(difference)
            records.append((seed, yolo11n, yolo26n, difference))
        direction = (
            "yolo11n_higher"
            if all(value > 0.0 for value in differences)
            else "yolo26n_higher"
            if all(value < 0.0 for value in differences)
            else "mixed"
        )
        for seed, yolo11n, yolo26n, difference in records:
            paired_rows.append(
                {
                    "metric": metric,
                    "training_seed": seed,
                    "yolo11n": yolo11n,
                    "yolo26n": yolo26n,
                    "difference_yolo11n_minus_yolo26n": difference,
                    "direction_across_three_seeds": direction,
                }
            )

    model_summary = {}
    for model_id in MODELS:
        selected = [item for item in run_rows if item["model_id"] == model_id]
        model_summary[model_id] = {}
        for metric in (
            "map_50_95",
            "map_50",
            "precision",
            "recall",
            "micro_f1",
            "macro_f1",
            "far_per_100_negative_frames",
        ):
            values = [float(item[metric]) for item in selected]
            model_summary[model_id][metric] = {
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values),
            }

    analysis = {
        "artifact": "rgb_secondary_analysis",
        "schema_version": 1,
        "interpretation": {
            "seed_dispersion": "optimization variability on one fixed subject split",
            "subject_analysis": "descriptive sensitivity across the three fixed test subjects",
            "no_generalization_interval": True,
            "paired_difference": "yolo11n minus yolo26n within the same training seed",
        },
        "inputs": {
            "annotations": annotations_path.relative_to(REPO_ROOT).as_posix(),
            "annotations_sha256": sha256(annotations_path),
            "splits": splits_path.relative_to(REPO_ROOT).as_posix(),
            "splits_sha256": sha256(splits_path),
            "aggregate": aggregate_path.relative_to(REPO_ROOT).as_posix(),
            "aggregate_sha256": sha256(aggregate_path),
        },
        "annotation_audit": annotation_audit(master),
        "model_summary": model_summary,
        "runs": runs,
        "paired_seed_differences": paired_rows,
    }
    analysis_path = SUMMARY_ROOT / "secondary_analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(
        SUMMARY_ROOT / "operating_point_by_run.csv",
        list(run_rows[0]),
        run_rows,
    )
    write_csv(
        SUMMARY_ROOT / "operating_point_by_class.csv",
        list(class_rows[0]),
        class_rows,
    )
    write_csv(
        SUMMARY_ROOT / "operating_point_by_subject.csv",
        list(subject_rows[0]),
        subject_rows,
    )
    write_csv(
        SUMMARY_ROOT / "paired_seed_differences.csv",
        list(paired_rows[0]),
        paired_rows,
    )
    print(f"Wrote {analysis_path.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
