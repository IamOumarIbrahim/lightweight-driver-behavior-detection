"""Shared COCO evaluation and frozen operating-point metrics."""

from __future__ import annotations

import contextlib
import io
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import ProtocolError, resolve_repo_path

PREDICTION_SCHEMA_VERSION = 1
REQUIRED_PREDICTION_KEYS = {"image_id", "category_id", "bbox", "score"}


def xywh_to_xyxy(box: Iterable[float]) -> tuple[float, float, float, float]:
    x, y, width, height = map(float, box)
    return x, y, x + width, y + height


def compute_iou(box1: Iterable[float], box2: Iterable[float]) -> float:
    """IoU for two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = map(float, box1)
    bx1, by1, bx2, by2 = map(float, box2)
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - intersection
    return intersection / union if union > 0.0 else 0.0


def validate_predictions(
    predictions: Iterable[dict[str, Any]], valid_image_ids: set[int] | None = None
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, prediction in enumerate(predictions):
        missing = REQUIRED_PREDICTION_KEYS - prediction.keys()
        if missing:
            raise ProtocolError(f"Prediction {index} is missing {sorted(missing)}")
        image_id = int(prediction["image_id"])
        category_id = int(prediction["category_id"])
        bbox = [float(value) for value in prediction["bbox"]]
        score = float(prediction["score"])
        if valid_image_ids is not None and image_id not in valid_image_ids:
            raise ProtocolError(
                f"Prediction {index} references image {image_id} outside the selected split"
            )
        if category_id not in {1, 2, 3, 4}:
            raise ProtocolError(
                f"Prediction {index} has invalid category {category_id}"
            )
        if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
            raise ProtocolError(f"Prediction {index} has invalid bbox")
        if bbox[2] <= 0.0 or bbox[3] <= 0.0:
            raise ProtocolError(f"Prediction {index} has a non-positive bbox")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ProtocolError(f"Prediction {index} has invalid score {score}")
        normalized.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "score": score,
            }
        )
    return normalized


def split_ground_truth(
    master: dict[str, Any], subjects: Iterable[str]
) -> dict[str, Any]:
    """Return a COCO object filtered by subject while preserving master image order."""
    subject_set = set(subjects)
    images = [
        image
        for image in master["images"]
        if Path(image["file_name"]).parts[1] in subject_set
    ]
    image_ids = {int(image["id"]) for image in images}
    annotations = [
        annotation
        for annotation in master["annotations"]
        if int(annotation["image_id"]) in image_ids
    ]
    result = {
        key: value
        for key, value in master.items()
        if key not in {"images", "annotations"}
    }
    result["images"] = images
    result["annotations"] = annotations
    return result


def load_ground_truth(
    annotations_path: str | Path, split: str, splits_path: str | Path
) -> dict[str, Any]:
    with resolve_repo_path(annotations_path).open("r", encoding="utf-8") as handle:
        master = json.load(handle)
    with resolve_repo_path(splits_path).open("r", encoding="utf-8") as handle:
        splits = json.load(handle)
    key = "validation" if split == "val" else split
    if key not in splits:
        raise ProtocolError(f"Unknown split: {split}")
    return split_ground_truth(master, splits[key])


def operating_point_metrics(
    ground_truth: dict[str, Any],
    predictions: Iterable[dict[str, Any]],
    threshold: float,
    iou_threshold: float = 0.50,
) -> dict[str, float | int]:
    """Compute frozen same-class greedy one-to-one micro metrics and FAR."""
    if not 0.0 <= threshold <= 1.0:
        raise ProtocolError("Threshold must be in [0, 1]")
    image_ids = {int(image["id"]) for image in ground_truth["images"]}
    predictions = validate_predictions(predictions, image_ids)
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    pred_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in ground_truth["annotations"]:
        gt_by_image[int(annotation["image_id"])].append(annotation)
    for prediction in predictions:
        if prediction["score"] >= threshold:
            pred_by_image[prediction["image_id"]].append(prediction)

    tp = fp = fn = fp_on_negative_frames = negative_frames = 0
    for image in ground_truth["images"]:
        image_id = int(image["id"])
        ground_truths = gt_by_image[image_id]
        detections = sorted(
            pred_by_image[image_id],
            key=lambda item: (-item["score"], item["category_id"], *item["bbox"]),
        )
        if not ground_truths:
            negative_frames += 1
            fp_on_negative_frames += len(detections)
        matched: set[int] = set()
        for detection in detections:
            best_iou = -1.0
            best_index: int | None = None
            detection_box = xywh_to_xyxy(detection["bbox"])
            for gt_index, annotation in enumerate(ground_truths):
                if (
                    gt_index in matched
                    or int(annotation["category_id"]) != detection["category_id"]
                ):
                    continue
                iou = compute_iou(detection_box, xywh_to_xyxy(annotation["bbox"]))
                if iou > best_iou:
                    best_iou, best_index = iou, gt_index
            if best_index is not None and best_iou >= iou_threshold:
                matched.add(best_index)
                tp += 1
            else:
                fp += 1
        fn += len(ground_truths) - len(matched)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    far = 100.0 * fp_on_negative_frames / negative_frames if negative_frames else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "micro_f1": f1,
        "fp_detections_on_negative_frames": fp_on_negative_frames,
        "negative_frames": negative_frames,
        "far_per_100_negative_frames": far,
        "threshold": threshold,
        "iou_threshold": iou_threshold,
    }


def threshold_grid() -> tuple[float, ...]:
    return tuple(index / 100.0 for index in range(1, 100))


def select_threshold_candidate(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Select by micro-F1, then precision, then the higher threshold."""
    candidates = list(candidates)
    if not candidates:
        raise ProtocolError("At least one threshold candidate is required")
    return max(
        candidates,
        key=lambda item: (item["micro_f1"], item["precision"], item["threshold"]),
    )


def calibrate_threshold(
    ground_truth: dict[str, Any], predictions: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Validation-only threshold search: F1, then precision, then higher threshold."""
    predictions = list(predictions)
    candidates = [
        operating_point_metrics(ground_truth, predictions, threshold)
        for threshold in threshold_grid()
    ]
    return select_threshold_candidate(candidates)


def select_checkpoint_candidate(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Select by mAP50:95, then mAP50, then the later epoch."""
    candidates = list(candidates)
    if not candidates:
        raise ProtocolError("At least one validation prediction artifact is required")
    return max(
        candidates, key=lambda item: (item["map_50_95"], item["map_50"], item["epoch"])
    )


def validate_checkpoint_candidate_coverage(
    candidates: Iterable[dict[str, Any]], expected_epochs: int
) -> list[dict[str, Any]]:
    """Require one unique validation artifact for every retained training epoch."""

    candidates = list(candidates)
    epochs = sorted(int(item["epoch"]) for item in candidates)
    valid_numberings = [
        list(range(expected_epochs)),
        list(range(1, expected_epochs + 1)),
    ]
    if len(candidates) != expected_epochs or epochs not in valid_numberings:
        raise ProtocolError(
            f"Checkpoint selection requires exactly {expected_epochs} consecutive retained epochs "
            "numbered either 0..epochs-1 or 1..epochs"
        )
    checkpoints = [str(item["checkpoint"]) for item in candidates]
    predictions = [str(item["validation_predictions"]) for item in candidates]
    if (
        len(set(checkpoints)) != expected_epochs
        or len(set(predictions)) != expected_epochs
    ):
        raise ProtocolError(
            "Each retained epoch requires a unique checkpoint and validation-prediction artifact"
        )
    return candidates


def coco_metrics(
    ground_truth: dict[str, Any], predictions: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Compute COCO mAP with the official ``pycocotools`` implementation."""
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise ProtocolError("pycocotools is required for COCO evaluation") from exc

    image_ids = {int(image["id"]) for image in ground_truth["images"]}
    predictions = validate_predictions(predictions, image_ids)
    coco_gt = COCO()
    coco_gt.dataset = ground_truth
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        if predictions:
            coco_dt = coco_gt.loadRes(predictions)
        else:
            coco_dt = COCO()
            coco_dt.dataset = {
                "images": ground_truth["images"],
                "annotations": [],
                "categories": ground_truth.get("categories", []),
            }
            coco_dt.createIndex()
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.imgIds = sorted(image_ids)
        evaluator.params.catIds = (
            [int(cat["id"]) for cat in ground_truth.get("categories", [])]
            if ground_truth.get("categories")
            else [1, 2, 3, 4]
        )
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    precision = evaluator.eval["precision"]  # IoU, recall, category, area, maxDets
    per_class: dict[str, float] = {}
    names = {
        int(item["id"]): item["name"] for item in ground_truth.get("categories", [])
    }
    for category_index, category_id in enumerate(evaluator.params.catIds):
        values = precision[:, :, category_index, 0, -1]
        valid = values[values > -1]
        if category_id in names:
            per_class[names[category_id]] = (
                float(np.mean(valid)) if valid.size else float("nan")
            )
    return {
        "map_50_95": float(evaluator.stats[0]),
        "map_50": float(evaluator.stats[1]),
        "per_class_ap_50_95": per_class,
        "implementation": "pycocotools",
    }


def read_prediction_envelope(
    path: str | Path, required_split: str | None = None
) -> dict[str, Any]:
    with resolve_repo_path(path).open("r", encoding="utf-8") as handle:
        envelope = json.load(handle)
    if envelope.get("schema_version") != PREDICTION_SCHEMA_VERSION:
        raise ProtocolError("Unsupported prediction artifact schema")
    if required_split is not None and envelope.get("split") != required_split:
        raise ProtocolError(
            f"Expected {required_split} predictions, got {envelope.get('split')}"
        )
    envelope["predictions"] = validate_predictions(envelope.get("predictions", []))
    return envelope
