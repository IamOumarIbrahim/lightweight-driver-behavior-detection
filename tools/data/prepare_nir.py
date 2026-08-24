"""Prepare the frozen Drive&Act NIR benchmark at ten frames per second."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

from tools.benchmark.paths import DATA_ROOT, NIR_RATIOS
from tools.benchmark.protocol import ProtocolError, sha256_file, validate_protocol

OFFSETS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29)
CLASS_IDS = {"drinking": 0, "phone_use": 1, "talking_on_phone": 1}


def signature(task: dict[str, Any]) -> tuple[int, str, str, int]:
    data = task["data"]
    return int(task["id"]), data["subject"], data["video_run"], int(data["frame_start"])


def result_value(task: dict[str, Any]) -> dict[str, Any] | None:
    for annotation in task.get("annotations", []):
        for result in annotation.get("result", []):
            if result.get("type") == "videorectangle" and result.get("value", {}).get(
                "sequence"
            ):
                return result["value"]
    return None


def interpolate_box(
    sequence: list[dict[str, Any]], time: float
) -> tuple[float, float, float, float]:
    points = sorted(
        sequence, key=lambda point: float(point.get("time", int(point["frame"]) / 30))
    )
    if not points:
        raise ValueError("Cannot interpolate an empty annotation sequence")
    before, after = points[0], points[-1]
    for left, right in pairwise(points):
        left_time = float(left.get("time", int(left["frame"]) / 30))
        right_time = float(right.get("time", int(right["frame"]) / 30))
        if left_time <= time <= right_time:
            before, after = left, right
            break
        if time < left_time:
            before = after = left
            break
    before_time = float(before.get("time", int(before["frame"]) / 30))
    after_time = float(after.get("time", int(after["frame"]) / 30))
    alpha = (
        0.0
        if after_time == before_time
        else min(1.0, max(0.0, (time - before_time) / (after_time - before_time)))
    )
    return tuple(
        float(before[key]) + alpha * (float(after[key]) - float(before[key]))
        for key in ("x", "y", "width", "height")
    )


def dense_boxes(task: dict[str, Any]) -> list[tuple[int, float, float, float, float]]:
    value = result_value(task)
    if value is None:
        return []
    labels = value.get("labels", [])
    if len(labels) != 1 or labels[0] not in CLASS_IDS:
        raise ProtocolError(f"Task {task['id']} has an unsupported NIR label: {labels}")
    class_id = CLASS_IDS[labels[0]]
    boxes = []
    for frame_index in range(1, 11):
        x, y, width, height = interpolate_box(value["sequence"], frame_index / 10)
        x, y = max(0.0, x), max(0.0, y)
        width, height = min(width, 100.0 - x), min(height, 100.0 - y)
        if width <= 0 or height <= 0:
            raise ProtocolError(
                f"Task {task['id']} has an invalid box at frame {frame_index}"
            )
        boxes.append((class_id, x / 100, y / 100, width / 100, height / 100))
    return boxes


def load_ratio_tasks(protocol: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result = {}
    for ratio in NIR_RATIOS:
        spec = protocol["dataset"]["ratio_annotations"][ratio]
        path = DATA_ROOT.parent / spec["path"]
        if sha256_file(path) != spec["sha256"]:
            raise ProtocolError(f"NIR ratio {ratio} annotation fingerprint mismatch")
        result[ratio] = json.loads(path.read_text(encoding="utf-8"))
    eval_a = [
        signature(task)
        for task in result["1to2"]
        if task["data"]["split"] in {"val", "test"}
    ]
    eval_b = [
        signature(task)
        for task in result["1to6"]
        if task["data"]["split"] in {"val", "test"}
    ]
    if eval_a != eval_b:
        raise ProtocolError("Validation/test identity differs across NIR ratios")
    return result


def validate_counts(
    tasks_by_ratio: dict[str, list[dict[str, Any]]], protocol: dict[str, Any]
) -> None:
    expected = protocol["dataset"]["expected"]
    for ratio, tasks in tasks_by_ratio.items():
        grouped = defaultdict(list)
        for task in tasks:
            grouped[task["data"]["split"]].append(task)
        train = grouped["train"]
        positives = sum(result_value(task) is not None for task in train)
        negatives = len(train) - positives
        spec = expected[f"ratio_{ratio}_train"]
        if (positives, negatives, len(train)) != (
            spec["positive_snippets"],
            spec["negative_snippets"],
            spec["snippets"],
        ):
            raise ProtocolError(
                f"NIR ratio {ratio} training counts do not match the frozen protocol"
            )
        for split in ("val", "test"):
            items = grouped[split]
            positives = sum(result_value(task) is not None for task in items)
            negatives = len(items) - positives
            spec = expected[split]
            if (positives, negatives, len(items)) != (
                spec["positive_snippets"],
                spec["negative_snippets"],
                spec["snippets"],
            ):
                raise ProtocolError(
                    f"NIR {split} counts do not match the frozen protocol"
                )
    training_positive = {
        ratio: {
            signature(task)
            for task in tasks
            if task["data"]["split"] == "train" and result_value(task) is not None
        }
        for ratio, tasks in tasks_by_ratio.items()
    }
    training_negative = {
        ratio: {
            signature(task)
            for task in tasks
            if task["data"]["split"] == "train" and result_value(task) is None
        }
        for ratio, tasks in tasks_by_ratio.items()
    }
    if training_positive["1to2"] != training_positive["1to6"]:
        raise ProtocolError("NIR positive training snippets differ across ratios")
    if not training_negative["1to2"] < training_negative["1to6"]:
        raise ProtocolError(
            "NIR 1:2 negatives must be a strict subset of the 1:6 negative pool"
        )


def image_name(task_id: int, frame_index: int) -> str:
    return f"task_{task_id:05d}_frame_{frame_index:02d}.jpg"


def write_derived(tasks_by_ratio: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    output = DATA_ROOT / "processed" / "NIR"
    labels = output / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    all_tasks = {
        signature(task): task for tasks in tasks_by_ratio.values() for task in tasks
    }
    boxes_by_task = {key: dense_boxes(task) for key, task in all_tasks.items()}
    for key, task in all_tasks.items():
        boxes = boxes_by_task[key]
        for frame_index in range(1, 11):
            lines = []
            if boxes:
                class_id, x, y, width, height = boxes[frame_index - 1]
                lines.append(
                    f"{class_id} {x + width / 2:.6f} {y + height / 2:.6f} {width:.6f} {height:.6f}"
                )
            (
                labels / Path(image_name(task["id"], frame_index)).with_suffix(".txt")
            ).write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
                newline="\n",
            )

    def build_coco(
        tasks: list[dict[str, Any]], description: str, *, one_based: bool
    ) -> dict[str, Any]:
        images, annotations = [], []
        annotation_id = 1
        for task in tasks:
            boxes = boxes_by_task[signature(task)]
            for frame_index in range(1, 11):
                image_id = int(task["id"]) * 10 + frame_index
                images.append(
                    {
                        "id": image_id,
                        "file_name": image_name(task["id"], frame_index),
                        "width": 640,
                        "height": 640,
                    }
                )
                if boxes:
                    class_id, x, y, width, height = boxes[frame_index - 1]
                    bbox = [
                        round(x * 640, 6),
                        round(y * 640, 6),
                        round(width * 640, 6),
                        round(height * 640, 6),
                    ]
                    annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id + int(one_based),
                            "bbox": bbox,
                            "area": round(bbox[2] * bbox[3], 6),
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1
        return {
            "info": {"description": description, "fps": 10},
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": int(one_based), "name": "drinking"},
                {"id": 1 + int(one_based), "name": "phone_use"},
            ],
        }

    shared_eval = {
        split: [
            task for task in tasks_by_ratio["1to2"] if task["data"]["split"] == split
        ]
        for split in ("val", "test")
    }
    for split, tasks in shared_eval.items():
        path = output / "coco" / "evaluation" / f"instances_{split}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                build_coco(tasks, f"NIR shared {split} at 10 FPS", one_based=True),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        dfine_path = (
            output / "coco" / "dfine" / "evaluation" / f"instances_{split}.json"
        )
        dfine_path.parent.mkdir(parents=True, exist_ok=True)
        dfine_path.write_text(
            json.dumps(
                build_coco(tasks, f"NIR shared {split} for D-FINE", one_based=False),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        list_path = output / "yolo" / "evaluation" / f"{split}.txt"
        list_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"./../../images/{image_name(task['id'], frame)}"
            for task in tasks
            for frame in range(1, 11)
        ]
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    for ratio, tasks in tasks_by_ratio.items():
        train = [task for task in tasks if task["data"]["split"] == "train"]
        coco_path = (
            output / "coco" / "dfine" / f"ratio_{ratio}" / "instances_train.json"
        )
        coco_path.parent.mkdir(parents=True, exist_ok=True)
        coco_path.write_text(
            json.dumps(
                build_coco(
                    train, f"NIR ratio {ratio} train at 10 FPS", one_based=False
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        list_path = output / "yolo" / f"ratio_{ratio}" / "train.txt"
        list_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"./../../images/{image_name(task['id'], frame)}"
            for task in train
            for frame in range(1, 11)
        ]
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return {"unique_snippets": len(all_tasks), "unique_frames": len(all_tasks) * 10}


def source_index(
    source_root: Path, source_videos: dict[str, Any]
) -> dict[tuple[str, str], Path]:
    files: dict[str, list[Path]] = defaultdict(list)
    for path in source_root.rglob("*.mp4"):
        files[path.name].append(path)
    index = {}
    for entry in source_videos["videos"]:
        matches = files.get(entry["filename"], [])
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one Drive&Act source named {entry['filename']}, found {len(matches)}"
            )
        index[(entry["subject"], entry["video_run"])] = matches[0]
    return index


def extract_frames(
    tasks_by_ratio: dict[str, list[dict[str, Any]]], source_root: Path
) -> None:
    import cv2

    source_videos = json.loads(
        (DATA_ROOT / "annotations" / "NIR" / "source_videos.json").read_text(
            encoding="utf-8"
        )
    )
    sources = source_index(source_root, source_videos)
    all_tasks = {
        signature(task): task for tasks in tasks_by_ratio.values() for task in tasks
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in all_tasks.values():
        grouped[(task["data"]["subject"], task["data"]["video_run"])].append(task)
    output = DATA_ROOT / "processed" / "NIR" / "images"
    output.mkdir(parents=True, exist_ok=True)
    for source_key, tasks in sorted(grouped.items()):
        capture = cv2.VideoCapture(str(sources[source_key]))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {sources[source_key]}")
        for task in sorted(tasks, key=lambda item: int(item["data"]["frame_start"])):
            targets = [output / image_name(task["id"], frame) for frame in range(1, 11)]
            if all(path.is_file() for path in targets):
                continue
            start = int(task["data"]["frame_start"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            selected = {}
            for relative_frame in range(30):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"Source ended at task {task['id']}, relative frame {relative_frame}"
                    )
                if relative_frame in OFFSETS:
                    crop = frame[0:1024, 128:1152]
                    selected[relative_frame] = cv2.resize(
                        crop, (640, 640), interpolation=cv2.INTER_AREA
                    )
            for frame_index, offset in enumerate(OFFSETS, start=1):
                if not cv2.imwrite(
                    str(targets[frame_index - 1]),
                    selected[offset],
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                ):
                    raise RuntimeError(f"Failed to write {targets[frame_index - 1]}")
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Build labels and manifests without extracting frames",
    )
    parser.add_argument("--source-root", type=Path, default=DATA_ROOT / "Drive&Act")
    args = parser.parse_args()
    protocol = validate_protocol("NIR")
    tasks = load_ratio_tasks(protocol)
    validate_counts(tasks, protocol)
    report = write_derived(tasks)
    if not args.annotations_only:
        extract_frames(tasks, args.source_root.resolve())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
