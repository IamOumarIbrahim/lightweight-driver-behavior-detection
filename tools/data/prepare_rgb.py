"""Build RGB YOLO and D-FINE artifacts from published COCO annotations."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.benchmark.paths import DATA_ROOT
from tools.benchmark.protocol import validate_protocol


def subject_of(image: dict[str, Any]) -> str:
    parts = Path(image["file_name"]).parts
    if len(parts) < 2 or parts[0] != "images":
        raise ValueError(f"Non-canonical RGB image path: {image['file_name']}")
    return parts[1]


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


def build(*, check_images: bool = True) -> dict[str, int]:
    validate_protocol("RGB")
    annotation_path = DATA_ROOT / "annotations" / "RGB" / "annotations.json"
    split_path = DATA_ROOT / "annotations" / "RGB" / "splits.json"
    output = DATA_ROOT / "processed" / "RGB"
    images_root = output / "images"
    master = json.loads(annotation_path.read_text(encoding="utf-8"))
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    split_key = {"train": "train", "validation": "val", "test": "test"}
    subject_split = {
        subject: target
        for source, target in split_key.items()
        for subject in splits[source]
    }
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in master["annotations"]:
        by_image[int(annotation["image_id"])].append(annotation)

    split_images: dict[str, list[dict[str, Any]]] = {
        name: [] for name in split_key.values()
    }
    for image in master["images"]:
        split_images[subject_split[subject_of(image)]].append(image)
    random.Random(13).shuffle(split_images["train"])

    label_root = output / "labels"
    yolo_root = output / "yolo"
    for split, images in split_images.items():
        lines = []
        for image in images:
            relative = Path(*Path(image["file_name"]).parts[1:])
            image_path = images_root / relative
            if check_images and not image_path.is_file():
                raise FileNotFoundError(f"Missing RGB frame: {image_path}")
            label_path = label_root / relative.with_suffix(".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)
            labels = []
            for annotation in by_image[int(image["id"])]:
                x, y, width, height = map(float, annotation["bbox"])
                labels.append(
                    f"{int(annotation['category_id']) - 1} "
                    f"{(x + width / 2) / 640:.6f} {(y + height / 2) / 640:.6f} "
                    f"{width / 640:.6f} {height / 640:.6f}"
                )
            label_path.write_text(
                "\n".join(labels) + ("\n" if labels else ""),
                encoding="utf-8",
                newline="\n",
            )
            lines.append("./../" + (Path("images") / relative).as_posix())
        yolo_root.mkdir(parents=True, exist_ok=True)
        (yolo_root / f"{split}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
        )

        image_rank = {int(image["id"]): index for index, image in enumerate(images)}
        derived_images = [
            {**image, "file_name": Path(*Path(image["file_name"]).parts[1:]).as_posix()}
            for image in images
        ]
        derived_annotations = [
            {**annotation, "category_id": int(annotation["category_id"]) - 1}
            for annotation in master["annotations"]
            if int(annotation["image_id"]) in image_rank
        ]
        derived_annotations.sort(
            key=lambda annotation: (
                image_rank[int(annotation["image_id"])],
                int(annotation["id"]),
            )
        )
        write_json(
            output / "coco" / "dfine" / f"instances_{split}.json",
            {
                "info": {"description": f"RGB {split} split for D-FINE"},
                "licenses": master.get("licenses", []),
                "images": derived_images,
                "annotations": derived_annotations,
                "categories": [
                    {**category, "id": int(category["id"]) - 1}
                    for category in master["categories"]
                ],
            },
        )
        evaluation_annotations = [
            dict(annotation)
            for annotation in master["annotations"]
            if int(annotation["image_id"]) in image_rank
        ]
        evaluation_annotations.sort(
            key=lambda annotation: (
                image_rank[int(annotation["image_id"])],
                int(annotation["id"]),
            )
        )
        write_json(
            output / "coco" / "evaluation" / f"instances_{split}.json",
            {
                "info": {"description": f"RGB {split} evaluation split"},
                "licenses": master.get("licenses", []),
                "images": derived_images,
                "annotations": evaluation_annotations,
                "categories": master["categories"],
            },
        )

    return {"images": len(master["images"]), "annotations": len(master["annotations"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-only",
        action="store_true",
        help="Build labels before raw frames are present",
    )
    args = parser.parse_args()
    print(json.dumps(build(check_images=not args.annotations_only), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
