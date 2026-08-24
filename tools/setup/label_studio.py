"""Prepare a local Label Studio review workspace for the published NIR annotations."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from tools.benchmark.paths import CONFIGS_ROOT, DATA_ROOT
from tools.data.prepare_nir import dense_boxes, result_value


def to_ten_fps(task: dict) -> None:
    """Rewrite a published 30-frame tracklet for a ten-frame review video."""

    value = result_value(task)
    if value is None:
        return
    boxes = dense_boxes(task)
    label = value["labels"][0]
    sequence = []
    for frame, (_, x, y, width, height) in enumerate(boxes, start=1):
        sequence.append(
            {
                "frame": frame,
                "time": frame / 10,
                "x": x * 100,
                "y": y * 100,
                "width": width * 100,
                "height": height * 100,
                "rotation": 0,
                "enabled": True,
            }
        )
    task["annotations"] = [
        {
            "result": [
                {
                    "type": "videorectangle",
                    "value": {
                        "framesCount": 10,
                        "duration": 1.0,
                        "labels": [label],
                        "sequence": sequence,
                    },
                }
            ]
        }
    ]


def main() -> int:
    workspace = DATA_ROOT / "label_studio"
    snippets = workspace / "snippet_videos"
    workspace.mkdir(parents=True, exist_ok=True)
    snippets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIGS_ROOT / "label_studio.xml", workspace / "label_config.xml")
    tasks = json.loads(
        (DATA_ROOT / "annotations" / "NIR" / "annotations.json").read_text(
            encoding="utf-8"
        )
    )
    for task in tasks:
        to_ten_fps(task)
        filename = Path(task["data"]["video"]).name
        task["data"]["video"] = f"/data/local-files/?d=snippet_videos/{filename}"
    (workspace / "annotations_for_import.json").write_text(
        json.dumps(tasks, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Prepared {workspace}")
    print(
        "Start Label Studio, create a project with label_config.xml, then import annotations_for_import.json."
    )
    print(
        "Build 10-FPS review videos with scripts/data/05_build_nir_review_snippets.bat."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
