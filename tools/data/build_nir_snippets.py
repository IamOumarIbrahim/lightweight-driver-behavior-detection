"""Build portable 10-FPS NIR review videos for the Label Studio workspace."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tools.benchmark.paths import DATA_ROOT
from tools.data.prepare_nir import OFFSETS, source_index


def valid_video(path: Path) -> bool:
    import cv2

    if not path.is_file() or path.stat().st_size == 0:
        return False
    capture = cv2.VideoCapture(str(path))
    valid = capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    capture.release()
    return valid


def build(source_root: Path, output: Path) -> dict[str, int]:
    import cv2

    annotations = json.loads(
        (DATA_ROOT / "annotations" / "NIR" / "annotations.json").read_text(
            encoding="utf-8"
        )
    )
    source_videos = json.loads(
        (DATA_ROOT / "annotations" / "NIR" / "source_videos.json").read_text(
            encoding="utf-8"
        )
    )
    sources = source_index(source_root, source_videos)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for task in annotations:
        grouped[(task["data"]["subject"], task["data"]["video_run"])].append(task)

    output.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    for source_key, tasks in sorted(grouped.items()):
        capture = cv2.VideoCapture(str(sources[source_key]))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {sources[source_key]}")
        try:
            for task in sorted(
                tasks, key=lambda item: int(item["data"]["frame_start"])
            ):
                destination = output / Path(task["data"]["video"]).name
                if valid_video(destination):
                    skipped += 1
                    continue
                temporary = destination.with_name(destination.stem + ".partial.mp4")
                writer = cv2.VideoWriter(str(temporary), fourcc, 10.0, (640, 640))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create {temporary}")
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(task["data"]["frame_start"]))
                try:
                    for relative_frame in range(30):
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError(
                                f"Source ended at task {task['id']}, relative frame {relative_frame}"
                            )
                        if relative_frame in OFFSETS:
                            crop = frame[0:1024, 128:1152]
                            writer.write(
                                cv2.resize(
                                    crop, (640, 640), interpolation=cv2.INTER_AREA
                                )
                            )
                finally:
                    writer.release()
                if not valid_video(temporary):
                    raise RuntimeError(
                        f"Generated review video is invalid: {temporary}"
                    )
                temporary.replace(destination)
                written += 1
        finally:
            capture.release()
    return {"written": written, "already_valid": skipped, "total": len(annotations)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DATA_ROOT / "Drive&Act")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_ROOT / "label_studio" / "snippet_videos",
    )
    args = parser.parse_args()
    print(
        json.dumps(build(args.source_root.resolve(), args.output.resolve()), indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
