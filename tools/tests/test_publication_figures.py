from __future__ import annotations

from pathlib import Path

from tools.publication.figures import (
    build_accuracy_speed_figure,
    load_rgb_aggregate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_frozen_rgb_aggregate_is_publication_ready() -> None:
    rows = load_rgb_aggregate(
        REPO_ROOT / "results" / "RGB" / "summary" / "final_benchmark_aggregate.json"
    )

    assert [row.model_id for row in rows] == ["yolo11n", "yolo26n"]
    assert all(row.runs == 3 for row in rows)
    assert all(row.sustained_fps.mean > 0 for row in rows)
    assert all(0 < row.map_50_95.mean < 1 for row in rows)


def test_accuracy_speed_exports_all_formats(tmp_path: Path) -> None:
    rows = load_rgb_aggregate(
        REPO_ROOT / "results" / "RGB" / "summary" / "final_benchmark_aggregate.json"
    )
    outputs = build_accuracy_speed_figure(rows, tmp_path)

    assert set(outputs) == {"pdf", "svg", "png"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in outputs.values())
