from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_rgb_aggregate_is_publication_ready() -> None:
    aggregate_path = (
        REPO_ROOT / "results" / "RGB" / "summary" / "final_benchmark_aggregate.json"
    )
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    rows = {row["model_id"]: row for row in aggregate["rows"]}

    assert list(rows) == ["yolo11n", "yolo26n"]
    assert all(row["runs"] == 3 for row in rows.values())
    assert all(
        row["tensor_to_final_detections_sustained_fps"]["mean"] > 0
        for row in rows.values()
    )
    assert all(0 < row["map_50_95"]["mean"] < 1 for row in rows.values())


def test_publication_pipeline_is_ggplot2_only() -> None:
    script = (REPO_ROOT / "tools" / "publication" / "figures.R").read_text(
        encoding="utf-8"
    )
    launcher = (REPO_ROOT / "scripts" / "publication" / "build_figures.bat").read_text(
        encoding="utf-8"
    )

    assert "library(ggplot2)" in script
    assert "ggplot(" in script
    assert "tools.publication.figures" not in launcher
    assert not (REPO_ROOT / "tools" / "publication" / "figures.py").exists()


def test_r_package_lock_and_figure_hashes() -> None:
    with (REPO_ROOT / "tools" / "publication" / "R-packages.lock.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        packages = list(csv.DictReader(handle))
    versions = {row["Package"]: row["Version"] for row in packages}

    assert versions["ggplot2"] == "4.0.3"
    assert versions["jsonlite"] == "2.0.0"
    assert len(versions) == len(packages)
    for stem in ("accuracy_vs_speed", "per_class_ap"):
        manifest_path = (
            REPO_ROOT
            / "results"
            / "RGB"
            / "summary"
            / "figures"
            / f"{stem}.manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["generator"] == "ggplot2"
        for output in manifest["outputs"].values():
            path = REPO_ROOT / output["path"]
            assert path.is_file()
            assert sha256(path) == output["sha256"]
