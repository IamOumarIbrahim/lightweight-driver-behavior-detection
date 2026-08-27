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


def test_nir_figure_uses_grouped_columns_and_ratio_divider() -> None:
    script = (REPO_ROOT / "tools" / "publication" / "figures.R").read_text(
        encoding="utf-8"
    )
    start = script.index("build_nir_training_negative_exposure <- function")
    end = script.index("if (!only_nir)", start)
    nir_figure = script[start:end]

    assert "geom_col(" in nir_figure
    assert "geom_vline(" in nir_figure
    assert "xintercept = 1.5" in nir_figure
    assert "ratio_means <- aggregate(" in nir_figure
    assert "geom_segment(" in nir_figure
    assert 'colour = "red3"' in nir_figure
    assert "linewidth = 1.1" in nir_figure
    assert 'ratio_means$ratio == "1:2", 0.5, 1.5' in nir_figure
    assert 'ratio_means$ratio == "1:2", 1.5, 2.5' in nir_figure
    assert "geom_line(" not in nir_figure


def test_late_manuscript_figures_are_anchored_to_avoid_column_gap() -> None:
    manuscript = (REPO_ROOT / "docs" / "manuscript" / "main.tex").read_text(
        encoding="utf-8"
    )

    assert r"\usepackage{float}" in manuscript
    assert manuscript.count(r"\begin{figure}[H]") == 2


def test_r_package_lock_and_figure_hashes() -> None:
    with (REPO_ROOT / "tools" / "publication" / "R-packages.lock.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        packages = list(csv.DictReader(handle))
    versions = {row["Package"]: row["Version"] for row in packages}

    assert versions["ggplot2"] == "4.0.3"
    assert versions["jsonlite"] == "2.0.0"
    assert versions["png"] == "0.1-9"
    assert len(versions) == len(packages)
    manifests = (
        REPO_ROOT / "results" / "summary" / "figures" / "protocol_workflow.manifest.json",
        REPO_ROOT / "results" / "RGB" / "summary" / "figures" / "normalized_model_comparison.manifest.json",
        REPO_ROOT / "results" / "RGB" / "summary" / "figures" / "accuracy_vs_speed.manifest.json",
        REPO_ROOT / "results" / "RGB" / "summary" / "figures" / "per_class_ap.manifest.json",
        REPO_ROOT / "results" / "RGB" / "summary" / "figures" / "qualitative_examples.manifest.json",
        REPO_ROOT / "results" / "RGB" / "summary" / "figures" / "subject_sensitivity.manifest.json",
        REPO_ROOT / "results" / "RGB" / "summary" / "figures" / "validation_operating_point.manifest.json",
        REPO_ROOT / "results" / "NIR" / "summary" / "figures" / "training_negative_exposure.manifest.json",
    )
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["generator"] == "ggplot2"
        for output in manifest["outputs"].values():
            path = REPO_ROOT / output["path"]
            assert path.is_file()
            assert sha256(path) == output["sha256"]
        for source in manifest.get("inputs", []):
            path = REPO_ROOT / source["path"]
            assert path.is_file()
            assert sha256(path) == source["sha256"]


def test_validation_sweep_is_complete_and_path_safe() -> None:
    sweep_path = (
        REPO_ROOT
        / "results"
        / "RGB"
        / "summary"
        / "validation_operating_point_sweep.csv"
    )
    text = sweep_path.read_text(encoding="utf-8")
    assert "C:\\" not in text
    with sweep_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * 3 * 99
    assert sum(row["selected_primary"] == "true" for row in rows) == 6
    assert {row["model_id"] for row in rows} == {"yolo11n", "yolo26n"}


def test_nir_exposure_figure_tracks_pending_extension_models() -> None:
    source_path = (
        REPO_ROOT
        / "results"
        / "NIR"
        / "summary"
        / "training_negative_exposure_source.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    assert source["status"] == "partial"
    assert source["seed"] == 13
    assert source["ratios"] == ["1:2", "1:6"]
    assert source["completed_models"] == ["yolo11n", "yolo26n", "dfine_n"]
    assert source["pending_models"] == [
        "ssdlite_mobilenet_v3_large",
        "rtdetrv2_s",
        "yolox_nano",
        "yolov10n",
        "yolov8n",
    ]
    assert len(source["rows"]) == 6
    for row in source["rows"]:
        metrics_path = REPO_ROOT / row["metrics_path"]
        assert metrics_path.is_file()
        assert sha256(metrics_path) == row["metrics_sha256"]
        assert 0 < row["map_50_95"] < 1
        assert 0 < row["micro_f1"] < 1
        assert 0 < row["macro_f1"] < 1
        assert row["false_detections_per_100_negative_frames"] >= 0
        assert row["tensor_to_final_p50_ms"] > 0
        assert row["tensor_to_final_sustained_fps"] > 0
    manifest_path = (
        source_path.parent / "figures" / "training_negative_exposure.manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert manifest["models"] == source["completed_models"]
    assert manifest["pending_models"] == source["pending_models"]
