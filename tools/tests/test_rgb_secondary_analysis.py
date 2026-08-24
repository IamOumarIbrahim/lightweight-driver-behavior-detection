from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_public_rgb_prediction_envelopes_are_complete_and_path_safe() -> None:
    for model in ("yolo11n", "yolo26n"):
        for seed in (13, 37, 73):
            path = (
                REPO_ROOT
                / "results"
                / "RGB"
                / model
                / f"seed_{seed}"
                / "test_predictions.json"
            )
            envelope = json.loads(path.read_text(encoding="utf-8"))
            assert envelope["model_id"] == model
            assert envelope["training_seed"] == seed
            assert envelope["predictions"]
            assert not Path(envelope["provenance"]["source_result"]).is_absolute()
            assert "C:\\" not in json.dumps(envelope)


def test_secondary_analysis_preserves_audit_and_paired_interpretation() -> None:
    path = REPO_ROOT / "results" / "RGB" / "summary" / "secondary_analysis.json"
    analysis = json.loads(path.read_text(encoding="utf-8"))

    assert analysis["annotation_audit"]["images"] == 15723
    assert analysis["annotation_audit"]["annotations"] == 3001
    assert set(analysis["annotation_audit"]["problems"].values()) == {0}
    assert analysis["model_summary"]["yolo11n"]["macro_f1"]["mean"] == pytest.approx(
        0.8342294249
    )
    assert analysis["model_summary"]["yolo26n"]["macro_f1"]["mean"] == pytest.approx(
        0.7938628297
    )
    paired = analysis["paired_seed_differences"]
    yawn = [row for row in paired if row["metric"] == "ap_50_95_yawning"]
    drinking = [row for row in paired if row["metric"] == "ap_50_95_drinking"]
    assert len(yawn) == 3 and {row["direction_across_three_seeds"] for row in yawn} == {
        "yolo11n_higher"
    }
    assert len(drinking) == 3 and {
        row["direction_across_three_seeds"] for row in drinking
    } == {"mixed"}
