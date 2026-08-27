from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_dfine_training_and_protected_test_provenance_is_complete() -> None:
    path = REPO_ROOT / "results" / "RGB" / "dfine_n" / "training_runs.json"
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)

    assert "C:\\" not in text
    assert payload["status"] == "protected_test_complete"
    assert payload["training_controls"]["epochs"] == 220
    assert payload["training_controls"]["effective_batch_size"] == 32
    assert [run["training_seed"] for run in payload["runs"]] == [13, 37, 73]

    for run in payload["runs"]:
        assert run["epochs_completed"] == 220
        assert run["final_epoch_index"] == 219
        assert run["parameters"] == 3_724_463
        observation = run["training_time_validation_observation"]
        assert 0 <= observation["best_epoch_index"] <= 219
        assert 0 < observation["best_map_50_95"] <= 1
        assert 0 < observation["best_map_50"] <= 1
        validation = run["frozen_validation"]
        assert validation["selected_epoch_index"] == observation["best_epoch_index"]
        assert 0 < validation["map_50_95"] <= 1
        assert 0 < validation["map_50"] <= 1
        assert 0.01 <= validation["threshold"] <= 0.99
        assert 0 < validation["micro_f1"] <= 1
        protected = run["protected_test"]
        assert protected["manifest_id"] == validation["manifest"]["id"]
        assert protected["threshold"] == validation["threshold"]
        assert 0 < protected["map_50_95"] <= 1
        assert 0 < protected["map_50"] <= 1
        assert 0 < protected["micro_f1"] <= 1
        assert 0 < protected["macro_f1"] <= 1
        assert protected["negative_frames"] == 2599
        assert protected["result"]["bytes"] > 0
        assert re.fullmatch(r"[0-9a-f]{64}", protected["result"]["sha256"])
        for artifact in run["local_artifact_fingerprints"].values():
            assert artifact["bytes"] > 0
            assert re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])

    publication = payload["publication_state"]
    assert publication["checkpoint_selection"] == "frozen"
    assert publication["confidence_threshold_calibration"] == "frozen"
    assert publication["protected_test"] == "completed_once_per_seed"
    assert publication["eligible_for_dfine_aggregate"] is True
    assert publication["eligible_for_mixed_suite_nine_run_aggregate"] is False
    assert payload["validation_suite"]["manifests"] == 9
    aggregate = payload["protected_test_aggregate"]
    assert aggregate["runs"] == 3
    assert aggregate["training_seeds"] == [13, 37, 73]
    assert 0 < aggregate["map_50_95"]["mean"] <= 1
    assert 0 < aggregate["macro_f1"]["mean"] <= 1
    assert aggregate["artifact"]["bytes"] > 0
    assert re.fullmatch(r"[0-9a-f]{64}", aggregate["artifact"]["sha256"])
