import json

import yaml

from tools.benchmark.paths import REPO_ROOT, RESULTS_ROOT, result_dir, run_dir


def test_repo_root_is_checkout():
    assert (REPO_ROOT / ".git").is_dir()
    assert (REPO_ROOT / "README.md").is_file()


def test_results_root_contains_only_organized_artifacts():
    root_files = sorted(path.name for path in RESULTS_ROOT.iterdir() if path.is_file())
    assert root_files == ["README.md"]


def test_release_metadata_uses_current_version_and_concept_doi():
    citation = yaml.safe_load(
        (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    )
    zenodo = json.loads((REPO_ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert citation["version"] == zenodo["version"] == "1.0.1"
    assert citation["doi"] == "10.5281/zenodo.22172309"
    assert citation["repository-artifact"].endswith(citation["doi"])
    assert citation["doi"] in readme


def test_canonical_artifact_paths():
    assert (
        run_dir("RGB", "yolo11n", seed=13).relative_to(REPO_ROOT).as_posix()
        == "runs/RGB/yolo11n/seed_13"
    )
    assert (
        result_dir("NIR", "dfine_n", ratio="1to6").relative_to(REPO_ROOT).as_posix()
        == "results/NIR/dfine_n/ratio_1to6"
    )
    assert (
        run_dir("NIR", "rtdetrv2_s", ratio="1to2")
        .relative_to(REPO_ROOT)
        .as_posix()
        == "runs/NIR/rtdetrv2_s/ratio_1to2"
    )
