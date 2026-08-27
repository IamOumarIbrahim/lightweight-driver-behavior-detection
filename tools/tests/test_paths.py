from tools.benchmark.paths import REPO_ROOT, result_dir, run_dir


def test_repo_root_is_checkout():
    assert (REPO_ROOT / ".git").is_dir()
    assert (REPO_ROOT / "README.md").is_file()


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
