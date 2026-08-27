import json
from pathlib import Path

import pytest
import yaml

from tools.backends import container_entry
from tools.backends.docker_backend import command, container_path, image_tag
from tools.benchmark.paths import NIR_MODELS, NIR_RATIOS, REPO_ROOT
from tools.workflow.train import build_plan


def test_additional_model_plans_are_nir_only():
    for model, suffix in (
        ("rtmdet_tiny", "rtmdet/ratio_1to2.py"),
        ("efficientdet_d1", "efficientdet/ratio_1to2.yaml"),
    ):
        plan = build_plan("NIR", model, None, "1to2")
        assert plan["epochs"] == 100
        assert plan["batch"] == 8
        assert plan["nbs"] == 32
        assert Path(plan["additional_config"]).as_posix().endswith(suffix)


def test_docker_command_is_stable_and_gpu_scoped():
    result = command("train", "--model", "rtmdet_tiny", "--ratio", "1to2")
    assert result[:3] == ["docker", "run", "--rm"]
    assert result[result.index("--gpus") + 1] == "device=0"
    assert image_tag() in result
    assert container_path(REPO_ROOT / "configs" / "NIR") == "/workspace/configs/NIR"
    doctor = command("doctor")
    assert doctor[-1] == "doctor"
    assert "--volume" in doctor
    assert doctor[doctor.index("--gpus") + 1] == "device=0"


def test_efficientdet_patch_and_config_freeze_accumulation():
    patch = (
        REPO_ROOT / "configs" / "patches" / "efficientdet-gradient-accumulation.patch"
    ).read_text(encoding="utf-8")
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "NIR" / "efficientdet" / "base.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["model"] == "tf_efficientdet_d1"
    assert config["batch_size"] == 8
    assert config["grad_accum_steps"] == 4
    assert "window_samples" in patch
    assert "update_grad=should_step" in patch
    entrypoint = (
        REPO_ROOT / "tools" / "backends" / "container_entry.py"
    ).read_text(encoding="utf-8")
    assert "shutil.copy2(source_weight, cached_weight)" in entrypoint
    assert "def require_cuda_runtime" in entrypoint
    assert "def validate_training_inputs" in entrypoint


def test_container_inventory_preflight_requires_every_mounted_image(
    tmp_path, monkeypatch
):
    image_root = tmp_path / "images"
    image_root.mkdir()
    annotation_path = tmp_path / "instances.json"
    payload = {
        "images": [
            {"id": 1, "file_name": "one.jpg"},
            {"id": 2, "file_name": "two.jpg"},
        ],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "drinking"},
            {"id": 1, "name": "phone_use"},
        ],
    }
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")
    (image_root / "one.jpg").write_bytes(b"image")
    monkeypatch.setattr(container_entry, "IMAGE_ROOT", image_root)

    with pytest.raises(RuntimeError, match="1 absent; first=two.jpg"):
        container_entry._validate_coco_inventory(
            annotation_path, expected_images=2, expected_category_ids=[0, 1]
        )

    (image_root / "two.jpg").write_bytes(b"image")
    report = container_entry._validate_coco_inventory(
        annotation_path, expected_images=2, expected_category_ids=[0, 1]
    )
    assert report["images"] == 2


def test_backend_setup_smoke_mounts_the_repository_before_importing_entrypoint():
    source = (REPO_ROOT / "tools" / "setup" / "backends.py").read_text(
        encoding="utf-8"
    )
    assert 'docker_command("--help", gpu=False)' in source
    assert "verify_gpu_runtime()" in source


def test_rtmdet_config_preserves_negatives_and_stage_switch():
    source = (
        REPO_ROOT / "configs" / "NIR" / "rtmdet" / "base.py"
    ).read_text(encoding="utf-8")
    assert 'filter_cfg=dict(filter_empty_gt=False' in source
    assert "accumulative_counts=4" in source
    assert "stage2_epoch = 93" in source


def test_ten_run_launchers_match_the_frozen_matrix():
    assert len(NIR_MODELS) * len(NIR_RATIOS) == 10
    scripts = REPO_ROOT / "scripts" / "NIR"
    assert (scripts / "train_all_ten.bat").is_file()
    assert (scripts / "validate_all_ten.bat").is_file()
    assert (scripts / "test_all_ten.bat").is_file()
    assert not (scripts / "train_all_six.bat").exists()
    for model in NIR_MODELS:
        for ratio in NIR_RATIOS:
            assert (scripts / f"train_{model}_ratio_{ratio}.bat").is_file()
