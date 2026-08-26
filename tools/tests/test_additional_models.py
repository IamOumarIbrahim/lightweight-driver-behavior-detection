from pathlib import Path

import yaml

from tools.backends.docker_backend import command, container_path, image_tag
from tools.benchmark.paths import REPO_ROOT
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


def test_rtmdet_config_preserves_negatives_and_stage_switch():
    source = (
        REPO_ROOT / "configs" / "NIR" / "rtmdet" / "base.py"
    ).read_text(encoding="utf-8")
    assert 'filter_cfg=dict(filter_empty_gt=False' in source
    assert "accumulative_counts=4" in source
    assert "stage2_epoch = 93" in source
