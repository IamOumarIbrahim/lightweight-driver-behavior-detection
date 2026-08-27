import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from tools.backends.train_ssdlite import CocoDetectionDataset
from tools.benchmark.adapters import (
    RTDETRv2Adapter,
    SSDLiteAdapter,
    UltralyticsAdapter,
    YOLOXAdapter,
    create_adapter,
)
from tools.benchmark.protocol import ProtocolError, load_backends, model_spec
from tools.workflow.evaluate import checkpoint_for
from tools.workflow.train import build_plan
from tools.workflow.train_new_nir import jobs


def test_native_backend_specs_are_pinned() -> None:
    backends = load_backends()
    assert backends["torchvision_ssdlite"]["version"] == "0.19.1+cu121"
    assert len(backends["rtdetrv2"]["commit"]) == 40
    assert len(backends["yolox"]["commit"]) == 40
    for backend in ("torchvision_ssdlite", "rtdetrv2", "yolox"):
        assert len(backends[backend]["weight"]["sha256"]) == 64


def test_new_model_specs_resolve_to_native_backends() -> None:
    expected = {
        "ssdlite_mobilenet_v3_large": "torchvision_ssdlite",
        "rtdetrv2_s": "rtdetrv2",
        "yolox_nano": "yolox",
        "yolov10n": "ultralytics",
        "yolov8n": "ultralytics",
    }
    for model_id, adapter in expected.items():
        model, backend = model_spec(model_id, "NIR")
        assert model["adapter"] == adapter
        assert isinstance(backend, dict)


def test_new_adapter_registry(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"adapter-construction-only")
    cases = {
        "ssdlite_mobilenet_v3_large": SSDLiteAdapter,
        "rtdetrv2_s": RTDETRv2Adapter,
        "yolox_nano": YOLOXAdapter,
        "yolov10n": UltralyticsAdapter,
        "yolov8n": UltralyticsAdapter,
    }
    for model_id, adapter_type in cases.items():
        assert isinstance(
            create_adapter(model_id, checkpoint, device="cpu", class_count=2),
            adapter_type,
        )


def test_new_training_plans_and_checkpoint_names(tmp_path: Path) -> None:
    for model_id, suffix in {
        "ssdlite_mobilenet_v3_large": "ssdlite/ratio_1to2.yaml",
        "rtdetrv2_s": "rtdetrv2/ratio_1to2.yml",
        "yolox_nano": "yolox/ratio_1to2.py",
    }.items():
        plan = build_plan("NIR", model_id, None, "1to2")
        assert plan["backend_config"].replace("\\", "/").endswith(suffix)

    (tmp_path / "best.pt").write_bytes(b"ssdlite")
    assert checkpoint_for("ssdlite_mobilenet_v3_large", tmp_path).name == "best.pt"
    (tmp_path / "best.pth").write_bytes(b"rtdetr")
    assert checkpoint_for("rtdetrv2_s", tmp_path).name == "best.pth"
    (tmp_path / "best_ckpt.pth").write_bytes(b"yolox")
    assert checkpoint_for("yolox_nano", tmp_path).name == "best_ckpt.pth"


def test_extension_models_are_nir_only() -> None:
    with pytest.raises(ProtocolError, match="not frozen for the RGB track"):
        build_plan("RGB", "yolov8n", 13, None)


def test_pending_launcher_selects_only_the_ten_new_runs() -> None:
    selected = jobs()
    assert len(selected) == 10
    assert {model for model, _ in selected} == {
        "ssdlite_mobilenet_v3_large",
        "rtdetrv2_s",
        "yolox_nano",
        "yolov10n",
        "yolov8n",
    }
    assert not {"yolo11n", "yolo26n", "dfine_n"} & {
        model for model, _ in selected
    }


def test_ssdlite_class_ids_and_validation_annotations_are_one_based(
    tmp_path: Path,
) -> None:
    config_path = Path("configs/NIR/ssdlite/base.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["train_category_id_offset"] == 1
    assert config["val_annotations"].endswith("coco/evaluation/instances_val.json")

    Image.new("RGB", (8, 8)).save(tmp_path / "sample.jpg")
    annotations = {
        "images": [{"id": 1, "file_name": "sample.jpg", "width": 8, "height": 8}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "bbox": [1, 1, 4, 4],
                "area": 16,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 0, "name": "drinking"}],
    }
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    dataset = CocoDetectionDataset(
        tmp_path, annotation_path, category_id_offset=config["train_category_id_offset"]
    )
    _, target = dataset[0]
    assert target["labels"].tolist() == [1]
