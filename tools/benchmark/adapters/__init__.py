"""Frozen detector adapter registry."""

from .base import DetectorAdapter
from .dfine import DFineAdapter
from .ultralytics import UltralyticsAdapter


def create_adapter(
    model_id: str,
    checkpoint,
    device: str = "cuda:0",
    allow_pretrained_head_mismatch: bool = False,
    class_count: int = 4,
    config_path=None,
) -> DetectorAdapter:
    if model_id in {"yolo11n", "yolo26n"}:
        return UltralyticsAdapter(
            model_id,
            checkpoint,
            device=device,
            allow_pretrained_head_mismatch=allow_pretrained_head_mismatch,
            class_count=class_count,
            config_path=config_path,
        )
    if model_id == "dfine_n":
        return DFineAdapter(
            model_id,
            checkpoint,
            device=device,
            allow_pretrained_head_mismatch=allow_pretrained_head_mismatch,
            class_count=class_count,
            config_path=config_path,
        )
    raise ValueError(f"Unknown frozen model: {model_id}")


__all__ = ["DFineAdapter", "DetectorAdapter", "UltralyticsAdapter", "create_adapter"]
