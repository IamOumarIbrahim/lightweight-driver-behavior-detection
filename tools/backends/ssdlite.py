"""Native Torchvision SSDLite construction and checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tools.benchmark.protocol import ProtocolError


def build_ssdlite(class_count: int, *, input_size: int = 640) -> torch.nn.Module:
    """Build the official architecture with a benchmark-sized transform."""

    from torchvision.models.detection import ssdlite320_mobilenet_v3_large

    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=class_count + 1,
    )
    # Torchvision intentionally hard-codes this factory to 320. The detector's
    # transform and normalized default boxes remain shape-safe at 640, so the
    # shared benchmark adaptation is explicit rather than passed as an ignored kwarg.
    model.transform.fixed_size = (input_size, input_size)
    return model


def load_matching_pretrained(
    model: torch.nn.Module, weight_path: str | Path
) -> dict[str, Any]:
    """Load all compatible COCO detector tensors, excluding the resized class head."""

    state = torch.load(Path(weight_path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ProtocolError("Unsupported Torchvision SSDLite weight structure")
    current = model.state_dict()
    matched = {
        key: value
        for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    mismatched = sorted(
        key
        for key, value in state.items()
        if key in current and current[key].shape != value.shape
    )
    unexpected = sorted(set(state) - set(current))
    missing = sorted(set(current) - set(matched))
    model.load_state_dict(matched, strict=False)
    if not matched or not mismatched:
        raise ProtocolError(
            "SSDLite pretrained transfer did not detect the expected class-head mismatch"
        )
    return {
        "matched": len(matched),
        "mismatched": mismatched,
        "unexpected": unexpected,
        "missing": missing,
    }


def checkpoint_state(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ProtocolError("Unsupported SSDLite training checkpoint")
    return payload
