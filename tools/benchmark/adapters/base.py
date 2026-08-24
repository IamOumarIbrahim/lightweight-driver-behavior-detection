"""Common detector adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from PIL import Image


class DetectorAdapter(ABC):
    class_count = 4
    input_size = (640, 640)

    def __init__(
        self,
        model_id: str,
        checkpoint: str | Path,
        device: str = "cuda:0",
        allow_pretrained_head_mismatch: bool = False,
        class_count: int = 4,
        config_path: str | Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.checkpoint = Path(checkpoint).resolve()
        self.device = torch.device(device)
        self.allow_pretrained_head_mismatch = allow_pretrained_head_mismatch
        self.class_count = class_count
        self.config_path = Path(config_path).resolve() if config_path else None
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)

    @abstractmethod
    def load(self) -> DetectorAdapter:
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Decode-independent preprocessing, returning BCHW float data on the adapter device."""
        raise NotImplementedError

    @abstractmethod
    def raw_forward(self, batch: torch.Tensor) -> Any:
        """Model-only forward boundary used by every runtime profile."""
        raise NotImplementedError

    @abstractmethod
    def normalize(self, raw_outputs: Any, image_ids: list[int]) -> list[dict[str, Any]]:
        """Normalize raw outputs to COCO xywh predictions with one-based category IDs."""
        raise NotImplementedError

    def infer(self, batch: torch.Tensor, image_ids: list[int]) -> list[dict[str, Any]]:
        return self.normalize(self.raw_forward(batch), image_ids)

    @abstractmethod
    def parameter_count(self) -> int:
        raise NotImplementedError

    def export_inference_artifact(self, destination: str | Path) -> Path:
        """Export a comparable inference-only FP16 state dictionary.

        The artifact intentionally excludes optimizer, scheduler, scaler, EMA wrapper,
        and training history so serialized size means the same thing for every backend.
        """
        if not hasattr(self, "model"):
            raise RuntimeError(
                "Adapter must be loaded before exporting an inference artifact"
            )
        destination = Path(destination).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {
            key: (
                value.detach().cpu().half()
                if value.is_floating_point()
                else value.detach().cpu()
            )
            for key, value in self.model.state_dict().items()
        }
        payload = {
            "schema_version": 1,
            "artifact": "standardized_inference_state_dict",
            "model_id": self.model_id,
            "tensor_precision": "fp16",
            "state_dict": state_dict,
        }
        temporary = destination.with_name(f".{destination.name}.tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)
        return destination

    def synthetic_input(self) -> torch.Tensor:
        return torch.zeros((1, 3, 640, 640), dtype=torch.float32, device=self.device)
