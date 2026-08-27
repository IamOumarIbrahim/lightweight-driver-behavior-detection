"""Pinned official RT-DETRv2-S (R18VD) benchmark adapter."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..paths import CONFIGS_ROOT, THIRD_PARTY_ROOT
from ..protocol import ProtocolError
from .base import DetectorAdapter


class RTDETRv2Adapter(DetectorAdapter):
    precision_mode = "cuda_amp_fp16"

    def load(self) -> RTDETRv2Adapter:
        checkout = THIRD_PARTY_ROOT / "RT-DETR" / "rtdetrv2_pytorch"
        if not checkout.is_dir():
            raise ProtocolError(
                "Pinned RT-DETR checkout is missing; run scripts/setup/02_setup_backends.bat"
            )
        if str(checkout) not in sys.path:
            sys.path.insert(0, str(checkout))
        try:
            from src.core import YAMLConfig
        except ImportError as exc:
            raise ProtocolError("RT-DETRv2 dependencies are incomplete") from exc

        config_path = (
            self.config_path or CONFIGS_ROOT / "NIR" / "rtdetrv2" / "base.yml"
        )
        cfg = YAMLConfig(str(config_path))
        cfg.yaml_cfg["PResNet"]["pretrained"] = False
        model = cfg.model
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        state = None
        if isinstance(checkpoint, dict):
            ema = checkpoint.get("ema")
            if isinstance(ema, dict):
                state = ema.get("module")
            if not isinstance(state, dict):
                state = checkpoint.get("model")
        if not isinstance(state, dict):
            raise ProtocolError("Unsupported RT-DETRv2 checkpoint structure")
        model.load_state_dict(state, strict=True)
        self.model = model.deploy().to(self.device).eval()
        self.postprocessor = cfg.postprocessor.deploy().to(self.device).eval()
        return self

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize(self.input_size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array).unsqueeze(0).to(self.device)

    def raw_forward(self, batch: torch.Tensor) -> Any:
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self.model(batch)
        return self.model(batch.float())

    @torch.inference_mode()
    def normalize(self, raw_outputs: Any, image_ids: list[int]) -> list[dict[str, Any]]:
        sizes = torch.tensor([[640, 640]] * len(image_ids), device=self.device)
        labels, boxes, scores = self.postprocessor(raw_outputs, sizes)
        if len(labels) != len(image_ids):
            raise ProtocolError("Batch size and image ID count differ")
        predictions: list[dict[str, Any]] = []
        for image_id, image_labels, image_boxes, image_scores in zip(
            image_ids, labels, boxes, scores
        ):
            for label, box, score in zip(image_labels, image_boxes, image_scores):
                category_id = int(label.item()) + 1
                if category_id not in set(range(1, self.class_count + 1)):
                    raise ProtocolError(
                        f"Adapter emitted class {category_id} outside frozen ontology"
                    )
                x1, y1, x2, y2 = map(float, box.tolist())
                if x2 <= x1 or y2 <= y1:
                    continue
                predictions.append(
                    {
                        "image_id": int(image_id),
                        "category_id": category_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score.item()),
                    }
                )
        return predictions

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())
