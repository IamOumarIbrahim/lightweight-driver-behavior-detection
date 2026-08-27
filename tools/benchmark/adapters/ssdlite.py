"""Torchvision SSDLite-MobileNetV3-Large benchmark adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from tools.backends.ssdlite import build_ssdlite, checkpoint_state

from ..protocol import ProtocolError
from .base import DetectorAdapter


class SSDLiteAdapter(DetectorAdapter):
    precision_mode = "cuda_amp_fp16"

    def load(self) -> SSDLiteAdapter:
        self.model = build_ssdlite(self.class_count, input_size=640)
        state = checkpoint_state(self.checkpoint)
        self.model.load_state_dict(state["model"], strict=True)
        self.model = self.model.to(self.device).eval()
        return self

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize(self.input_size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array).unsqueeze(0).to(self.device)

    def raw_forward(self, batch: torch.Tensor) -> Any:
        images = [image for image in batch]
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self.model(images)
        return self.model(images)

    @torch.inference_mode()
    def normalize(self, raw_outputs: Any, image_ids: list[int]) -> list[dict[str, Any]]:
        if len(raw_outputs) != len(image_ids):
            raise ProtocolError("Batch size and image ID count differ")
        predictions: list[dict[str, Any]] = []
        for image_id, output in zip(image_ids, raw_outputs):
            for box, label, score in zip(
                output["boxes"].cpu(),
                output["labels"].cpu(),
                output["scores"].cpu(),
            ):
                category_id = int(label.item())
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
