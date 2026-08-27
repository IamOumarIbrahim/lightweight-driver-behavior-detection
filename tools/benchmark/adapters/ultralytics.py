"""Ultralytics adapter for the frozen YOLO-family models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

from ..protocol import ProtocolError
from .base import DetectorAdapter


class UltralyticsAdapter(DetectorAdapter):
    precision_mode = "cuda_amp_fp16"

    def load(self) -> UltralyticsAdapter:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ProtocolError("ultralytics is not installed") from exc
        self.wrapper = YOLO(str(self.checkpoint))
        self.model = self.wrapper.model.to(self.device).eval()
        names = self.wrapper.names
        if len(names) != self.class_count and not self.allow_pretrained_head_mismatch:
            raise ProtocolError(
                f"{self.model_id} checkpoint has {len(names)} classes, expected {self.class_count}"
            )
        self.is_end2end = (
            getattr(self.model, "end2end", False)
            or getattr(getattr(self.model, "model", [None])[-1], "end2end", False)
            or self.model_id.startswith("yolo26")
        )
        return self

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize(self.input_size, Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(array).unsqueeze(0).to(self.device)

    def raw_forward(self, batch: torch.Tensor) -> Any:
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self.model(batch)
        return self.model(batch)

    @torch.inference_mode()
    def normalize(self, raw_outputs: Any, image_ids: list[int]) -> list[dict[str, Any]]:
        if getattr(self, "is_end2end", False):
            preds = (
                raw_outputs[0]
                if isinstance(raw_outputs, (tuple, list))
                else raw_outputs
            )
            if len(preds) != len(image_ids):
                raise ProtocolError("Batch size and image ID count differ")
            predictions: list[dict[str, Any]] = []
            for image_id, image_detections in zip(image_ids, preds):
                mask = image_detections[:, 4] > 0.001
                for detection in image_detections[mask].cpu():
                    x1, y1, x2, y2, score, class_id = map(float, detection[:6].tolist())
                    category_id = int(class_id) + 1
                    if category_id not in set(range(1, self.class_count + 1)):
                        if self.allow_pretrained_head_mismatch:
                            continue
                        raise ProtocolError(
                            f"Adapter emitted class {category_id} outside frozen ontology"
                        )
                    predictions.append(
                        {
                            "image_id": int(image_id),
                            "category_id": category_id,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": score,
                        }
                    )
            return predictions

        from ultralytics.utils.nms import non_max_suppression

        detections = non_max_suppression(
            raw_outputs,
            conf_thres=0.001,
            iou_thres=0.7,
            nc=len(self.wrapper.names),
            max_det=300,
        )
        if len(detections) != len(image_ids):
            raise ProtocolError("Batch size and image ID count differ")
        predictions: list[dict[str, Any]] = []
        for image_id, image_detections in zip(image_ids, detections):
            for detection in image_detections.cpu():
                x1, y1, x2, y2, score, class_id = map(float, detection[:6].tolist())
                category_id = int(class_id) + 1
                if category_id not in set(range(1, self.class_count + 1)):
                    if self.allow_pretrained_head_mismatch:
                        continue
                    raise ProtocolError(
                        f"Adapter emitted class {category_id} outside frozen ontology"
                    )
                predictions.append(
                    {
                        "image_id": int(image_id),
                        "category_id": category_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )
        return predictions

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())
