"""Pinned official YOLOX-Nano benchmark adapter at the shared 640 input."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..paths import CONFIGS_ROOT, THIRD_PARTY_ROOT
from ..protocol import ProtocolError
from .base import DetectorAdapter


class YOLOXAdapter(DetectorAdapter):
    precision_mode = "cuda_amp_fp16"

    def load(self) -> YOLOXAdapter:
        checkout = THIRD_PARTY_ROOT / "YOLOX"
        if not checkout.is_dir():
            raise ProtocolError(
                "Pinned YOLOX checkout is missing; run scripts/setup/02_setup_backends.bat"
            )
        if str(checkout) not in sys.path:
            sys.path.insert(0, str(checkout))
        try:
            from yolox.exp import get_exp
        except ImportError as exc:
            raise ProtocolError("YOLOX dependencies are incomplete") from exc
        config_path = self.config_path or CONFIGS_ROOT / "NIR" / "yolox" / "base.py"
        self.exp = get_exp(str(config_path), None)
        self.exp.num_classes = self.class_count
        model = self.exp.get_model()
        checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            raise ProtocolError("Unsupported YOLOX checkpoint structure")
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()
        return self

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        from yolox.data import ValTransform

        bgr = np.asarray(image.convert("RGB"), dtype=np.uint8)[:, :, ::-1]
        array, _ = ValTransform(legacy=False)(bgr, None, self.input_size)
        return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).to(self.device)

    def raw_forward(self, batch: torch.Tensor) -> Any:
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return self.model(batch)
        return self.model(batch.float())

    @torch.inference_mode()
    def normalize(self, raw_outputs: Any, image_ids: list[int]) -> list[dict[str, Any]]:
        from yolox.utils import postprocess

        detections = postprocess(
            raw_outputs,
            self.class_count,
            conf_thre=0.001,
            nms_thre=0.65,
            class_agnostic=False,
        )
        if len(detections) != len(image_ids):
            raise ProtocolError("Batch size and image ID count differ")
        predictions: list[dict[str, Any]] = []
        for image_id, image_detections in zip(image_ids, detections):
            if image_detections is None:
                continue
            for detection in image_detections.cpu():
                x1, y1, x2, y2 = map(float, detection[:4].tolist())
                category_id = int(detection[6].item()) + 1
                if category_id not in set(range(1, self.class_count + 1)):
                    raise ProtocolError(
                        f"Adapter emitted class {category_id} outside frozen ontology"
                    )
                if x2 <= x1 or y2 <= y1:
                    continue
                predictions.append(
                    {
                        "image_id": int(image_id),
                        "category_id": category_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float((detection[4] * detection[5]).item()),
                    }
                )
        return predictions

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())
