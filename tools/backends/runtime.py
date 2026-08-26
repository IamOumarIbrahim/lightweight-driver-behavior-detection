"""In-container adapters for the isolated additional-model image."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


class RuntimeBase:
    official_flops: int

    def environment_metadata(self) -> dict[str, Any]:
        properties = torch.cuda.get_device_properties(0)
        return {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": True,
            "gpu": properties.name,
            "gpu_total_bytes": properties.total_memory,
            "compute_capability": f"{properties.major}.{properties.minor}",
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @torch.inference_mode()
    def flop_estimates(self, sample: torch.Tensor) -> dict[str, Any]:
        try:
            from thop import profile

            macs, _ = profile(self.flop_module(), inputs=(sample,), verbose=False)
            value = int(2 * macs)
            status = "estimated"
            method = "THOP MACs multiplied by 2 inside pinned backend image"
        except Exception as exc:  # operator coverage varies across backend stacks.
            value = self.official_flops
            status = "reported_fallback"
            method = f"Official model-zoo value; THOP unavailable ({type(exc).__name__})"
        return {
            "thop": {"flops": value, "method": method, "status": status},
            "official_reported": {
                "flops": self.official_flops,
                "method": "Official model-zoo 640x640 value",
                "status": "reported",
            },
            "interpretation": "Tool-dependent estimate; official value is retained as a cross-check.",
        }


class RTMDetRuntime(RuntimeBase):
    official_flops = 8_100_000_000

    def __init__(self, config: Path, checkpoint: Path):
        from mmdet.apis import init_detector
        from mmyolo.utils import register_all_modules

        register_all_modules()
        self.model = init_detector(str(config), str(checkpoint), device="cuda:0")
        self.model.eval()

    def preprocess(self, path: Path) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        from mmdet.structures import DetDataSample

        with Image.open(path) as source:
            rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        bgr = np.ascontiguousarray(rgb[..., ::-1].transpose(2, 0, 1))
        sample = DetDataSample()
        sample.set_metainfo(
            {
                "ori_shape": (640, 640),
                "img_shape": (640, 640),
                "scale_factor": (1.0, 1.0),
                "pad_param": (0.0, 0.0, 0.0, 0.0),
            }
        )
        processed = self.model.data_preprocessor(
            {"inputs": [torch.from_numpy(bgr)], "data_samples": [sample]},
            training=False,
        )
        return processed["inputs"], [item.metainfo for item in processed["data_samples"]]

    @torch.inference_mode()
    def raw_forward(self, batch):
        inputs, metadata = batch
        with torch.cuda.amp.autocast(dtype=torch.float16):
            features = self.model.extract_feat(inputs)
            outputs = self.model.bbox_head(features)
        return outputs, metadata

    @torch.inference_mode()
    def normalize(self, raw_outputs, image_ids: list[int]) -> list[dict[str, Any]]:
        outputs, metadata = raw_outputs
        instances = self.model.bbox_head.predict_by_feat(
            *outputs, batch_img_metas=metadata, rescale=True, with_nms=True
        )
        predictions = []
        for image_id, result in zip(image_ids, instances):
            for box, score, label in zip(result.bboxes.cpu(), result.scores.cpu(), result.labels.cpu()):
                category_id = int(label.item()) + 1
                if category_id not in (1, 2):
                    continue
                x1, y1, x2, y2 = map(float, box.tolist())
                predictions.append(
                    {
                        "image_id": int(image_id),
                        "category_id": category_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score.item()),
                    }
                )
        return predictions

    def flop_module(self):
        runtime = self

        class Forward(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.detector = runtime.model

            def forward(self, batch):
                inputs, _ = batch
                return self.detector.bbox_head(self.detector.extract_feat(inputs))

        return Forward()


class EfficientDetRuntime(RuntimeBase):
    official_flops = 6_100_000_000

    def __init__(self, _config: Path, checkpoint: Path):
        from effdet import create_model
        from effdet.anchors import Anchors

        self.model = create_model(
            "tf_efficientdet_d1",
            bench_task="",
            num_classes=2,
            checkpoint_path=str(checkpoint),
            checkpoint_ema=True,
        ).cuda().eval()
        self.config = self.model.config
        self.anchors = Anchors.from_config(self.config)

    def preprocess(self, path: Path) -> torch.Tensor:
        with Image.open(path) as source:
            image = source.convert("RGB").resize((640, 640), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
        return torch.from_numpy((array - mean) / std).unsqueeze(0).cuda()

    @torch.inference_mode()
    def raw_forward(self, batch: torch.Tensor):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            return self.model(batch)

    @torch.inference_mode()
    def normalize(self, raw_outputs, image_ids: list[int]) -> list[dict[str, Any]]:
        from effdet.bench import _batch_detection, _post_process

        class_out, box_out = raw_outputs
        class_out, box_out, indices, classes = _post_process(
            class_out,
            box_out,
            num_levels=self.config.num_levels,
            num_classes=2,
            max_detection_points=self.config.max_detection_points,
        )
        detections = _batch_detection(
            len(image_ids),
            class_out,
            box_out,
            self.anchors.boxes,
            indices,
            classes,
            None,
            None,
            max_det_per_image=self.config.max_det_per_image,
            soft_nms=self.config.soft_nms,
        )
        predictions = []
        for image_id, image_detections in zip(image_ids, detections.cpu()):
            for detection in image_detections:
                x1, y1, x2, y2, score, label = map(float, detection[:6].tolist())
                if score < 0.001:
                    continue
                category_id = int(label)
                if category_id not in (1, 2):
                    continue
                predictions.append(
                    {
                        "image_id": int(image_id),
                        "category_id": category_id,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": score,
                    }
                )
        return predictions

    def flop_module(self):
        return self.model


def create_runtime(model: str, config: Path, checkpoint: Path) -> RuntimeBase:
    if model == "rtmdet_tiny":
        return RTMDetRuntime(config, checkpoint)
    if model == "efficientdet_d1":
        return EfficientDetRuntime(config, checkpoint)
    raise ValueError(f"Unsupported container model: {model}")
