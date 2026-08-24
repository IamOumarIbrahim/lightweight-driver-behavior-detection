"""Shared CUDA AMP profiling and independently estimated model complexity."""

from __future__ import annotations

import platform
import sys
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from .adapters.base import DetectorAdapter
from .protocol import ProtocolError


def environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        metadata["gpu"] = properties.name
        metadata["gpu_total_bytes"] = properties.total_memory
        metadata["compute_capability"] = f"{properties.major}.{properties.minor}"
    return metadata


def latency_summary(
    latencies_ms: Iterable[float], *, timing: str, boundary: str
) -> dict[str, Any]:
    latencies = np.asarray(list(latencies_ms), dtype=np.float64)
    if (
        latencies.size == 0
        or not np.all(np.isfinite(latencies))
        or np.any(latencies <= 0)
    ):
        raise ProtocolError("Latency samples must be finite positive values")
    total_ms = float(latencies.sum())
    return {
        "timed_frames": int(latencies.size),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "total_ms": total_ms,
        "sustained_fps": float(latencies.size / (total_ms / 1000.0)),
        "timing": timing,
        "boundary": boundary,
    }


class CudaForwardProfiler:
    """Measure model-only and tensor-to-final-detections latency in one pass."""

    def __init__(self, adapter: DetectorAdapter, warmups: int = 10) -> None:
        if adapter.device.type != "cuda" or not torch.cuda.is_available():
            raise ProtocolError("The frozen profiler requires a CUDA device")
        self.adapter = adapter
        self.warmups = warmups
        self.events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.end_to_end_ms: list[float] = []
        self._pending_wall_start: float | None = None

    @torch.inference_mode()
    def prepare(self, sample: torch.Tensor) -> None:
        if (
            sample.shape != (1, 3, 640, 640)
            or sample.dtype != torch.float32
            or sample.device.type != "cuda"
        ):
            raise ProtocolError(
                "Profiler input must be CUDA FP32 with shape 1x3x640x640"
            )
        for _ in range(self.warmups):
            self.adapter.infer(sample, [0])
        torch.cuda.synchronize(self.adapter.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.adapter.device)

    @torch.inference_mode()
    def forward(self, sample: torch.Tensor) -> Any:
        if self._pending_wall_start is not None:
            raise ProtocolError("Call profiler.finalize before recording another frame")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        self._pending_wall_start = time.perf_counter()
        start.record()
        outputs = self.adapter.raw_forward(sample)
        end.record()
        self.events.append((start, end))
        return outputs

    @torch.inference_mode()
    def finalize(self, raw_outputs: Any, image_ids: list[int]) -> list[dict[str, Any]]:
        if self._pending_wall_start is None:
            raise ProtocolError("No pending profiled forward pass")
        predictions = self.adapter.normalize(raw_outputs, image_ids)
        torch.cuda.synchronize(self.adapter.device)
        elapsed_ms = (time.perf_counter() - self._pending_wall_start) * 1000.0
        self.end_to_end_ms.append(elapsed_ms)
        self._pending_wall_start = None
        return predictions

    def finish(self) -> dict[str, Any]:
        if not self.events:
            raise ProtocolError("No timed forward passes were recorded")
        if self._pending_wall_start is not None or len(self.events) != len(
            self.end_to_end_ms
        ):
            raise ProtocolError("Every timed forward pass must be finalized")
        torch.cuda.synchronize(self.adapter.device)
        forward_ms = [start.elapsed_time(end) for start, end in self.events]
        return {
            "batch_size": 1,
            "precision": "fp16_autocast",
            "precision_mode": getattr(self.adapter, "precision_mode", "cuda_amp_fp16"),
            "model_and_input_storage": "fp32",
            "input_shape": [1, 3, 640, 640],
            "warmup_passes": self.warmups,
            "model_forward": latency_summary(
                forward_ms,
                timing="synchronized_cuda_events",
                boundary="preprocessed_tensor_to_raw_model_output",
            ),
            "tensor_to_final_detections": latency_summary(
                self.end_to_end_ms,
                timing="synchronized_high_resolution_wall_clock",
                boundary="preprocessed_tensor_to_normalized_detections_including_required_postprocessing",
            ),
            "peak_allocated_vram_bytes": int(
                torch.cuda.max_memory_allocated(self.adapter.device)
            ),
            "environment": environment_metadata(),
        }


class _ProfileBoundary(torch.nn.Module):
    def __init__(self, detector: DetectorAdapter) -> None:
        super().__init__()
        self.model = detector.model
        object.__setattr__(self, "_detector", detector)

    def forward(self, value):
        return self._detector.raw_forward(value)


@torch.inference_mode()
def model_flop_estimates(
    adapter: DetectorAdapter, sample: torch.Tensor | None = None
) -> dict[str, Any]:
    """Return THOP and torch-profiler estimates; neither is treated as ground truth."""
    from thop import profile

    sample = sample if sample is not None else adapter.synthetic_input()
    boundary = _ProfileBoundary(adapter)
    macs, _ = profile(boundary, inputs=(sample,), verbose=False)
    estimates: dict[str, Any] = {
        "thop": {
            "flops": int(2 * macs),
            "method": "ultralytics-thop MACs multiplied by 2",
            "status": "estimated",
        }
    }
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if sample.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            boundary(sample)
            if sample.device.type == "cuda":
                torch.cuda.synchronize(adapter.device)
        profiler_flops = int(sum(event.flops or 0 for event in prof.key_averages()))
        estimates["torch_profiler"] = {
            "flops": profiler_flops,
            "method": "torch.profiler with_flops operator sum",
            "status": "estimated" if profiler_flops > 0 else "zero_operator_coverage",
        }
    except Exception as exc:  # noqa: BLE001 - profiler operator support varies by build.
        estimates["torch_profiler"] = {
            "flops": None,
            "method": "torch.profiler with_flops operator sum",
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }
    estimates["interpretation"] = (
        "Tool-dependent estimates; disagreement indicates operator-coverage uncertainty."
    )
    return estimates


@torch.inference_mode()
def model_flops(adapter: DetectorAdapter, sample: torch.Tensor | None = None) -> int:
    """Backward-compatible THOP estimate; new artifacts should use model_flop_estimates."""
    return int(model_flop_estimates(adapter, sample)["thop"]["flops"])


@torch.inference_mode()
def synthetic_profile(
    adapter: DetectorAdapter, repeats: int = 100, warmups: int = 10
) -> dict[str, Any]:
    sample = adapter.synthetic_input()
    profiler = CudaForwardProfiler(adapter, warmups=warmups)
    profiler.prepare(sample)
    for index in range(repeats):
        raw = profiler.forward(sample)
        profiler.finalize(raw, [index])
    report = profiler.finish()
    report.update(
        {
            "model_id": adapter.model_id,
            "input_source": "synthetic",
            "parameters": adapter.parameter_count(),
            "flop_estimates": model_flop_estimates(adapter, sample),
        }
    )
    return report
