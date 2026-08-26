"""CUDA-container entrypoint for added-model training and inference."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path("/workspace")
IMAGE_ROOT = REPO_ROOT / "data" / "processed" / "NIR" / "images"


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a YAML mapping: {path}")
    return value


def latest_checkpoint(training_dir: Path, model: str) -> Path | None:
    if model == "rtmdet_tiny":
        pointer = training_dir / "last_checkpoint"
        if pointer.is_file():
            candidate = Path(pointer.read_text(encoding="utf-8").strip())
            if not candidate.is_absolute():
                candidate = training_dir / candidate
            if candidate.is_file():
                return candidate
        candidates = sorted(training_dir.glob("epoch_*.pth"))
    else:
        candidates = [
            training_dir / "last.pth.tar",
            training_dir / "checkpoint-latest.pth.tar",
        ]
        candidates.extend(sorted(training_dir.glob("checkpoint-*.pth.tar"), reverse=True))
    return next((item for item in candidates if item.is_file()), None)


def prepare_effdet_dataset(ratio_config: Path) -> Path:
    ratio = load_yaml(ratio_config)
    root = Path("/tmp") / f"dms-eval-efficientdet-{ratio['ratio']}"
    annotations = root / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    source_train = REPO_ROOT / ratio["train_annotations"]
    train = json.loads(source_train.read_text(encoding="utf-8"))
    category_ids = sorted(int(item["id"]) for item in train["categories"])
    if category_ids == [0, 1]:
        for item in train["categories"]:
            item["id"] = int(item["id"]) + 1
        for item in train["annotations"]:
            item["category_id"] = int(item["category_id"]) + 1
    if sorted(int(item["id"]) for item in train["categories"]) != [1, 2]:
        raise RuntimeError("EfficientDet training categories must resolve to one-based IDs 1 and 2")
    (annotations / "instances_train2017.json").write_text(
        json.dumps(train), encoding="utf-8"
    )
    shutil.copy2(
        REPO_ROOT / ratio["validation_annotations"],
        annotations / "instances_val2017.json",
    )
    for name in ("train2017", "val2017"):
        link = root / name
        if link.is_symlink() or link.exists():
            if link.resolve() == IMAGE_ROOT.resolve():
                continue
            raise RuntimeError(f"Unexpected EfficientDet dataset path: {link}")
        link.symlink_to(IMAGE_ROOT, target_is_directory=True)
    return root


def train(args: argparse.Namespace) -> int:
    run_root = REPO_ROOT / "runs" / "NIR" / args.model / f"ratio_{args.ratio}"
    training_dir = run_root / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    resume = latest_checkpoint(training_dir, args.model)
    if any(training_dir.iterdir()) and resume is None:
        raise RuntimeError(
            f"Refusing unsafe resume: {training_dir} contains files but no resumable checkpoint"
        )
    if args.model == "rtmdet_tiny":
        config = REPO_ROOT / "configs" / "NIR" / "rtmdet" / f"ratio_{args.ratio}.py"
        command = [
            sys.executable,
            "/opt/mmyolo/tools/train.py",
            str(config),
            "--work-dir",
            str(training_dir),
        ]
        if resume is not None:
            command.extend(["--resume", str(resume)])
    elif args.model == "efficientdet_d1":
        ratio_config = (
            REPO_ROOT / "configs" / "NIR" / "efficientdet" / f"ratio_{args.ratio}.yaml"
        )
        dataset_root = prepare_effdet_dataset(ratio_config)
        base_config = REPO_ROOT / "configs" / "NIR" / "efficientdet" / "base.yaml"
        cache = REPO_ROOT / "third_party" / "torch-cache" / "hub" / "checkpoints"
        cache.mkdir(parents=True, exist_ok=True)
        cached_weight = cache / "tf_efficientdet_d1_40-a30f94af.pth"
        source_weight = REPO_ROOT / "third_party" / "weights" / cached_weight.name
        if not cached_weight.exists():
            cached_weight.symlink_to(source_weight)
        command = [
            sys.executable,
            "/opt/efficientdet/train.py",
            str(dataset_root),
            "--config",
            str(base_config),
            "--output",
            str(run_root),
        ]
        if resume is not None:
            command.extend(["--resume", str(resume)])
        else:
            command.append("--pretrained")
        os.environ["TORCH_HOME"] = str(REPO_ROOT / "third_party" / "torch-cache")
    else:
        raise RuntimeError(f"Unsupported container model: {args.model}")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


def _latency_summary(values: list[float], boundary: str) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    total = float(array.sum())
    return {
        "timed_frames": int(array.size),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "total_ms": total,
        "sustained_fps": float(array.size / (total / 1000.0)),
        "timing": "synchronized_cuda_events" if boundary.endswith("output") else "synchronized_high_resolution_wall_clock",
        "boundary": boundary,
    }


def infer(args: argparse.Namespace) -> int:
    from tools.backends.runtime import create_runtime

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required inside the additional-model image")
    ground_truth = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    runtime = create_runtime(args.model, Path(args.config), Path(args.checkpoint))
    predictions: list[dict[str, Any]] = []
    forward_ms: list[float] = []
    end_to_end_ms: list[float] = []
    first_image = ground_truth["images"][0]
    sample = runtime.preprocess(IMAGE_ROOT / first_image["file_name"])
    if args.profile:
        for _ in range(10):
            runtime.normalize(runtime.raw_forward(sample), [0])
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    for image in ground_truth["images"]:
        batch = runtime.preprocess(IMAGE_ROOT / image["file_name"])
        if args.profile:
            import time

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall = time.perf_counter()
            start.record()
            raw = runtime.raw_forward(batch)
            end.record()
            values = runtime.normalize(raw, [int(image["id"])])
            torch.cuda.synchronize()
            forward_ms.append(float(start.elapsed_time(end)))
            end_to_end_ms.append((time.perf_counter() - wall) * 1000.0)
        else:
            values = runtime.normalize(runtime.raw_forward(batch), [int(image["id"])])
        predictions.extend(values)
    profile = None
    if args.profile:
        profile = {
            "batch_size": 1,
            "precision": "fp16_autocast",
            "precision_mode": "cuda_amp_fp16",
            "model_and_input_storage": "fp32",
            "input_shape": [1, 3, 640, 640],
            "warmup_passes": 10,
            "model_forward": _latency_summary(
                forward_ms, "preprocessed_tensor_to_raw_model_output"
            ),
            "tensor_to_final_detections": _latency_summary(
                end_to_end_ms,
                "preprocessed_tensor_to_normalized_detections_including_required_postprocessing",
            ),
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "environment": runtime.environment_metadata(),
        }
    payload = {
        "predictions": predictions,
        "runtime_profile": profile,
        "parameters": runtime.parameter_count(),
        "flop_estimates": runtime.flop_estimates(sample),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(output)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--model", required=True, choices=["rtmdet_tiny", "efficientdet_d1"])
    train_parser.add_argument("--ratio", required=True, choices=["1to2", "1to6"])
    train_parser.set_defaults(function=train)
    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--model", required=True, choices=["rtmdet_tiny", "efficientdet_d1"])
    infer_parser.add_argument("--config", required=True)
    infer_parser.add_argument("--checkpoint", required=True)
    infer_parser.add_argument("--ground-truth", required=True)
    infer_parser.add_argument("--output", required=True)
    infer_parser.add_argument("--profile", action="store_true")
    infer_parser.set_defaults(function=infer)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
