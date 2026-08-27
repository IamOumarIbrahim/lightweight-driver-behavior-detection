"""Run validation first, then a confirmation-gated protected test pass."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from tools.benchmark.adapters import create_adapter
from tools.benchmark.evaluation import (
    calibrate_threshold,
    coco_metrics,
    operating_point_metrics,
    snippet_operating_point_metrics,
)
from tools.benchmark.paths import (
    CONFIGS_ROOT,
    MODELS,
    NIR_MODELS,
    NIR_RATIOS,
    NIR_SEED,
    REPO_ROOT,
    RGB_MODELS,
    RGB_SEEDS,
    is_authoritative_rgb_yolo,
    result_dir,
    run_dir,
)
from tools.benchmark.profiling import CudaForwardProfiler, model_flop_estimates
from tools.benchmark.protocol import (
    ProtocolError,
    sha256_file,
    validate_protocol,
    verify_authoritative_fingerprints,
)


ULTRALYTICS_MODELS = {"yolo11n", "yolo26n", "yolov10n", "yolov8n"}


def require_frozen_model(track: str, model: str) -> None:
    expected = RGB_MODELS if track == "RGB" else NIR_MODELS
    if model not in expected:
        raise ProtocolError(f"Model {model} is not frozen for the {track} track")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def identity(track: str, seed: int | None, ratio: str | None) -> dict[str, Any]:
    if track == "RGB" and seed in RGB_SEEDS and ratio is None:
        return {"seed": seed, "training_seed": seed, "ratio": None}
    if track == "NIR" and ratio in NIR_RATIOS and seed in (None, NIR_SEED):
        return {"ratio": ratio, "training_seed": NIR_SEED, "seed": None}
    raise ProtocolError("RGB uses --seed 13/37/73; NIR uses --ratio 1to2/1to6")


def checkpoint_for(model: str, training_dir: Path) -> Path:
    if model in ULTRALYTICS_MODELS:
        candidates = [training_dir / "weights" / "best.pt"]
    elif model == "ssdlite_mobilenet_v3_large":
        candidates = [training_dir / "best.pt", training_dir / "last.pt"]
    elif model == "rtdetrv2_s":
        candidates = [training_dir / "best.pth", training_dir / "last.pth"]
    elif model == "yolox_nano":
        candidates = [
            training_dir / "best_ckpt.pth",
            training_dir / "latest_ckpt.pth",
        ]
    else:
        candidates = [
            training_dir / "best_stg2.pth",
            training_dir / "best_stg1.pth",
            training_dir / "last.pth",
        ]
    checkpoint = next((path for path in candidates if path.is_file()), None)
    if checkpoint is None:
        raise FileNotFoundError(f"No trained checkpoint found under {training_dir}")
    return checkpoint


def paths_for(track: str, model: str, run_identity: dict[str, Any]) -> dict[str, Path]:
    key = (
        {"seed": run_identity["seed"]}
        if track == "RGB"
        else {"ratio": run_identity["ratio"]}
    )
    run = run_dir(track, model, **key)
    result = result_dir(track, model, **key)
    return {
        "run": run,
        "result": result,
        "training": run / "training",
        "validation": run / "validation",
        "test": run / "test",
        "ledger": run / "test" / ".protected_test_started",
    }


def ground_truth_path(track: str, split: str) -> Path:
    return (
        REPO_ROOT
        / "data"
        / "processed"
        / track
        / "coco"
        / "evaluation"
        / f"instances_{split}.json"
    )


def load_ground_truth(track: str, split: str) -> dict[str, Any]:
    path = ground_truth_path(track, split)
    if not path.is_file():
        raise FileNotFoundError(
            f"Prepared {track} {split} annotations are missing: {path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def adapter_for(track: str, model: str, checkpoint: Path, ratio: str | None = None):
    config = None
    if model == "dfine_n":
        config_name = "base.yml" if track == "RGB" else f"ratio_{ratio}.yml"
        config = CONFIGS_ROOT / track / "dfine" / config_name
    elif model == "rtdetrv2_s":
        config = CONFIGS_ROOT / "NIR" / "rtdetrv2" / f"ratio_{ratio}.yml"
    elif model == "yolox_nano":
        config = CONFIGS_ROOT / "NIR" / "yolox" / f"ratio_{ratio}.py"
    class_count = 4 if track == "RGB" else 2
    return create_adapter(
        model, checkpoint, device="cuda:0", class_count=class_count, config_path=config
    ).load()


@torch.inference_mode()
def predict(
    adapter, track: str, ground_truth: dict[str, Any], *, profile: bool
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    profiler = CudaForwardProfiler(adapter, warmups=10) if profile else None
    if profiler:
        profiler.prepare(adapter.synthetic_input())
    predictions = []
    image_root = REPO_ROOT / "data" / "processed" / track / "images"
    for image in ground_truth["images"]:
        with Image.open(image_root / image["file_name"]) as source:
            batch = adapter.preprocess(source)
        outputs = profiler.forward(batch) if profiler else adapter.raw_forward(batch)
        predictions.extend(
            profiler.finalize(outputs, [int(image["id"])])
            if profiler
            else adapter.normalize(outputs, [int(image["id"])])
        )
    return predictions, profiler.finish() if profiler else None


def validation(args: argparse.Namespace) -> int:
    track = args.track.upper()
    require_frozen_model(track, args.model)
    protocol = validate_protocol(track)
    run_identity = identity(track, args.seed, args.ratio)
    paths = paths_for(track, args.model, run_identity)
    if is_authoritative_rgb_yolo(track, args.model, paths["result"]):
        print(
            f"Authoritative RGB YOLO result is frozen; validation will not be rerun: {paths['result']}"
        )
        return 0
    checkpoint = checkpoint_for(args.model, paths["training"])
    manifest_path = paths["validation"] / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("checkpoint_sha256") == sha256_file(checkpoint):
            print(f"Validation already complete: {manifest_path}")
            return 0
        raise ProtocolError(
            f"Refusing to overwrite validation for a different checkpoint: {manifest_path}"
        )
    if not args.execute_validation:
        print(
            f"Dry-run passed for {track}/{args.model}; add --execute-validation to run inference."
        )
        return 0
    if not torch.cuda.is_available():
        raise ProtocolError("CUDA is required for benchmark inference")
    ground_truth = load_ground_truth(track, "val")
    adapter = adapter_for(track, args.model, checkpoint, run_identity["ratio"])
    predictions, _ = predict(adapter, track, ground_truth, profile=False)
    metrics = coco_metrics(ground_truth, predictions)
    threshold = calibrate_threshold(ground_truth, predictions)
    predictions_path = paths["validation"] / "predictions.json"
    write_json(predictions_path, {"split": "val", "predictions": predictions})
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "track": track,
        "model": args.model,
        **run_identity,
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)).replace("\\", "/"),
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_fingerprints": verify_authoritative_fingerprints(protocol, track),
        "validation_ground_truth_sha256": sha256_file(ground_truth_path(track, "val")),
        "validation_predictions_sha256": sha256_file(predictions_path),
        "validation_metrics": metrics,
        "operating_point": threshold,
        "test_policy": "validation_manifest_frozen_before_single_confirmation_gated_test_pass",
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "metrics": metrics,
                "threshold": threshold,
            },
            indent=2,
        )
    )
    return 0


def test(args: argparse.Namespace) -> int:
    track = args.track.upper()
    require_frozen_model(track, args.model)
    validate_protocol(track)
    run_identity = identity(track, args.seed, args.ratio)
    paths = paths_for(track, args.model, run_identity)
    if is_authoritative_rgb_yolo(track, args.model, paths["result"]):
        print(
            f"Authoritative RGB YOLO result is frozen; protected test will not be rerun: {paths['result']}"
        )
        return 0
    result_path = paths["result"] / "metrics.json"
    if result_path.is_file():
        print(f"Protected test already complete: {result_path}")
        return 0
    manifest_path = paths["validation"] / "manifest.json"
    if not manifest_path.is_file():
        raise ProtocolError(
            "Validation must complete and freeze a manifest before test access"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = checkpoint_for(args.model, paths["training"])
    if manifest.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ProtocolError("Checkpoint changed after validation")
    if not args.execute_test or args.confirm != "RUN_PROTECTED_TEST":
        print(
            "Dry-run passed. Test was not accessed; use --execute-test --confirm RUN_PROTECTED_TEST."
        )
        return 0
    if paths["ledger"].exists():
        raise ProtocolError(
            f"Protected test has already started for this run: {paths['ledger']}"
        )
    if not torch.cuda.is_available():
        raise ProtocolError("CUDA is required for benchmark inference")
    paths["ledger"].parent.mkdir(parents=True, exist_ok=True)
    paths["ledger"].write_text(
        datetime.now(timezone.utc).isoformat() + "\n", encoding="utf-8", newline="\n"
    )
    ground_truth = load_ground_truth(track, "test")
    adapter = adapter_for(track, args.model, checkpoint, run_identity["ratio"])
    predictions, runtime = predict(adapter, track, ground_truth, profile=True)
    parameters = adapter.parameter_count()
    flop_estimates = model_flop_estimates(adapter)
    metrics = coco_metrics(ground_truth, predictions)
    operating = operating_point_metrics(
        ground_truth, predictions, manifest["operating_point"]["threshold"]
    )
    predictions_path = paths["test"] / "predictions.json"
    write_json(predictions_path, {"split": "test", "predictions": predictions})
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "track": track,
        "model": args.model,
        **run_identity,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "validation_manifest_sha256": sha256_file(manifest_path),
        "test_ground_truth_sha256": sha256_file(ground_truth_path(track, "test")),
        "test_predictions_sha256": sha256_file(predictions_path),
        "coco_metrics": metrics,
        "operating_point": operating,
        "runtime_profile": runtime,
        "parameters": parameters,
        "flop_estimates": flop_estimates,
        "checkpoint_bytes_local": checkpoint.stat().st_size,
    }
    if track == "NIR":
        result["snippet_operating_point"] = snippet_operating_point_metrics(
            ground_truth,
            predictions,
            manifest["operating_point"]["threshold"],
        )
    write_json(result_path, result)
    print(json.dumps(result, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, function in (("validate", validation), ("test", test)):
        sub = subparsers.add_parser(name)
        sub.add_argument("--track", required=True, choices=["RGB", "NIR", "rgb", "nir"])
        sub.add_argument("--model", required=True, choices=MODELS)
        sub.add_argument("--seed", type=int)
        sub.add_argument("--ratio", choices=NIR_RATIOS)
        if name == "validate":
            sub.add_argument("--execute-validation", action="store_true")
        else:
            sub.add_argument("--execute-test", action="store_true")
            sub.add_argument("--confirm")
        sub.set_defaults(function=function)
    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
