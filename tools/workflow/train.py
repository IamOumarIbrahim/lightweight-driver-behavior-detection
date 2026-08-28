"""Safely launch or resume one frozen RGB or NIR training run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["YOLO_AUTOINSTALL"] = "false"

from tools.benchmark.paths import (
    CONFIGS_ROOT,
    MODELS,
    NIR_MODELS,
    NIR_RATIOS,
    NIR_SEED,
    REPO_ROOT,
    RGB_MODELS,
    RGB_SEEDS,
    run_dir,
)
from tools.benchmark.protocol import ProtocolError, load_backends, load_protocol
from tools.setup.backends import (
    ensure_dfine,
    ensure_patched_checkout,
    ensure_torchvision_ssdlite,
    ensure_ultralytics,
    ensure_weight,
)


ULTRALYTICS_MODELS = {"yolo11n", "yolo26n", "yolov10n", "yolov8n"}


def append_event(path: Path, event: str, **details: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **details,
    }
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def completed(model: str, training_dir: Path, epochs: int) -> bool:
    if model in ULTRALYTICS_MODELS:
        results = training_dir / "results.csv"
        return (
            results.is_file()
            and len(results.read_text(encoding="utf-8").splitlines()) - 1 >= epochs
        )
    if model == "yolox_nano":
        checkpoint = training_dir / "latest_ckpt.pth"
        if not checkpoint.is_file():
            return False
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return int(state.get("start_epoch", 0)) >= epochs
    if model == "ssdlite_mobilenet_v3_large":
        checkpoint = training_dir / "last.pt"
        if not checkpoint.is_file():
            return False
        import torch

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return int(state.get("epoch", -1)) >= epochs - 1
    log = training_dir / "log.txt"
    if not log.is_file():
        return False
    lines = [
        line for line in log.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lines:
        return False
    try:
        return int(json.loads(lines[-1]).get("epoch", -1)) >= epochs - 1
    except (ValueError, json.JSONDecodeError):
        return False


def build_plan(
    track: str, model: str, seed: int | None, ratio: str | None
) -> dict[str, Any]:
    track = track.upper()
    protocol = load_protocol(track)
    if track == "RGB":
        if seed not in RGB_SEEDS or ratio is not None:
            raise ProtocolError("RGB requires --seed 13, 37, or 73")
        if model not in RGB_MODELS:
            raise ProtocolError(f"Model {model} is not frozen for the RGB track")
        identity = {"seed": seed}
        dataset = CONFIGS_ROOT / "RGB" / "yolo" / "dataset.yaml"
        dfine_config = CONFIGS_ROOT / "RGB" / "dfine" / "base.yml"
    elif track == "NIR":
        if ratio not in NIR_RATIOS or seed not in (None, NIR_SEED):
            raise ProtocolError("NIR requires --ratio 1to2 or 1to6 and uses seed 13")
        seed = NIR_SEED
        identity = {"ratio": ratio}
        dataset = CONFIGS_ROOT / "NIR" / "yolo" / f"dataset_ratio_{ratio}.yaml"
        dfine_config = CONFIGS_ROOT / "NIR" / "dfine" / f"ratio_{ratio}.yml"
        if model not in NIR_MODELS:
            raise ProtocolError(f"Model {model} is not frozen for the NIR track")
    else:
        raise ProtocolError(f"Unknown track: {track}")
    root = run_dir(track, model, **identity)
    backend_configs = {
        "dfine_n": dfine_config,
        "ssdlite_mobilenet_v3_large": CONFIGS_ROOT
        / "NIR"
        / "ssdlite"
        / f"ratio_{ratio}.yaml",
        "rtdetrv2_s": CONFIGS_ROOT
        / "NIR"
        / "rtdetrv2"
        / f"ratio_{ratio}.yml",
        "yolox_nano": CONFIGS_ROOT
        / "NIR"
        / "yolox"
        / f"ratio_{ratio}.py",
    }
    return {
        "track": track,
        "model": model,
        "seed": seed,
        "ratio": ratio,
        "epochs": int(protocol["training"]["epochs"]),
        "batch": int(protocol["training"]["physical_batch_size"]),
        "nbs": int(protocol["training"]["effective_batch_size"]),
        "run_dir": str(root),
        "training_dir": str(root / "training"),
        "dataset": str(dataset),
        "dfine_config": str(dfine_config),
        "backend_config": str(backend_configs.get(model, dataset)),
    }


def require_dataset(plan: dict[str, Any]) -> None:
    if plan["model"] in ULTRALYTICS_MODELS:
        import yaml

        config = yaml.safe_load(Path(plan["dataset"]).read_text(encoding="utf-8"))
        root = REPO_ROOT / config["path"]
        for split in ("train", "val"):
            path = root / config[split]
            if not path.is_file() or not path.read_text(encoding="utf-8").strip():
                raise ProtocolError(
                    f"Prepared {plan['track']} {split} list is missing: {path}"
                )
    elif plan["model"] in {
        "dfine_n",
        "ssdlite_mobilenet_v3_large",
        "rtdetrv2_s",
        "yolox_nano",
    }:
        if not Path(plan["backend_config"]).is_file():
            raise ProtocolError(
                f"Backend configuration is missing: {plan['backend_config']}"
            )
    else:
        raise ProtocolError(f"Unsupported model backend: {plan['model']}")


def run_yolo(plan: dict[str, Any]) -> None:
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise ProtocolError("CUDA is required for training")
    ensure_ultralytics(False)
    backend = load_backends()["ultralytics"]["models"][plan["model"]]
    training_dir = Path(plan["training_dir"])
    last = training_dir / "weights" / "last.pt"
    if completed(plan["model"], training_dir, plan["epochs"]):
        print(f"Already complete; skipping {training_dir}")
        return
    if training_dir.exists() and not last.is_file():
        raise ProtocolError(
            f"Cannot safely resume incomplete directory without last.pt: {training_dir}"
        )
    if last.is_file():
        print(f"Resuming from {last}")
        YOLO(str(last)).train(resume=True)
        return
    weight = ensure_weight(backend, False)
    model = YOLO(str(weight["file"]))
    model.train(
        data=plan["dataset"],
        epochs=plan["epochs"],
        batch=plan["batch"],
        nbs=plan["nbs"],
        imgsz=640,
        seed=plan["seed"],
        device=0,
        project=plan["run_dir"],
        name="training",
        exist_ok=False,
        amp=True,
        optimizer="auto",
        cos_lr=False,
        val=True,
        patience=0,
        save=True,
        save_period=-1,
        plots=True,
        deterministic=True,
        cache=False,
        verbose=True,
    )


def run_dfine(plan: dict[str, Any]) -> None:
    import torch

    if not torch.cuda.is_available():
        raise ProtocolError("CUDA is required for training")
    ensure_dfine(False)
    backend = load_backends()["dfine"]
    training_dir = Path(plan["training_dir"])
    last = training_dir / "last.pth"
    if completed(plan["model"], training_dir, plan["epochs"]):
        print(f"Already complete; skipping {training_dir}")
        return
    if training_dir.exists() and any(training_dir.iterdir()) and not last.is_file():
        raise ProtocolError(
            f"Cannot safely resume incomplete directory without last.pth: {training_dir}"
        )
    command = [
        sys.executable,
        str(REPO_ROOT / "third_party" / "D-FINE" / "train.py"),
        "--config",
        plan["dfine_config"],
        "--device",
        "cuda:0",
        "--seed",
        str(plan["seed"]),
        "--use-amp",
        "--output-dir",
        plan["training_dir"],
        "--summary-dir",
        str(training_dir / "summary"),
    ]
    if last.is_file():
        command.extend(["--resume", str(last)])
    else:
        weight = ensure_weight(backend["weight"], False)
        command.extend(["--tuning", str(weight["file"])])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    environment["PYTHONUNBUFFERED"] = "1"
    training_dir.mkdir(parents=True, exist_ok=True)
    with (training_dir / "launcher.log").open(
        "a", encoding="utf-8", newline="\n"
    ) as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


def _subprocess_environment(extra_pythonpath: Path | None = None) -> dict[str, str]:
    """Build an environment where benchmark modules cannot be shadowed upstream."""

    environment = os.environ.copy()
    python_paths = [str(REPO_ROOT)]
    if extra_pythonpath is not None:
        python_paths.append(str(extra_pythonpath))
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _stream_command(
    command: list[str], training_dir: Path, *, extra_pythonpath: Path | None = None
) -> None:
    environment = _subprocess_environment(extra_pythonpath)
    training_dir.mkdir(parents=True, exist_ok=True)
    with (training_dir / "launcher.log").open(
        "a", encoding="utf-8", newline="\n"
    ) as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, command)


def run_ssdlite(plan: dict[str, Any]) -> None:
    import torch

    if not torch.cuda.is_available():
        raise ProtocolError("CUDA is required for training")
    ensure_torchvision_ssdlite()
    backend = load_backends()["torchvision_ssdlite"]
    training_dir = Path(plan["training_dir"])
    last = training_dir / "last.pt"
    if completed(plan["model"], training_dir, plan["epochs"]):
        print(f"Already complete; skipping {training_dir}")
        return
    if training_dir.exists() and any(training_dir.iterdir()) and not last.is_file():
        raise ProtocolError(
            f"Cannot safely resume incomplete directory without last.pt: {training_dir}"
        )
    weight = ensure_weight(backend["weight"], False)
    command = [
        sys.executable,
        "-m",
        "tools.backends.train_ssdlite",
        "--config",
        plan["backend_config"],
        "--pretrained",
        str(weight["file"]),
    ]
    if last.is_file():
        command.extend(["--resume", str(last)])
    _stream_command(command, training_dir)


def run_rtdetrv2(plan: dict[str, Any]) -> None:
    import torch

    if not torch.cuda.is_available():
        raise ProtocolError("CUDA is required for training")
    ensure_patched_checkout("rtdetrv2", False)
    backend = load_backends()["rtdetrv2"]
    training_dir = Path(plan["training_dir"])
    last = training_dir / "last.pth"
    if completed(plan["model"], training_dir, plan["epochs"]):
        print(f"Already complete; skipping {training_dir}")
        return
    if training_dir.exists() and any(training_dir.iterdir()) and not last.is_file():
        raise ProtocolError(
            f"Cannot safely resume incomplete directory without last.pth: {training_dir}"
        )
    checkout = REPO_ROOT / "third_party" / "RT-DETR"
    command = [
        sys.executable,
        str(checkout / "rtdetrv2_pytorch" / "tools" / "train.py"),
        "--config",
        plan["backend_config"],
        "--device",
        "cuda:0",
        "--seed",
        str(plan["seed"]),
        "--use-amp",
        "--output-dir",
        plan["training_dir"],
        "--summary-dir",
        str(training_dir / "summary"),
    ]
    if last.is_file():
        command.extend(["--resume", str(last)])
    else:
        weight = ensure_weight(backend["weight"], False)
        command.extend(["--tuning", str(weight["file"])])
    _stream_command(command, training_dir, extra_pythonpath=checkout / "rtdetrv2_pytorch")


def run_yolox(plan: dict[str, Any]) -> None:
    import torch

    if not torch.cuda.is_available():
        raise ProtocolError("CUDA is required for training")
    ensure_patched_checkout("yolox", False)
    backend = load_backends()["yolox"]
    checkout = REPO_ROOT / "third_party" / "YOLOX"
    training_dir = Path(plan["training_dir"])
    last = training_dir / "latest_ckpt.pth"
    if completed(plan["model"], training_dir, plan["epochs"]):
        print(f"Already complete; skipping {training_dir}")
        return
    if training_dir.exists() and any(training_dir.iterdir()) and not last.is_file():
        raise ProtocolError(
            f"Cannot safely resume incomplete directory without latest_ckpt.pth: {training_dir}"
        )
    command = [
        sys.executable,
        str(checkout / "tools" / "train.py"),
        "--exp_file",
        plan["backend_config"],
        "--devices",
        "1",
        "--batch-size",
        str(plan["batch"]),
        "--fp16",
        "--experiment-name",
        "training",
    ]
    if last.is_file():
        command.extend(["--resume", "--ckpt", str(last)])
    else:
        weight = ensure_weight(backend["weight"], False)
        command.extend(["--ckpt", str(weight["file"])])
    _stream_command(command, training_dir, extra_pythonpath=checkout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", required=True, choices=["RGB", "NIR", "rgb", "nir"])
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ratio", choices=NIR_RATIOS)
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args.track, args.model, args.seed, args.ratio)
    print(json.dumps(plan, indent=2))
    require_dataset(plan)
    if not args.execute_training:
        print("Dry-run passed. Training was not started.")
        return 0
    log_path = Path(plan["run_dir"]) / "events.jsonl"
    append_event(log_path, "training_requested", plan=plan)
    try:
        runners = {
            "dfine_n": run_dfine,
            "ssdlite_mobilenet_v3_large": run_ssdlite,
            "rtdetrv2_s": run_rtdetrv2,
            "yolox_nano": run_yolox,
        }
        runner = run_yolo if args.model in ULTRALYTICS_MODELS else runners[args.model]
        runner(plan)
    except Exception as exc:
        append_event(log_path, "training_failed", error=repr(exc))
        raise
    append_event(log_path, "training_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
