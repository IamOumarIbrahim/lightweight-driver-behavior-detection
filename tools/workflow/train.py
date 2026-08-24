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
    NIR_RATIOS,
    NIR_SEED,
    REPO_ROOT,
    RGB_SEEDS,
    run_dir,
)
from tools.benchmark.protocol import ProtocolError, load_backends, load_protocol
from tools.setup.backends import ensure_dfine, ensure_ultralytics, ensure_weight


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
    if model.startswith("yolo"):
        results = training_dir / "results.csv"
        return (
            results.is_file()
            and len(results.read_text(encoding="utf-8").splitlines()) - 1 >= epochs
        )
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
    else:
        raise ProtocolError(f"Unknown track: {track}")
    root = run_dir(track, model, **identity)
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
    }


def require_dataset(plan: dict[str, Any]) -> None:
    if plan["model"].startswith("yolo"):
        import yaml

        config = yaml.safe_load(Path(plan["dataset"]).read_text(encoding="utf-8"))
        root = REPO_ROOT / config["path"]
        for split in ("train", "val"):
            path = root / config[split]
            if not path.is_file() or not path.read_text(encoding="utf-8").strip():
                raise ProtocolError(
                    f"Prepared {plan['track']} {split} list is missing: {path}"
                )
    else:
        if not Path(plan["dfine_config"]).is_file():
            raise ProtocolError(
                f"D-FINE configuration is missing: {plan['dfine_config']}"
            )


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
        (run_yolo if args.model.startswith("yolo") else run_dfine)(plan)
    except Exception as exc:
        append_event(log_path, "training_failed", error=repr(exc))
        raise
    append_event(log_path, "training_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
