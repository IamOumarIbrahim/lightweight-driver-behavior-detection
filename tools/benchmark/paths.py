"""Canonical repository paths and artifact naming."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_ROOT = REPO_ROOT / "configs"
DATA_ROOT = REPO_ROOT / "data"
RUNS_ROOT = REPO_ROOT / "runs"
RESULTS_ROOT = REPO_ROOT / "results"
THIRD_PARTY_ROOT = REPO_ROOT / "third_party"

TRACKS = ("RGB", "NIR")
RGB_MODELS = ("yolo11n", "yolo26n", "dfine_n")
NIR_MODELS = (
    "yolo11n",
    "yolo26n",
    "rtmdet_tiny",
    "efficientdet_d1",
    "dfine_n",
)
MODELS = NIR_MODELS
RGB_SEEDS = (13, 37, 73)
NIR_RATIOS = ("1to2", "1to6")
NIR_SEED = 13


def repo_path(value: str | Path) -> Path:
    """Resolve a repository-relative path without depending on the shell cwd."""

    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def run_key(track: str, *, seed: int | None = None, ratio: str | None = None) -> str:
    if track == "RGB" and seed in RGB_SEEDS and ratio is None:
        return f"seed_{seed}"
    if track == "NIR" and ratio in NIR_RATIOS and seed in (None, NIR_SEED):
        return f"ratio_{ratio}"
    raise ValueError(
        f"Invalid run identity: track={track!r}, seed={seed!r}, ratio={ratio!r}"
    )


def run_dir(track: str, model: str, **identity: object) -> Path:
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    return RUNS_ROOT / track / model / run_key(track, **identity)


def result_dir(track: str, model: str, **identity: object) -> Path:
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    return RESULTS_ROOT / track / model / run_key(track, **identity)


def is_authoritative_rgb_yolo(track: str, model: str, path: Path) -> bool:
    """Return whether a user-approved RGB YOLO result already freezes the run."""

    return (
        track == "RGB"
        and model in {"yolo11n", "yolo26n"}
        and (path / "analysis.json").is_file()
    )
