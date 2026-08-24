from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

plt.switch_backend("Agg")


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    REPO_ROOT / "results" / "RGB" / "summary" / "final_benchmark_aggregate.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "results" / "RGB" / "summary" / "figures"
EXPECTED_MODELS = ("yolo11n", "yolo26n", "dfine_n")
MODEL_LABELS = {
    "yolo11n": "YOLO11n",
    "yolo26n": "YOLO26n",
    "dfine_n": "D-FINE-N",
}
MODEL_COLORS = {
    "yolo11n": "#0072B2",
    "yolo26n": "#D55E00",
    "dfine_n": "#009E73",
}
MODEL_MARKERS = {
    "yolo11n": "o",
    "yolo26n": "s",
    "dfine_n": "^",
}


@dataclass(frozen=True)
class Estimate:
    mean: float
    sample_std: float


@dataclass(frozen=True)
class ModelResult:
    model_id: str
    runs: int
    map_50_95: Estimate
    sustained_fps: Estimate
    artifact_mb: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _estimate(row: dict[str, Any], key: str) -> Estimate:
    value = row.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"{key} must contain mean and sample_std")
    mean = float(value["mean"])
    sample_std = float(value["sample_std"])
    if not math.isfinite(mean) or not math.isfinite(sample_std) or sample_std < 0:
        raise ValueError(f"{key} contains an invalid estimate")
    return Estimate(mean=mean, sample_std=sample_std)


def load_rgb_aggregate(path: Path) -> list[ModelResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact") != "dms_eval_aggregate":
        raise ValueError(f"Unexpected aggregate type in {path}")
    if payload.get("dispersion") != "sample_standard_deviation":
        raise ValueError("Publication error bars require sample standard deviation")

    parsed: dict[str, ModelResult] = {}
    for row in payload.get("rows", []):
        model_id = str(row.get("model_id", ""))
        if model_id not in EXPECTED_MODELS:
            continue
        runs = int(row.get("runs", 0))
        if runs < 2:
            raise ValueError(f"{model_id} needs at least two runs for an SD error bar")
        artifact_bytes = _estimate(row, "inference_artifact_bytes").mean
        if artifact_bytes <= 0:
            raise ValueError(f"{model_id} has an invalid inference artifact size")
        if model_id in parsed:
            raise ValueError(f"Duplicate aggregate row for {model_id}")
        parsed[model_id] = ModelResult(
            model_id=model_id,
            runs=runs,
            map_50_95=_estimate(row, "map_50_95"),
            sustained_fps=_estimate(row, "tensor_to_final_detections_sustained_fps"),
            artifact_mb=artifact_bytes / 1_000_000.0,
        )

    if not parsed:
        raise ValueError(f"No supported completed RGB model rows in {path}")
    return [parsed[model_id] for model_id in EXPECTED_MODELS if model_id in parsed]


def _configure_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "lightweight-driver-behavior-detection",
            "svg.fonttype": "none",
        }
    )


def _axis_limits(
    values: list[tuple[float, float]], minimum_padding: float
) -> tuple[float, float]:
    low = min(mean - spread for mean, spread in values)
    high = max(mean + spread for mean, spread in values)
    span = max(high - low, minimum_padding)
    padding = 0.16 * span
    return low - padding, high + padding


def build_accuracy_speed_figure(
    rows: list[ModelResult], output_dir: Path
) -> dict[str, Path]:
    _configure_ieee_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(3.5, 2.45), constrained_layout=True)

    label_offsets = ((5, 6), (5, -7), (-5, 6))
    for index, row in enumerate(rows):
        color = MODEL_COLORS[row.model_id]
        marker = MODEL_MARKERS[row.model_id]
        axis.errorbar(
            row.sustained_fps.mean,
            row.map_50_95.mean,
            xerr=row.sustained_fps.sample_std,
            yerr=row.map_50_95.sample_std,
            fmt="none",
            ecolor=color,
            elinewidth=0.9,
            capsize=2.2,
            capthick=0.9,
            zorder=2,
        )
        axis.scatter(
            row.sustained_fps.mean,
            row.map_50_95.mean,
            s=32.0 * row.artifact_mb,
            marker=marker,
            facecolor=color,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
        x_offset, y_offset = label_offsets[index]
        axis.annotate(
            f"{MODEL_LABELS[row.model_id]}\n{row.artifact_mb:.2f} MB",
            (row.sustained_fps.mean, row.map_50_95.mean),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="left" if x_offset > 0 else "right",
            va="bottom" if y_offset > 0 else "top",
            fontsize=7,
        )

    axis.set_xlim(
        *_axis_limits(
            [(row.sustained_fps.mean, row.sustained_fps.sample_std) for row in rows],
            2.0,
        )
    )
    axis.set_ylim(
        *_axis_limits(
            [(row.map_50_95.mean, row.map_50_95.sample_std) for row in rows], 0.01
        )
    )
    axis.set_xlabel("Sustained throughput (FPS)")
    axis.set_ylabel("mAP@0.5:0.95")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.grid(True, color="#D0D0D0", linewidth=0.45, alpha=0.8)
    axis.set_axisbelow(True)
    axis.text(
        0.985,
        0.025,
        "marker area scales with model file size",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#444444",
    )

    outputs = {
        "pdf": output_dir / "accuracy_vs_speed.pdf",
        "svg": output_dir / "accuracy_vs_speed.svg",
        "png": output_dir / "accuracy_vs_speed.png",
    }
    fig.savefig(
        outputs["pdf"],
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={
            "Title": "RGB accuracy-efficiency trade-off",
            "Creator": "tools.publication.figures",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        outputs["svg"],
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={
            "Title": "RGB accuracy-efficiency trade-off",
            "Creator": "tools.publication.figures",
            "Date": None,
        },
    )
    svg_text = outputs["svg"].read_text(encoding="utf-8")
    outputs["svg"].write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    fig.savefig(
        outputs["png"],
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={"Software": "tools.publication.figures"},
    )
    plt.close(fig)
    return outputs


def write_manifest(
    source: Path, rows: list[ModelResult], outputs: dict[str, Path]
) -> Path:
    manifest_path = outputs["png"].with_suffix(".manifest.json")
    payload = {
        "artifact": "publication_figure_manifest",
        "figure": "rgb_accuracy_vs_speed",
        "source": str(source.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_sha256": _sha256(source),
        "models": [row.model_id for row in rows],
        "pending_models": [
            model_id
            for model_id in EXPECTED_MODELS
            if model_id not in {row.model_id for row in rows}
        ],
        "dispersion": "sample_standard_deviation",
        "bubble_area": "inference_artifact_bytes",
        "outputs": {
            kind: {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for kind, path in outputs.items()
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic IEEE-sized publication figures"
    )
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE, help="Frozen RGB aggregate JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Publication figure directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    rows = load_rgb_aggregate(source)
    outputs = build_accuracy_speed_figure(rows, output_dir)
    manifest = write_manifest(source, rows, outputs)
    print(
        f"Built RGB accuracy-efficiency figure for: {', '.join(row.model_id for row in rows)}"
    )
    if len(rows) < len(EXPECTED_MODELS):
        pending = [
            model_id
            for model_id in EXPECTED_MODELS
            if model_id not in {row.model_id for row in rows}
        ]
        print(f"Pending completed aggregate rows: {', '.join(pending)}")
    for path in (*outputs.values(), manifest):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
