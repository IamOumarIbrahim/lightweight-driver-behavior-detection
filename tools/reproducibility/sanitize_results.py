"""Remove machine-specific paths from publication-ready RGB results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.benchmark.paths import RESULTS_ROOT


def write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    analysis_hashes = {}
    for model in ("yolo11n", "yolo26n"):
        for seed in (13, 37, 73):
            path = RESULTS_ROOT / "RGB" / model / f"seed_{seed}" / "analysis.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            for examples in data.get("examples", {}).values():
                for example in examples:
                    if "rendered_path" in example:
                        example["rendered_path"] = (
                            f"images/{Path(example['rendered_path']).name}"
                        )
            write(path, data)
            analysis_hashes[(model, seed)] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

    aggregate_path = RESULTS_ROOT / "RGB" / "summary" / "final_benchmark_aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    for run in aggregate.get("runs", []):
        model, seed = run["model_id"], int(run["training_seed"])
        run["qualitative_analysis"]["path"] = (
            f"results/RGB/{model}/seed_{seed}/analysis.json"
        )
        run["qualitative_analysis"]["sha256"] = analysis_hashes[(model, seed)]
        run["source"] = f"runs/RGB/{model}/seed_{seed}/test/result.json"
    write(aggregate_path, aggregate)
    print("Sanitized six RGB analyses and the aggregate provenance paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
