"""Run only the pending NIR extension training jobs sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys

from tools.benchmark.paths import NIR_EXTENSION_MODELS, NIR_RATIOS, REPO_ROOT


def jobs() -> tuple[tuple[str, str], ...]:
    return tuple(
        (model, ratio)
        for model in NIR_EXTENSION_MODELS
        for ratio in NIR_RATIOS
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args()
    pending_jobs = jobs()
    if not args.execute_training:
        print(
            f"Dry-run: {len(pending_jobs)} pending extension jobs will run in model "
            "order, ratio 1to2 then 1to6."
        )
        return 0
    for model, ratio in pending_jobs:
        command = [
            sys.executable,
            "-m",
            "tools.workflow.train",
            "--track",
            "NIR",
            "--model",
            model,
            "--ratio",
            ratio,
            "--execute-training",
        ]
        print(f"\n=== {model} / {ratio} ===", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
