"""Run all ten NIR training jobs sequentially with safe resume."""

from __future__ import annotations

import argparse
import subprocess
import sys

from tools.benchmark.paths import MODELS, NIR_RATIOS, REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-training", action="store_true")
    args = parser.parse_args()
    if not args.execute_training:
        print("Dry-run: ten jobs will run in model order, ratio 1to2 then 1to6.")
        return 0
    for model in MODELS:
        for ratio in NIR_RATIOS:
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
