"""Run every frozen NIR validation or protected test sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys

from tools.benchmark.paths import MODELS, NIR_RATIOS, REPO_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["validate", "test"])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        jobs = len(MODELS) * len(NIR_RATIOS)
        print(
            f"Dry-run: {jobs} NIR {args.phase} jobs will run in model order, "
            "ratio 1to2 then 1to6."
        )
        return 0
    for model in MODELS:
        for ratio in NIR_RATIOS:
            command = [
                sys.executable,
                "-m",
                "tools.workflow.evaluate",
                args.phase,
                "--track",
                "NIR",
                "--model",
                model,
                "--ratio",
                ratio,
            ]
            if args.execute and args.phase == "validate":
                command.append("--execute-validation")
            elif args.execute:
                command.extend(["--execute-test", "--confirm", "RUN_PROTECTED_TEST"])
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
