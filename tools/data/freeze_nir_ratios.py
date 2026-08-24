"""Freeze the NIR 1:2 negatives as a nested seed-13 subset of the 1:6 pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict

from tools.benchmark.paths import DATA_ROOT, NIR_SEED
from tools.data.prepare_nir import result_value, signature


def rank(task: dict) -> str:
    payload = f"{NIR_SEED}|{signature(task)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def subject_quotas(groups: dict[str, list[dict]], target: int) -> dict[str, int]:
    total = sum(len(items) for items in groups.values())
    exact = {subject: len(items) * target / total for subject, items in groups.items()}
    quotas = {subject: math.floor(value) for subject, value in exact.items()}
    remaining = target - sum(quotas.values())
    priority = sorted(
        groups, key=lambda subject: (-(exact[subject] - quotas[subject]), subject)
    )
    for subject in priority[:remaining]:
        quotas[subject] += 1
    return quotas


def nested_subset(tasks: list[dict], target_negatives: int = 540) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for task in tasks:
        if task["data"]["split"] == "train" and result_value(task) is None:
            groups[task["data"]["subject"]].append(task)
    quotas = subject_quotas(groups, target_negatives)
    selected = {
        signature(task)
        for subject, items in groups.items()
        for task in sorted(items, key=rank)[: quotas[subject]]
    }
    return [
        task
        for task in tasks
        if task["data"]["split"] != "train"
        or result_value(task) is not None
        or signature(task) in selected
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="Replace the published 1:2 ratio file"
    )
    args = parser.parse_args()
    root = DATA_ROOT / "annotations" / "NIR"
    source = json.loads((root / "ratio_1to6.json").read_text(encoding="utf-8"))
    frozen = nested_subset(source)
    payload = (json.dumps(frozen, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    print(f"tasks: {len(frozen)}")
    print(f"sha256: {hashlib.sha256(payload).hexdigest()}")
    if args.write:
        (root / "ratio_1to2.json").write_bytes(payload)
        print("Updated data/annotations/NIR/ratio_1to2.json")
    else:
        print("Dry-run only; add --write to update the frozen file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
