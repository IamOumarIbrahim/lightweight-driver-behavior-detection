"""Install/verify pinned model backends and official pretrained weights."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import urllib.request
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.benchmark.paths import REPO_ROOT
from tools.benchmark.protocol import (
    ProtocolError,
    load_backends,
    resolve_repo_path,
    sha256_file,
)


def run(command: list[str], cwd: Path = REPO_ROOT) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _run_git_patch(
    checkout: Path,
    patch: Path,
    *,
    reverse: bool = False,
    check_only: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    """Apply/check a patch after normalizing only its transport line endings."""

    command = ["git", "apply"]
    if reverse:
        command.append("--reverse")
    if check_only:
        command.append("--check")
    command.append("-")
    payload = patch.read_bytes().replace(b"\r\n", b"\n")
    return subprocess.run(
        command, cwd=checkout, input=payload, capture_output=True, check=False
    )


def _patch_ultralytics_trainer_source(source: str) -> str:
    """Apply the frozen accumulation patch using exact pinned-version anchors."""

    marker = "# DMS-Eval: keep the shared four-step update cadence"
    if marker in source:
        return source
    replacements = {
        "                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))": "                    # DMS-Eval: keep the shared four-step update cadence during LR/momentum warm-up.\n"
        "                    self.accumulate = max(round(self.args.nbs / self.batch_size), 1)",
        "                    # Backward\n                    self.scaler.scale(self.loss).backward()": "                    # DMS-Eval: sample-normalize the final incomplete accumulation window.\n"
        "                    from tools.benchmark.training import accumulation_loss_scale\n\n"
        "                    self.loss *= accumulation_loss_scale(\n"
        "                        batch_index=i,\n"
        "                        total_batches=nb,\n"
        "                        accumulation_steps=self.accumulate,\n"
        '                        current_batch_size=int(batch["img"].shape[0]),\n'
        "                        total_samples=len(self.train_loader.dataset),\n"
        "                        physical_batch_size=self.batch_size,\n"
        '                        batch_loss_reduction="sum",\n'
        "                    )\n"
        "                    self.scaler.scale(self.loss).backward()",
        "                if ni - last_opt_step >= self.accumulate:": "                if (i + 1) % self.accumulate == 0 or (i + 1) == nb:",
    }
    patched = source
    for old, new in replacements.items():
        if old not in patched:
            raise ProtocolError(
                "Pinned Ultralytics trainer does not match the frozen patch anchors"
            )
        patched = patched.replace(old, new, 1)
    return patched


def ensure_ultralytics(install: bool) -> dict[str, str]:
    spec = load_backends()["ultralytics"]
    try:
        actual_version = version(spec["package"])
    except PackageNotFoundError as exc:
        raise ProtocolError("Pinned Ultralytics package is missing") from exc
    if actual_version != spec["version"]:
        raise ProtocolError(
            f"Ultralytics is {actual_version}, expected {spec['version']}"
        )
    package_spec = importlib.util.find_spec("ultralytics")
    if package_spec is None or not package_spec.submodule_search_locations:
        raise ProtocolError("Cannot locate the pinned Ultralytics package")
    package_root = Path(next(iter(package_spec.submodule_search_locations)))
    recipe_path = package_root.parent / spec["recipe"]["file"]
    if sha256_file(recipe_path) != spec["recipe"]["sha256"]:
        raise ProtocolError("Pinned Ultralytics default recipe fingerprint mismatch")
    trainer_path = package_root / "engine" / "trainer.py"
    source = trainer_path.read_text(encoding="utf-8")
    patched = _patch_ultralytics_trainer_source(source)
    if patched != source:
        if not install:
            raise ProtocolError("Ultralytics fixed-accumulation patch is not applied")
        trainer_path.write_text(patched, encoding="utf-8", newline="\n")
    if (
        "# DMS-Eval: keep the shared four-step update cadence"
        not in trainer_path.read_text(encoding="utf-8")
    ):
        raise ProtocolError("Ultralytics fixed-accumulation patch verification failed")
    return {"version": actual_version, "recipe": "verified", "patch": "applied"}


def ensure_dfine(install: bool) -> dict[str, str]:
    spec = load_backends()["dfine"]
    checkout = resolve_repo_path(spec["checkout"])
    if not checkout.exists():
        if not install:
            raise ProtocolError("Pinned D-FINE checkout is missing")
        run(["git", "clone", "--filter=blob:none", spec["repository"], str(checkout)])
        run(["git", "checkout", "--detach", spec["commit"]], checkout)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
    ).strip()
    if head != spec["commit"]:
        raise ProtocolError(f"D-FINE checkout is {head}, expected {spec['commit']}")
    recipe = checkout / spec["recipe"]["file"]
    if sha256_file(recipe) != spec["recipe"]["sha256"]:
        raise ProtocolError("Pinned D-FINE-N official recipe fingerprint mismatch")
    patch = REPO_ROOT / "configs" / "patches" / "dfine-gradient-accumulation.patch"
    reverse = _run_git_patch(checkout, patch, reverse=True, check_only=True)
    if reverse.returncode != 0:
        if not install:
            raise ProtocolError("D-FINE gradient-accumulation patch is not applied")
        applied = _run_git_patch(checkout, patch)
        if applied.returncode != 0:
            details = applied.stderr.decode("utf-8", errors="replace").strip()
            raise ProtocolError(f"D-FINE gradient-accumulation patch failed: {details}")
        verified = _run_git_patch(checkout, patch, reverse=True, check_only=True)
        if verified.returncode != 0:
            raise ProtocolError(
                "D-FINE gradient-accumulation patch verification failed"
            )
    return {"commit": head, "patch": "applied"}


def ensure_weight(spec: dict, install: bool) -> dict[str, str | int]:
    path = resolve_repo_path(spec["file"])
    if not path.exists():
        if not install:
            raise ProtocolError(f"Missing official weight: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        urllib.request.urlretrieve(spec["url"], temporary)
        temporary.replace(path)
    actual_hash = sha256_file(path)
    if actual_hash != spec["sha256"] or path.stat().st_size != spec["size_bytes"]:
        raise ProtocolError(f"Weight integrity check failed: {path}")
    return {"file": str(path), "size_bytes": path.stat().st_size, "sha256": actual_hash}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--install",
        action="store_true",
        help="Download/clone missing official artifacts and apply the pinned patch",
    )
    args = parser.parse_args()
    backends = load_backends()
    report = {
        "ultralytics": ensure_ultralytics(args.install),
        "dfine_checkout": ensure_dfine(args.install),
        "weights": {},
    }
    for model_id, spec in backends["ultralytics"]["models"].items():
        report["weights"][model_id] = ensure_weight(spec, args.install)
    report["weights"]["dfine_n"] = ensure_weight(
        backends["dfine"]["weight"], args.install
    )
    for key, value in report.items():
        print(f"{key}: {value}")
    print("Backend artifact verification PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
