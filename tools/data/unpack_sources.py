"""Safely unpack archives placed inside data/DMD or data/Drive&Act."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path

from tools.benchmark.paths import DATA_ROOT


def safe_destination(root: Path, member: str) -> Path:
    destination = (root / member).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f"Archive member escapes destination: {member}")
    return destination


def unpack(path: Path) -> Path:
    destination = (
        path.with_suffix("")
        if path.suffix.lower() == ".zip"
        else path.parent / path.name.split(".tar")[0]
    )
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                safe_destination(destination, member.filename)
            archive.extractall(destination)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                safe_destination(destination, member.name)
            archive.extractall(destination, filter="data")
    else:
        raise ValueError(f"Unsupported archive: {path}")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="Delete successfully extracted archives",
    )
    args = parser.parse_args()
    archives = []
    for root in (DATA_ROOT / "DMD", DATA_ROOT / "Drive&Act"):
        if root.exists():
            archives.extend(
                path
                for path in root.rglob("*")
                if path.is_file()
                and (path.suffix.lower() == ".zip" or ".tar" in path.name.lower())
            )
    if not archives:
        print("No archives found. Existing folders were left unchanged.")
        return 0
    for archive in sorted(archives):
        destination = unpack(archive)
        print(f"Extracted {archive} -> {destination}")
        if args.delete_archives:
            archive.unlink()
            print(f"Deleted extracted archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
