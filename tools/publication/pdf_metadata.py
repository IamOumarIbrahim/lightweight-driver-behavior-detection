"""Normalize ggplot2 PDF metadata without altering graphical page content."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, ByteStringObject

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXED_CREATION_DATE = "D:20260825000000+04'00'"


def normalize_pdf(path: Path, title: str, figure_id: str, source_sha256: str) -> None:
    resolved = path.resolve()
    resolved.relative_to(REPO_ROOT.resolve())
    reader = PdfReader(resolved)
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    writer._info.clear()  # Cairo's timestamp lives in this cloned dictionary.
    writer.add_metadata(
        {
            "/Title": title,
            "/Creator": "R ggplot2",
            "/Producer": "pypdf deterministic metadata normalization",
            "/CreationDate": FIXED_CREATION_DATE,
        }
    )
    stable_id = hashlib.sha256(f"{figure_id}|{source_sha256}".encode("ascii")).digest()
    writer._ID = ArrayObject([ByteStringObject(stable_id), ByteStringObject(stable_id)])
    temporary = resolved.with_suffix(".normalized.tmp.pdf")
    with temporary.open("wb") as handle:
        writer.write(handle)
    os.replace(temporary, resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("title")
    parser.add_argument("figure_id")
    parser.add_argument("source_sha256")
    args = parser.parse_args()
    normalize_pdf(args.pdf, args.title, args.figure_id, args.source_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
