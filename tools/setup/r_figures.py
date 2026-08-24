"""Install the pinned project-local R and ggplot2 publication runtime."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from tools.benchmark.paths import REPO_ROOT

LOCK_PATH = REPO_ROOT / "tools" / "publication" / "R-runtime.lock.json"
PACKAGE_LOCK_PATH = REPO_ROOT / "tools" / "publication" / "R-packages.lock.csv"
THIRD_PARTY = REPO_ROOT / "third_party"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_third_party_path(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    resolved.relative_to(THIRD_PARTY.resolve(strict=False))
    return resolved


def download(url: str, destination: Path, expected_sha256: str) -> None:
    destination = require_third_party_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == expected_sha256:
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        while block := response.read(1024 * 1024):
            handle.write(block)
    actual = sha256(partial)
    if actual != expected_sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch for {url}: {actual}")
    os.replace(partial, destination)


def r_environment(library: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["LC_ALL"] = "English_United States.utf8"
    environment["R_LIBS_USER"] = str(library)
    return environment


def run_r(rscript: Path, library: Path, expression: str, *, check: bool) -> int:
    completed = subprocess.run(
        [str(rscript), "--vanilla", "-e", expression],
        cwd=REPO_ROOT,
        env=r_environment(library),
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if check and completed.returncode:
        raise RuntimeError(f"R exited with code {completed.returncode}")
    return completed.returncode


def verify_packages(rscript: Path, library: Path) -> bool:
    library_r = library.as_posix().replace("'", "\\'")
    lock_r = PACKAGE_LOCK_PATH.as_posix().replace("'", "\\'")
    expression = (
        f"lib <- '{library_r}'; lock <- read.csv('{lock_r}', check.names=FALSE); "
        ".libPaths(c(lib, .libPaths())); ip <- installed.packages(lib.loc=lib); "
        "ok <- all(lock$Package %in% rownames(ip)) && "
        "all(vapply(seq_len(nrow(lock)), function(i) "
        "identical(as.character(ip[lock$Package[i], 'Version']), "
        "lock$Version[i]), logical(1))); "
        "quit(status=if (ok) 0L else 3L)"
    )
    return run_r(rscript, library, expression, check=False) == 0


def install_r(installer: Path, install_dir: Path) -> Path:
    rscript = install_dir / "bin" / "Rscript.exe"
    if rscript.is_file():
        return rscript
    arguments = [
        str(installer),
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        "/CURRENTUSER",
        "/NOICONS",
        f"/DIR={install_dir}",
        "/COMPONENTS=main,x64",
        "/MERGETASKS=!desktopicon,!quicklaunchicon,!recordversion,!associate",
    ]
    completed = subprocess.run(
        arguments,
        cwd=REPO_ROOT,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode:
        raise RuntimeError(f"R installer exited with code {completed.returncode}")
    if not rscript.is_file():
        raise RuntimeError(f"Rscript was not installed at {rscript}")
    return rscript


def main() -> int:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    installer_record = lock["installer"]
    installer_name = Path(str(installer_record["url"])).name
    installer = THIRD_PARTY / installer_name
    install_dir = require_third_party_path(REPO_ROOT / str(lock["install_directory"]))
    library = require_third_party_path(REPO_ROOT / str(lock["library_directory"]))
    cache = require_third_party_path(REPO_ROOT / str(lock["package_cache"]))

    download(
        str(installer_record["url"]),
        installer,
        str(installer_record["sha256"]),
    )
    rscript = install_r(installer, install_dir)
    library.mkdir(parents=True, exist_ok=True)
    if not verify_packages(rscript, library):
        package_archives = []
        with PACKAGE_LOCK_PATH.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                archive_name = f"{row['Package']}_{row['Version']}.zip"
                url = (
                    "https://cran.r-project.org/bin/windows/contrib/"
                    f"{lock['cran_binary_series']}/{archive_name}"
                )
                archive = cache / archive_name
                download(url, archive, row["SHA256"])
                package_archives.append(archive)
        library_r = library.as_posix().replace("'", "\\'")
        archive_paths = [
            "'" + archive.as_posix().replace("'", "\\'") + "'"
            for archive in package_archives
        ]
        expression = (
            f"lib <- '{library_r}'; "
            "dir.create(lib, recursive=TRUE, showWarnings=FALSE); "
            f"install.packages(c({','.join(archive_paths)}), "
            "repos=NULL, type='win.binary', lib=lib)"
        )
        run_r(rscript, library, expression, check=True)
    if not verify_packages(rscript, library):
        raise RuntimeError("Pinned R package verification failed after installation")
    print(
        f"R {lock['r_version']} and pinned ggplot2 packages are ready under "
        "third_party."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"R figure setup failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
