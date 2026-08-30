"""Run CPU-only repository and dataset preflight checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from pypdf import PdfReader

from tools.benchmark.paths import REPO_ROOT, RESULTS_ROOT
from tools.benchmark.protocol import ProtocolError, validate_protocol
from tools.data.prepare_nir import load_ratio_tasks, signature, validate_counts


class Checks:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[str] = []

    def check(self, condition: bool, message: str) -> None:
        (self.passed if condition else self.failed).append(message)


def tracked_files() -> list[str]:
    output = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _font_is_embedded(font_reference: object) -> bool:
    font = font_reference.get_object()
    if font.get("/Subtype") == "/Type3":
        return True
    descriptor = font.get("/FontDescriptor")
    if font.get("/Subtype") == "/Type0":
        descendants = font.get("/DescendantFonts", [])
        if descendants:
            descriptor = descendants[0].get_object().get("/FontDescriptor")
    if descriptor is None:
        return False
    descriptor = descriptor.get_object()
    return any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3"))


def unembedded_fonts(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    failures: set[str] = set()
    visited: set[tuple[int, int] | int] = set()

    def walk_resources(resource_reference: object) -> None:
        resources = resource_reference.get_object()
        indirect = getattr(resources, "indirect_reference", None)
        marker: tuple[int, int] | int = (
            (indirect.idnum, indirect.generation) if indirect else id(resources)
        )
        if marker in visited:
            return
        visited.add(marker)
        for font_reference in resources.get("/Font", {}).values():
            font = font_reference.get_object()
            if not _font_is_embedded(font_reference):
                failures.add(str(font.get("/BaseFont", "unnamed font")))
        for xobject_reference in resources.get("/XObject", {}).values():
            xobject = xobject_reference.get_object()
            nested = xobject.get("/Resources")
            if nested is not None:
                walk_resources(nested)

    for page in reader.pages:
        resources = page.get("/Resources")
        if resources is not None:
            walk_resources(resources)
    return sorted(failures)


def repository_checks(checks: Checks) -> None:
    files = tracked_files()
    root_folders = sorted(
        {Path(path).parts[0] for path in files if len(Path(path).parts) > 1}
    )
    checks.check(
        len(root_folders) <= 7,
        f"At most seven tracked root folders ({', '.join(root_folders)})",
    )
    checks.check(
        root_folders == ["configs", "data", "docs", "results", "scripts", "tools"],
        "Tracked root folders use the canonical six-folder layout",
    )
    forbidden_suffixes = (".pt", ".pth", ".onnx", ".engine", ".pyc")
    checks.check(
        not [path for path in files if path.lower().endswith(forbidden_suffixes)],
        "No checkpoints, engines, or bytecode are tracked",
    )
    checks.check(
        not [path for path in files if path.startswith("runs/")],
        "Local run artifacts are not tracked",
    )
    result_root_files = [
        path
        for path in files
        if Path(path).parent == Path("results") and path != "results/README.md"
    ]
    checks.check(
        not result_root_files,
        f"No ad hoc files are tracked directly under results: {result_root_files}",
    )
    for folder in root_folders:
        checks.check(
            (REPO_ROOT / folder / "README.md").is_file(), f"{folder}/README.md exists"
        )

    offenders = []
    extensions = {
        ".py",
        ".r",
        ".bat",
        ".yaml",
        ".yml",
        ".xml",
        ".json",
        ".md",
        ".tex",
        ".bib",
        ".txt",
        ".csv",
    }
    tokens = (
        "C:\\Dev\\",
        "dataset2",
        "ratio1to3",
        "ratio_1to3",
        "from core.",
        "import core.",
    )
    for relative in files:
        # This checker necessarily names the forbidden tokens it searches for.
        if relative == "tools/reproducibility/preflight.py":
            continue
        path = REPO_ROOT / relative
        if path.suffix.lower() not in extensions or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(token in text for token in tokens):
            offenders.append(relative)
    checks.check(
        not offenders,
        f"Active code/configs contain no obsolete or machine-specific paths: {offenders}",
    )

    link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)|(?:href|src)=[\"']([^\"']+)[\"']")
    broken_links = []
    for relative in files:
        if not relative.lower().endswith(".md"):
            continue
        source = REPO_ROOT / relative
        if not source.is_file():
            continue
        for match in link_pattern.finditer(source.read_text(encoding="utf-8")):
            target = next(group for group in match.groups() if group)
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0].split("?", 1)[0]
            if target and not (source.parent / target).resolve().exists():
                broken_links.append(f"{relative} -> {target}")
    checks.check(not broken_links, f"Local documentation links resolve: {broken_links}")

    manuscript = (REPO_ROOT / "docs" / "manuscript" / "main.tex").read_text(
        encoding="utf-8"
    )
    bibliography = (REPO_ROOT / "docs" / "manuscript" / "references.bib").read_text(
        encoding="utf-8"
    )
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", manuscript)
        for key in group.split(",")
    }
    available = set(re.findall(r"@\w+\{([^,]+),", bibliography))
    checks.check(
        cited <= available,
        f"Every manuscript citation has a bibliography entry: {sorted(cited - available)}",
    )
    checks.check(
        not re.search(
            r"lorem|placeholder reference|author (?:five|six|seven|eight|nine|ten)",
            manuscript + bibliography,
            re.IGNORECASE,
        ),
        "Manuscript and bibliography contain no filler prose or placeholder references",
    )
    checks.check(
        r"\begin{eqnarray}" not in manuscript and "$$" not in manuscript,
        "Displayed mathematics uses IEEE-recommended equation environments",
    )
    checks.check(
        r"\mathrm{FP}_{\mathrm{neg}}" in manuscript
        and r"N_{\mathrm{neg}}" in manuscript
        and r"\newcommand{\fd}{\mathrm{FD}_{100}}" in manuscript,
        "Equation acronyms are upright and scalar variables remain italic",
    )

    manuscript_pdf = REPO_ROOT / "docs" / "manuscript" / "main.pdf"
    checks.check(manuscript_pdf.is_file(), "Compiled manuscript review PDF exists")
    if manuscript_pdf.is_file():
        manuscript_reader = PdfReader(str(manuscript_pdf))
        manuscript_pages = len(manuscript_reader.pages)
        checks.check(
            manuscript_pages <= 6,
            f"Conference manuscript is at most six pages ({manuscript_pages} pages)",
        )
        pdf_version = float(manuscript_reader.pdf_header.removeprefix("%PDF-"))
        checks.check(
            1.4 <= pdf_version < 1.9,
            f"Manuscript PDF version is IEEE Xplore compatible ({pdf_version:.1f})",
        )
        checks.check(
            not manuscript_reader.is_encrypted,
            "Manuscript PDF has no password or security encryption",
        )
        annotations = [
            annotation
            for page in manuscript_reader.pages
            for annotation in page.get("/Annots", [])
        ]
        checks.check(
            not annotations and not manuscript_reader.outline,
            "Manuscript PDF contains no links, annotations, or bookmarks",
        )
        checks.check(
            not manuscript_reader.attachments,
            "Manuscript PDF contains no attachments or package payloads",
        )
        missing_fonts = unembedded_fonts(manuscript_pdf)
        checks.check(
            not missing_fonts,
            f"Every manuscript and included-figure font is embedded: {missing_fonts}",
        )

    accepted_graphics = {".ps", ".eps", ".pdf", ".png", ".tif", ".tiff"}
    graphic_directories = (
        RESULTS_ROOT / "summary" / "figures",
        RESULTS_ROOT / "RGB" / "summary" / "figures",
        RESULTS_ROOT / "NIR" / "summary" / "figures",
    )
    included_graphics = set(
        re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", manuscript)
    )
    resolved_graphics: list[Path] = []
    unresolved_graphics: list[str] = []
    for graphic in sorted(included_graphics):
        candidates = [directory / graphic for directory in graphic_directories]
        resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
        if resolved is None:
            unresolved_graphics.append(graphic)
        else:
            resolved_graphics.append(resolved)
    checks.check(
        not unresolved_graphics,
        f"Every included manuscript graphic resolves: {unresolved_graphics}",
    )
    rejected_formats = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in resolved_graphics
        if path.suffix.lower() not in accepted_graphics
    ]
    checks.check(
        not rejected_formats,
        f"Included graphics use IEEE-accepted formats: {rejected_formats}",
    )
    figure_font_failures = {}
    for path in resolved_graphics:
        if path.suffix.lower() != ".pdf":
            continue
        failures = unembedded_fonts(path)
        if failures:
            figure_font_failures[path.relative_to(REPO_ROOT).as_posix()] = failures
    checks.check(
        not figure_font_failures,
        f"Every included PDF graphic embeds or subsets its fonts: {figure_font_failures}",
    )

    aggregate_path = RESULTS_ROOT / "RGB" / "summary" / "final_benchmark_aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate_runs = {
        (item["model_id"], int(item["training_seed"])): item
        for item in aggregate["runs"]
    }
    artifact_failures = []
    for model in ("yolo11n", "yolo26n"):
        for seed in (13, 37, 73):
            result = RESULTS_ROOT / "RGB" / model / f"seed_{seed}"
            analysis_path = result / "analysis.json"
            if not analysis_path.is_file():
                artifact_failures.append(
                    f"missing {analysis_path.relative_to(REPO_ROOT)}"
                )
                continue
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            run = aggregate_runs.get((model, seed))
            digest = hashlib.sha256(analysis_path.read_bytes()).hexdigest()
            if run is None or run["qualitative_analysis"]["sha256"] != digest:
                artifact_failures.append(
                    f"aggregate hash mismatch for {model}/seed_{seed}"
                )
            for examples in analysis.get("examples", {}).values():
                for example in examples:
                    rendered = Path(example["rendered_path"])
                    image_path = result / rendered
                    if rendered.is_absolute() or not image_path.is_file():
                        artifact_failures.append(
                            f"invalid rendered path for {model}/seed_{seed}: {rendered}"
                        )
                    elif (
                        hashlib.sha256(image_path.read_bytes()).hexdigest()
                        != example["rendered_sha256"]
                    ):
                        artifact_failures.append(
                            f"image hash mismatch for {model}/seed_{seed}: {rendered}"
                        )
    checks.check(
        not artifact_failures,
        f"Six RGB YOLO publication artifacts and hashes resolve: {artifact_failures}",
    )

    secondary_failures = []
    secondary_path = RESULTS_ROOT / "RGB" / "summary" / "secondary_analysis.json"
    if not secondary_path.is_file():
        secondary_failures.append("missing secondary_analysis.json")
    else:
        secondary = json.loads(secondary_path.read_text(encoding="utf-8"))
        if any(secondary.get("annotation_audit", {}).get("problems", {}).values()):
            secondary_failures.append("RGB structural annotation audit contains failures")
        secondary_runs = {
            (item["model_id"], int(item["training_seed"])): item
            for item in secondary.get("runs", [])
        }
        for model in ("yolo11n", "yolo26n"):
            for seed in (13, 37, 73):
                prediction_path = (
                    RESULTS_ROOT
                    / "RGB"
                    / model
                    / f"seed_{seed}"
                    / "test_predictions.json"
                )
                if not prediction_path.is_file():
                    secondary_failures.append(
                        f"missing {prediction_path.relative_to(REPO_ROOT)}"
                    )
                    continue
                envelope = json.loads(prediction_path.read_text(encoding="utf-8"))
                provenance_path = Path(
                    str(envelope.get("provenance", {}).get("source_result", ""))
                )
                run = secondary_runs.get((model, seed))
                digest = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
                if (
                    envelope.get("model_id") != model
                    or int(envelope.get("training_seed", -1)) != seed
                    or provenance_path.is_absolute()
                ):
                    secondary_failures.append(
                        f"unsafe or mismatched prediction envelope for {model}/seed_{seed}"
                    )
                if run is None or run.get("prediction_sha256") != digest:
                    secondary_failures.append(
                        f"secondary-analysis hash mismatch for {model}/seed_{seed}"
                    )
    checks.check(
        not secondary_failures,
        f"RGB public predictions and secondary analysis resolve: {secondary_failures}",
    )

    figure_failures = []
    figure_manifests = [
        RESULTS_ROOT / "summary" / "figures" / "protocol_workflow.manifest.json",
        RESULTS_ROOT
        / "RGB"
        / "summary"
        / "figures"
        / "normalized_model_comparison.manifest.json",
        RESULTS_ROOT / "RGB" / "summary" / "figures" / "accuracy_vs_speed.manifest.json",
        RESULTS_ROOT / "RGB" / "summary" / "figures" / "per_class_ap.manifest.json",
        RESULTS_ROOT / "RGB" / "summary" / "figures" / "qualitative_examples.manifest.json",
        RESULTS_ROOT / "RGB" / "summary" / "figures" / "subject_sensitivity.manifest.json",
        RESULTS_ROOT / "RGB" / "summary" / "figures" / "validation_operating_point.manifest.json",
        RESULTS_ROOT / "NIR" / "summary" / "figures" / "training_negative_exposure.manifest.json",
    ]
    for figure_manifest_path in figure_manifests:
        if not figure_manifest_path.is_file():
            figure_failures.append(f"missing {figure_manifest_path.name}")
            continue
        figure_manifest = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
        if figure_manifest.get("generator") != "ggplot2":
            figure_failures.append(
                f"{figure_manifest_path.name} was not generated by ggplot2"
            )
        referenced = {
            "source": {
                "path": figure_manifest.get("source"),
                "sha256": figure_manifest.get("source_sha256"),
            }
        }
        referenced.update(figure_manifest.get("outputs", {}))
        referenced.update(
            {
                f"input-{index}": item
                for index, item in enumerate(figure_manifest.get("inputs", []), 1)
            }
        )
        if set(figure_manifest.get("outputs", {})) != {"pdf", "svg", "png"}:
            figure_failures.append("publication figure must provide PDF, SVG, and PNG")
        for name, artifact in referenced.items():
            relative = Path(str(artifact.get("path", "")))
            resolved = (REPO_ROOT / relative).resolve()
            if relative.is_absolute() or not resolved.is_relative_to(
                REPO_ROOT.resolve()
            ):
                figure_failures.append(f"unsafe {name} path: {relative}")
            elif not resolved.is_file():
                figure_failures.append(f"missing {name}: {relative}")
            elif hashlib.sha256(resolved.read_bytes()).hexdigest() != artifact.get(
                "sha256"
            ):
                figure_failures.append(f"hash mismatch for {name}: {relative}")
    checks.check(
        not figure_failures,
        f"ggplot2 publication figure sources and output hashes resolve: {figure_failures}",
    )


def rgb_checks(checks: Checks, require_images: bool) -> None:
    protocol = validate_protocol("RGB")
    annotations = json.loads(
        (REPO_ROOT / protocol["dataset"]["annotations"]).read_text(encoding="utf-8")
    )
    checks.check(
        len(annotations["images"]) == 15723 and len(annotations["annotations"]) == 3001,
        "RGB authoritative counts match",
    )
    splits = json.loads(
        (REPO_ROOT / protocol["dataset"]["splits"]).read_text(encoding="utf-8")
    )
    subject_sets = {
        "train": set(splits["train"]),
        "val": set(splits["validation"]),
        "test": set(splits["test"]),
    }
    checks.check(
        not (subject_sets["train"] & subject_sets["val"])
        and not (subject_sets["train"] & subject_sets["test"])
        and not (subject_sets["val"] & subject_sets["test"]),
        "RGB subject partitions are disjoint",
    )
    image_subjects = {
        Path(image["file_name"]).parts[1] for image in annotations["images"]
    }
    checks.check(
        image_subjects == set().union(*subject_sets.values()),
        "Every RGB image subject is assigned exactly once",
    )
    processed = REPO_ROOT / "data" / "processed" / "RGB"
    for split, expected in (("train", 9087), ("val", 3423), ("test", 3213)):
        coco = processed / "coco" / "evaluation" / f"instances_{split}.json"
        checks.check(coco.is_file(), f"RGB {split} evaluation COCO exists")
        if coco.is_file():
            checks.check(
                len(json.loads(coco.read_text(encoding="utf-8"))["images"]) == expected,
                f"RGB {split} derived count matches",
            )
    if require_images:
        checks.check(
            len(list((processed / "images").rglob("*.jpg"))) == 15723,
            "All 15,723 RGB frames exist",
        )


def nir_checks(checks: Checks, require_images: bool) -> None:
    protocol = validate_protocol("NIR")
    source_tasks = json.loads(
        (REPO_ROOT / protocol["dataset"]["annotations"]).read_text(encoding="utf-8")
    )
    expected_source = int(protocol["dataset"]["expected"]["source_snippets"])
    checks.check(
        len(source_tasks) == expected_source
        and len({int(task["id"]) for task in source_tasks}) == expected_source,
        f"NIR source pool contains {expected_source} unique snippets",
    )
    splits = json.loads(
        (REPO_ROOT / protocol["dataset"]["splits"]).read_text(encoding="utf-8")
    )
    subject_sets = {name: set(splits[name]) for name in ("train", "val", "test")}
    checks.check(
        not (subject_sets["train"] & subject_sets["val"])
        and not (subject_sets["train"] & subject_sets["test"])
        and not (subject_sets["val"] & subject_sets["test"]),
        "NIR subject partitions are disjoint",
    )
    checks.check(
        all(
            task["data"]["subject"] in subject_sets[task["data"]["split"]]
            for task in source_tasks
        ),
        "Every NIR source task matches its frozen subject partition",
    )
    tasks = load_ratio_tasks(protocol)
    try:
        validate_counts(tasks, protocol)
        checks.check(True, "NIR ratio and split counts match")
    except ProtocolError as exc:
        checks.check(False, str(exc))
    eval_signatures = [
        [
            signature(task)
            for task in tasks[ratio]
            if task["data"]["split"] in {"val", "test"}
        ]
        for ratio in ("1to2", "1to6")
    ]
    checks.check(
        eval_signatures[0] == eval_signatures[1],
        "NIR validation and test identities are byte-order identical across ratios",
    )
    checks.check(
        all(
            "local_path" not in json.dumps(task)
            for ratio in tasks.values()
            for task in ratio
        ),
        "Published NIR annotations contain no local paths",
    )
    processed = REPO_ROOT / "data" / "processed" / "NIR"
    expected_lists = {
        "yolo/ratio_1to2/train.txt": 810,
        "yolo/ratio_1to6/train.txt": 1890,
        "yolo/evaluation/val.txt": 881,
        "yolo/evaluation/test.txt": 850,
    }
    for relative, expected in expected_lists.items():
        path = processed / relative
        checks.check(
            path.is_file()
            and len(path.read_text(encoding="utf-8").splitlines()) == expected,
            f"NIR {relative} contains {expected} deterministic 1-FPS frames",
        )
    if require_images:
        required_ids = {int(task["id"]) for ratio in tasks.values() for task in ratio}
        image_count = sum(
            1
            for task_id in required_ids
            for frame in range(1, 2)
            if (
                processed / "images" / f"task_{task_id:05d}_frame_{frame:02d}.jpg"
            ).is_file()
        )
        checks.check(
            image_count == len(required_ids),
            f"All {len(required_ids):,} union NIR midpoint frames exist",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--track", choices=["repository", "RGB", "NIR", "all"], default="repository"
    )
    parser.add_argument("--require-images", action="store_true")
    args = parser.parse_args()
    checks = Checks()
    if args.track in {"repository", "all"}:
        repository_checks(checks)
    if args.track in {"RGB", "all"}:
        rgb_checks(checks, args.require_images)
    if args.track in {"NIR", "all"}:
        nir_checks(checks, args.require_images)
    print(f"PASS: {len(checks.passed)}")
    for message in checks.passed:
        print(f"  [OK] {message}")
    if checks.failed:
        print(f"FAIL: {len(checks.failed)}")
        for message in checks.failed:
            print(f"  [X] {message}")
        return 2
    print("Preflight complete: no checked blockers found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
