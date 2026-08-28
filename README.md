<h1 align="center">Subject-Disjoint Benchmarking of Lightweight Detectors for Visible Driver-Cue Localization</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Modality-RGB%20%7C%20NIR-blueviolet?style=flat" alt="Modality: RGB | NIR">
  <img src="https://img.shields.io/badge/Input-640%C3%97640-555?style=flat" alt="Input: 640×640">
  <img src="https://img.shields.io/badge/NIR%20Detectors-8%20native%20Windows%20systems-4c1?style=flat" alt="Eight NIR detector systems">
</p>

<p align="center">
  <a href="https://raw.githubusercontent.com/IamOumarIbrahim/lightweight-driver-behavior-detection/main/docs/manuscript/main.pdf" download="main.pdf"><img src="https://img.shields.io/badge/📄_Manuscript-Download_PDF-e02424?style=for-the-badge&logo=adobeacrobatreader&logoColor=white" alt="Download the manuscript PDF"></a>
</p>

## Table of Contents

- [Overview](#overview)
- [Current Benchmark Status](#current-benchmark-status)
- [Quick Reproduction](#quick-reproduction)
- [Repository Organization](#repository-organization)
- [Authors & Citation](#authors--citation)
- [Acknowledgments & License](#acknowledgments--license)

## Overview

### Research Question

How do complete nano detector systems trade single-frame cue localization, false detections on negative frames, and synchronized throughput under subject-disjoint evaluation?

### Abstract

This repository benchmarks released lightweight detector systems for bounding-box localization of discrete, momentary visual driver cues. The primary RGB benchmark compares [Ultralytics YOLO11n](https://docs.ultralytics.com/models/yolo11/), [Ultralytics YOLO26n](https://docs.ultralytics.com/models/yolo26), and [D-FINE-N](https://github.com/Peterande/D-FINE) on the [Driver Monitoring Dataset (DMD)](https://dmd.vicomtech.org/). The NIR training-negative exposure extension adds SSDLite-MobileNetV3-Large, RT-DETRv2-S, YOLOX-Nano, YOLOv10n, and YOLOv8n on [Drive&Act](https://driveandact.com/). Both tracks use subject-disjoint partitions, validation-only operating-point selection, and confirmation-gated protected testing. The task does not infer temporal fatigue, intent, or driver state.

## Current Benchmark Status

> [!IMPORTANT]
> All nine RGB runs and the original six NIR runs are complete. Ten native-Windows NIR extension runs are configured and pending training.

| Track | Model | Conditions | Status |
| --- | --- | --- | --- |
| RGB | YOLO11n | Seeds 13, 37, 73 | Final |
| RGB | YOLO26n | Seeds 13, 37, 73 | Final |
| RGB | D-FINE-N | Seeds 13, 37, 73 | Final three-seed result |
| NIR | YOLO11n, YOLO26n, D-FINE-N | Ratios 1:2 and 1:6, seed 13, 100 epochs | Six protected results published |
| NIR extension | SSDLite-MobileNetV3-Large, RT-DETRv2-S, YOLOX-Nano, YOLOv10n, YOLOv8n | Ratios 1:2 and 1:6, seed 13, 100 epochs | Ready for sequential training |

See the [results documentation](./docs/results.md) for authoritative values and the [methodology](./docs/methodology.md) for the frozen protocol.

## Quick Reproduction

The BAT files are the supported Windows interface; they resolve the repository root automatically and require confirmation before training or protected testing.

```bat
git clone https://github.com/IamOumarIbrahim/lightweight-driver-behavior-detection.git
cd lightweight-driver-behavior-detection

scripts\setup\01_create_environment.bat
scripts\setup\02_setup_backends.bat
scripts\preflight.bat
```

Place the licensed source folders at `data/DMD` and `data/Drive&Act`, then run:

```bat
scripts\data\01_unpack_sources.bat
scripts\data\02_extract_rgb_frames.bat
scripts\data\04_prepare_nir_1fps.bat
scripts\data\05_build_nir_review_snippets.bat  REM optional Label Studio review media
scripts\data\06_check_rgb_dataset.bat
scripts\data\07_check_nir_dataset.bat
```

Published annotations are already under `data/annotations`; dataset frames and trained checkpoints remain local. The following PowerShell command verifies the pinned native backends and starts or resumes only the ten pending extension jobs (five new models at two ratios) sequentially. It does not select the six completed YOLO11n, YOLO26n, or D-FINE-N jobs:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\NIR\train_new_ten.ps1 -Yes
```

Validation and protected testing remain deliberately separate.

> [!CAUTION]
> Dataset licenses govern the source media. This repository redistributes authored annotations and split metadata, not DMD or Drive&Act frames/videos.

## Repository Organization

```text
lightweight-driver-behavior-detection/
├── configs/               # Frozen RGB/NIR protocols, backend pins, patches
├── data/                  # Published annotations; local licensed media is ignored
├── docs/                  # Methodology, results, limitations, IEEE manuscript
├── results/               # Publication-ready metrics and qualitative artifacts
├── scripts/               # Descriptive Windows BAT launchers
├── tools/                 # Tested Python implementation behind the launchers
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── requirements.lock.txt
```

Only these six semantic folders are published. Local `runs`, `third_party`, and `ARCHIVE` directories are ignored.

## Authors & Citation

- **Oumar Mamoun Ibrahim** — Department of Computer Engineering, University of Sharjah<br>
  [U22200741@sharjah.ac.ae](mailto:U22200741@sharjah.ac.ae) · [ORCID 0009-0008-0312-1605](https://orcid.org/0009-0008-0312-1605)
- **Dr. Mohamad Khairi bin Ishak** — Department of Computer Engineering, University of Sharjah<br>
  [mishak@sharjah.ac.ae](mailto:mishak@sharjah.ac.ae) · [ORCID 0000-0002-3554-0061](https://orcid.org/0000-0002-3554-0061)
- **Khalid Ammar** — Department of Electrical and Computer Engineering, College of Engineering and Information Technology, Ajman University<br>
  [k.ammar@ajman.ac.ae](mailto:k.ammar@ajman.ac.ae)

```bibtex
@misc{ibrahim2026lightweight,
  title     = {Subject-Disjoint Benchmarking of Lightweight Detectors for Visible Driver-Cue Localization},
  author    = {Ibrahim, Oumar Mamoun and Bin Ishak, Mohamad Khairi and Ammar, Khalid},
  year      = {2026},
  note      = {Manuscript under review}
}
```

## Acknowledgments & License

This work builds on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [D-FINE](https://github.com/Peterande/D-FINE), [Label Studio](https://github.com/HumanSignal/label-studio), [DMD](https://dmd.vicomtech.org/), and [Drive&Act](https://driveandact.com/). Code is licensed under [Apache License 2.0](LICENSE); third-party datasets and dependencies retain their own licenses.
