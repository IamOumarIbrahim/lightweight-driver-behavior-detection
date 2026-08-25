# Results

[← Back to Main README](../README.md)

> [!IMPORTANT]
> The six RGB YOLO result directories and the values below are final and authoritative. D-FINE-N and NIR cells remain intentionally blank until their frozen runs complete.

## RGB YOLO Summary

YOLO11n achieves higher mAP@0.5, precision, recall, micro/macro-F1, and sustained throughput. YOLO26n achieves the higher stricter mAP@0.5:0.95 while using slightly fewer parameters and estimated FLOPs. Values are sample mean ± sample standard deviation over seeds 13, 37, and 73; this SD measures optimization variability on one fixed subject split, not held-out-driver uncertainty.

### Overall Detection Performance

| Model | Runs | mAP@0.5:0.95 (↑) | mAP@0.5 (↑) | Precision (↑) | Recall (↑) | Micro-F1 (↑) | Macro-F1 (↑) | False Det./100 Neg. Frames (↓) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **YOLO11n** | 3 | 0.5538 ± 0.0041 | **0.9179 ± 0.0044** | **0.9343 ± 0.0116** | **0.8073 ± 0.0170** | **0.8660 ± 0.0049** | **0.8342 ± 0.0136** | **0.63 ± 0.04** |
| **YOLO26n** | 3 | **0.5698 ± 0.0081** | 0.8928 ± 0.0099 | 0.9256 ± 0.0112 | 0.7763 ± 0.0124 | 0.8444 ± 0.0114 | 0.7939 ± 0.0149 | 0.67 ± 0.06 |

False detections per 100 negative frames is a static per-frame count. It is not a modeled alert rate because no temporal confirmation or debouncing is applied.

<p align="center">
  <img src="../results/RGB/summary/figures/accuracy_vs_speed.png" alt="Accuracy versus speed for the RGB benchmark" width="700"><br>
  <sub><b>Figure 1.</b> RGB accuracy–speed trade-off on the RTX 4060 benchmark system. Error bars are sample SD; marker area represents serialized model size.</sub>
</p>

### Paired-Seed and Subject Analysis

Across all three paired seeds, YOLO11n is higher in mAP@0.5, micro/macro-F1, and throughput, while YOLO26n is higher in mAP@0.5:0.95. Per-class AP has a consistent YOLO11n direction for `yawning`, consistent YOLO26n directions for `hand_over_mouth` and `phone_use`, and no consistent direction for `drinking`.

Mean subject-level micro-F1 for the three fixed test drivers is 0.9494, 0.6241, and 0.9472 for YOLO11n and 0.9611, 0.5411, and 0.9229 for YOLO26n. This variation is descriptive and shows why seed SD cannot be interpreted as uncertainty over unseen drivers.

<p align="center">
  <img src="../results/RGB/summary/figures/qualitative_examples.png" alt="Representative RGB true positive, false detection, and false negative examples" width="780"><br>
  <sub><b>Figure 2.</b> Seed-13 true-positive, false-detection, and false-negative examples. The cue-like nominal negative is retained because it exposes semantic annotation uncertainty.</sub>
</p>

### Inference Efficiency

| Model | Model Fwd (ms) (↓) | Tensor→Det (ms) (↓) | Sustained FPS (↑) | Params (M) (↓) | GFLOPs (↓) |
| --- | --- | --- | --- | --- | --- |
| **YOLO11n** | **13.46 ± 0.20** | **14.30 ± 0.20** | **68.0 ± 1.3** | 2.59 | 6.44 |
| **YOLO26n** | 18.88 ± 0.10 | 19.21 ± 0.09 | 51.7 ± 0.2 | **2.51** | **5.78** |

Latency excludes disk I/O, decode, preprocessing, and metric computation. It includes the frozen batch-1 CUDA AMP FP16 boundaries described in [methodology.md](./methodology.md).

### Per-Class Average Precision

| Model | AP Yawning (↑) | AP Hand Over Mouth (↑) | AP Drinking (↑) | AP Phone Use (↑) |
| --- | --- | --- | --- | --- |
| **YOLO11n** | **0.5998 ± 0.0129** | 0.5013 ± 0.0300 | 0.6072 ± 0.0266 | 0.5069 ± 0.0069 |
| **YOLO26n** | 0.5613 ± 0.0348 | **0.5634 ± 0.0224** | **0.6174 ± 0.0140** | **0.5371 ± 0.0055** |

<p align="center">
  <img src="../results/RGB/summary/figures/per_class_ap.png" alt="Per-class RGB AP" width="700"><br>
  <sub><b>Figure 3.</b> RGB AP@0.5:0.95 by warning cue, mean ± sample SD.</sub>
</p>

Machine-readable per-run and aggregate values are in [`results/RGB/summary`](../results/RGB/summary/README.md). Qualitative analyses are organized under `results/RGB/{model}/seed_{13,37,73}`.

## Pending Results

| Track | Model | Required Runs | Publication Status |
| --- | --- | --- | --- |
| RGB | D-FINE-N | Seeds 13, 37, 73 | Awaiting complete three-seed validation/test result |
| NIR | YOLO11n | Ratios 1:2, 1:6; seed 13; 100 epochs; deterministic 1-FPS midpoint | Awaiting training, validation, test |
| NIR | YOLO26n | Ratios 1:2, 1:6; seed 13; 100 epochs; deterministic 1-FPS midpoint | Awaiting training, validation, test |
| NIR | D-FINE-N | Ratios 1:2, 1:6; seed 13; 100 epochs; deterministic 1-FPS midpoint | Awaiting training, validation, test |

> [!NOTE]
> Remaining model comparisons and figures will be finalized only after these runs produce checksum-backed files under `results/RGB/dfine_n` and `results/NIR`. RGB and NIR remain separate experiments rather than a controlled cross-spectral comparison.
