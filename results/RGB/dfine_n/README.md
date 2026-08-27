# RGB D-FINE-N Results

All three 220-epoch training runs, validation-only selections, calibrations, and
single protected-test passes are complete. The frozen three-seed protected-test
aggregate is **0.5538 ± 0.0089 mAP@0.5:0.95**, **0.9022 ± 0.0229 mAP@0.5**,
and **0.9114 ± 0.0266 micro-F1**. Tensor-to-final-detections throughput is
**40.4 ± 1.2 FPS** on the frozen RTX 4060 protocol.

[`training_runs.json`](training_runs.json) records the controls, per-seed
validation and protected-test metrics, thresholds, suite and manifest IDs, and
SHA-256 fingerprints. Checkpoints, dense predictions, inference artifacts, and
raw logs remain local.

The six earlier YOLO protected results retain their original six-run suite ID.
The D-FINE-N passes use the later expanded nine-run suite ID, so the guarded
aggregator intentionally does not emit a mixed-suite nine-run artifact.
