# Limitations

[← Back to Main README](../README.md)

## Scope and Generalization

- RGB and NIR use different source datasets and ontologies. Cross-spectral comparisons therefore reflect complete track conditions, not a paired image-by-image domain shift experiment.
- Subject-disjoint partitions reduce identity leakage but do not establish generalization to unseen cameras, countries, vehicle cabins, eyewear, skin tones, or clinical fatigue states.
- The benchmark detects visible warning cues; it does not infer driver intent, impairment, or medical condition.

## Experimental Uncertainty

- RGB reports three predeclared seeds. NIR is a controlled training-negative-ratio study at seed 13, so it does not estimate multi-seed variance.
- Runtime values are specific to the stated RTX 4060 software/hardware environment. Accuracy is portable; latency should be re-profiled on deployment hardware.
- FLOPs are tool-dependent estimates and are reported with their estimator rather than treated as exact operation counts.

## Annotation and Metric Constraints

- The ontology permits at most one target cue per frame. Real driving can contain simultaneous behaviors outside this mutual-exclusion rule.
- Ten-frame NIR boxes are linearly interpolated from human tracklet keyframes. Rapid non-linear hand motion may introduce small localization error between keyframes.
- FAR counts detections on negative frames and complements, rather than replaces, COCO AP. Operational safety decisions require application-specific alert logic and prospective validation.

## Data Availability

DMD and Drive&Act media are governed by their respective licenses and are not redistributed. This repository publishes authored annotations, split metadata, integrity hashes, and deterministic preparation tools so licensed users can reconstruct the benchmark locally.
