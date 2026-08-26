# Limitations

[← Back to Main README](../README.md)

## Scope and Generalization

- RGB and NIR use different source datasets, ontologies, sampling, and annotation sources. NIR is a separate exploratory experiment, not a paired domain-shift or controlled spectral comparison.
- Subject-disjoint partitions reduce identity leakage but do not establish generalization to unseen cameras, countries, vehicle cabins, eyewear, skin tones, or clinical fatigue states.
- The benchmark localizes visible driver cues in single frames; it does not infer duration, fatigue, intent, impairment, or medical condition.

## Experimental Uncertainty

- RGB reports three predeclared seeds on one fixed subject split. The SD measures optimization variability, not generalization uncertainty across unseen drivers.
- NIR is a 100-epoch training-negative exposure study at seed 13. The 1:6 condition has three times the unique negatives and 6,000 rather than 2,600 optimizer updates (approximately 2.31 times as many), so the design does not isolate ratio under matched training signal or estimate multi-seed variance.
- Models retain native optimization and postprocessing, including anchor-based, anchor-free/NMS, and end-to-end paths. Results compare complete released systems rather than architecture alone.
- Runtime values are specific to the stated RTX 4060 software/hardware environment. Accuracy is portable; latency should be re-profiled on deployment hardware.
- FLOPs are tool-dependent estimates and are reported with their estimator rather than treated as exact operation counts.

## Annotation and Metric Constraints

- The ontology permits at most one target cue per frame. Real driving can contain simultaneous behaviors outside this mutual-exclusion rule.
- One primary annotator completed one smooth pass without independent review. Automated checks establish file/geometric integrity but not semantic agreement; the published qualitative false positive exposes a cue-like nominal negative that may be an ontology-boundary case or missed label.
- Neighboring RGB frames from the same recording remain correlated. NIR avoids within-snippet duplication by evaluating one deterministic midpoint frame per one-second snippet, but different snippets from the same recording can still be correlated.
- Each NIR midpoint box is linearly interpolated from human tracklet keyframes. Rapid non-linear hand motion may introduce localization error at the sampled instant, and 1-FPS sampling does not characterize cue evolution within the second.
- False detections per 100 negative frames is a static per-frame count and complements, rather than replaces, COCO AP. It is not an operational alert rate; safety decisions require application-specific temporal logic and prospective validation.

## Data Availability

DMD and Drive&Act media are governed by their respective licenses and are not redistributed. This repository publishes authored annotations, split metadata, integrity hashes, and deterministic preparation tools so licensed users can reconstruct the benchmark locally.
