# Backend Patches

The Ultralytics, D-FINE, RT-DETRv2, and YOLOX patches enforce the shared update
cadence. They make incomplete gradient-accumulation windows sample-correct and
preserve the state required for safe resume. Backend setup verifies exact
applicability against pinned upstream revisions before training.
