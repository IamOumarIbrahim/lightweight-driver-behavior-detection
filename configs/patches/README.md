# Backend Patches

The Ultralytics and D-FINE patches enforce the shared update cadence. The
EfficientDet patch adds stable output directories and sample-correct four-step
gradient accumulation to the pinned upstream trainer; the Docker build checks
the patch against the pinned source commit before applying it.

These patches make incomplete gradient-accumulation windows sample-correct for
both backends. The backend setup verifies exact applicability before training.
