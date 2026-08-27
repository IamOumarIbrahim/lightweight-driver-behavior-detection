# Backend Patches

The Ultralytics and D-FINE patches enforce the shared update cadence.

These patches make incomplete gradient-accumulation windows sample-correct for
both backends. Backend setup verifies exact applicability before training.
