# NIR Publication Summary

This folder holds compact, publication-ready NIR aggregates and figures. The
training-negative exposure source currently contains the six completed seed-13
results for YOLO11n, YOLO26n, and D-FINE-N at 1:2 and 1:6. Its status remains
partial until RTMDet-Tiny and EfficientDet-D1 complete both exposures.
Checkpoints, logs, and raw predictions remain local under ignored `runs`.

Regenerate the source with `scripts\publication\analyze_nir.bat`, then rebuild
the figure with `scripts\publication\build_figures.bat`.
