# NIR Publication Summary

This folder holds compact, publication-ready NIR aggregates and figures. The
training-negative exposure source contains all 16 completed seed-13 results for
YOLO11n, YOLO26n, D-FINE-N, SSDLite-MobileNetV3-Large, RT-DETRv2-S,
YOLOX-Nano, YOLOv10n, and YOLOv8n at 1:2 and 1:6. Its status is complete.
Checkpoints, logs, and raw predictions remain local under ignored `runs`.

Regenerate the source with `scripts\publication\analyze_nir.bat`, then rebuild
the figure with `scripts\publication\build_figures.bat`.
