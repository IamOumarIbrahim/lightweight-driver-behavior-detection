# NIR Publication Summary

This folder holds compact, publication-ready NIR aggregates and figures. The
training-negative exposure source currently contains ten completed seed-13
results for YOLO11n, YOLO26n, D-FINE-N, SSDLite-MobileNetV3-Large, and
RT-DETRv2-S at 1:2 and 1:6. Its status remains partial while YOLOX-Nano,
YOLOv10n, and YOLOv8n are trained and evaluated. Checkpoints, logs, and raw
predictions remain local under ignored `runs`.

Regenerate the source with `scripts\publication\analyze_nir.bat`, then rebuild
the figure with `scripts\publication\build_figures.bat`.
