# NIR Workflow

The ten pending extension jobs cover five new models at ratios 1:2 and 1:6. All
jobs use seed 13, 640x640 input, a fixed 100-epoch maximum, physical batch 8, and
effective batch 32. `train_new_ten.ps1` verifies every pinned backend and runs
only these pending jobs sequentially with safe resume. It excludes the completed
YOLO11n, YOLO26n, and D-FINE-N runs. Pass `-Yes` only for an unattended start.
Run `validate_all_sixteen.bat` before the confirmation-gated
`test_all_sixteen.bat`.

The suite is YOLO11n, YOLO26n, D-FINE-N, SSDLite-MobileNetV3-Large,
RT-DETRv2-S, YOLOX-Nano, YOLOv10n, and YOLOv8n. All active backends run natively
inside the main Windows virtual environment; Docker and WSL are not used.
