# NIR Workflow

Ten named training launchers cover five models at ratios 1:2 and 1:6. All jobs
use seed 13 and a fixed 100-epoch maximum. `train_all_ten.bat` runs them in the
frozen model order with safe resume; pass `--yes` only when an unattended start
is intentional. Run `validate_all_ten.bat` before the confirmation-gated
`test_all_ten.bat`.

RTMDet-Tiny and EfficientDet-D1 use the pinned Docker image built by
`scripts/setup/02_setup_backends.bat`; the three existing backends remain in the
main virtual environment. Each added-model launch repeats a lightweight CUDA
and mounted-data preflight before creating training state. It verifies the exact
train/validation COCO counts, category IDs, and every referenced midpoint image,
then prints a machine-readable preflight report. Missing GPU access or data now
fails before either upstream trainer starts.
