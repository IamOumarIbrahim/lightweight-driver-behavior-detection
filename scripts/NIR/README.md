# NIR Workflow

Six named training launchers cover three models at ratios 1:2 and 1:6. All jobs
use seed 13 and a fixed 100-epoch maximum. `train_all_six.bat` runs them in the
frozen model order with safe resume; pass `--yes` only when an unattended start
is intentional. Run `validate_all_six.bat` before the confirmation-gated
`test_all_six.bat`. All active backends run natively in the main Windows virtual
environment.
