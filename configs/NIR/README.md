# NIR Configuration

The only conditions are ratios 1:2 and 1:6 at seed 13. Ratio changes affect
training negatives only; validation and test artifacts are shared. Every run uses
a fixed 640x640 input, physical batch 8, effective batch 32, a 100-epoch maximum,
and no early stopping. Eight native-Windows models produce 16 sequential jobs.
