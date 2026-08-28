# Experiment Workflows

`train.py` safely starts or resumes one run. `train_new_nir.py` sequentially
runs only the ten configured extension jobs, skips completed pairs, and excludes
the six original NIR jobs. `evaluate.py` enforces validation before the single
confirmation-gated test pass.
