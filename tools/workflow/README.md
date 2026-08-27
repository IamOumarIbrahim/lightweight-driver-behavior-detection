# Experiment Workflows

`train.py` safely starts or resumes one run. `train_new_nir.py` sequentially
runs only the ten pending extension jobs, excluding the six completed original
NIR jobs. `evaluate.py` enforces validation before the single
confirmation-gated test pass.
