# Setup Order

1. Run `01_create_environment.bat`.
2. Start Docker Desktop, then run `02_setup_backends.bat`.
3. Optionally run `03_setup_label_studio.bat` and `04_start_label_studio.bat`.
4. Run `05_setup_r_figures.bat` before building publication figures.

Every transitive dependency is locked. Downloads are cached under ignored
`third_party`, and installation uses that local cache. Setup 01 also runs the
CPU test suite. Setup 05 installs R for the current user only, without shortcuts
or file associations. Setup 02 also builds the pinned CUDA image that isolates
MMYOLO/RTMDet and EfficientDet from the main Python environment.
