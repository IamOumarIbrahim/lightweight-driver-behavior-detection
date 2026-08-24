@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv-labelstudio\Scripts\python.exe" py -3.12 -m venv .venv-labelstudio || exit /b 1
if not exist "third_party\packages-label-studio" mkdir "third_party\packages-label-studio"
.venv-labelstudio\Scripts\python.exe -m pip download --dest "third_party\packages-label-studio" -r requirements-label-studio.lock.txt || exit /b 1
.venv-labelstudio\Scripts\python.exe -m pip install --no-index --find-links "third_party\packages-label-studio" -r requirements-label-studio.lock.txt || exit /b 1
.venv-labelstudio\Scripts\python.exe -m tools.setup.label_studio || exit /b 1
echo Label Studio environment ready. Run 04_start_label_studio.bat.
