@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" (
  echo Project environment is missing. Run scripts\setup\01_create_environment.bat first.
  exit /b 2
)
echo This exports existing protected-test predictions only. It does not load a model or run inference.
.venv\Scripts\python.exe -m tools.publication.export_rgb_predictions
if errorlevel 1 (
  echo.
  echo RGB prediction export failed. Confirm that the six frozen local result.json files exist.
  exit /b 1
)
echo.
echo Sanitized RGB test predictions are ready under results\RGB\MODEL\seed_SEED.
endlocal
