@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" (
  echo Project environment is missing. Run scripts\setup\01_create_environment.bat first.
  exit /b 2
)
.venv\Scripts\python.exe -m tools.publication.analyze_rgb
if errorlevel 1 (
  echo.
  echo RGB secondary analysis failed. Run export_rgb_predictions.bat first.
  exit /b 1
)
echo.
echo RGB class, subject, paired-seed, and annotation-audit outputs are ready.
endlocal
