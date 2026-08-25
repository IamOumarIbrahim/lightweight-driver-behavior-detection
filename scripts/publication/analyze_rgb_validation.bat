@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" (
  echo Project environment is missing. Run scripts\setup\01_create_environment.bat first.
  exit /b 2
)
.venv\Scripts\python.exe -m tools.publication.analyze_rgb_validation
if errorlevel 1 (
  echo.
  echo RGB validation sweep failed. Confirm all six frozen validation runs exist.
  exit /b 1
)
echo.
echo Public validation operating-point curves are ready for ggplot2.
endlocal
