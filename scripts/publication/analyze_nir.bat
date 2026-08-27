@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" (
  echo Project environment is missing. Run scripts\setup\01_create_environment.bat first.
  exit /b 2
)
.venv\Scripts\python.exe -m tools.publication.analyze_nir
if errorlevel 1 (
  echo.
  echo NIR publication aggregation failed. Verify each completed model has both protected-test results.
  exit /b 1
)
echo.
echo NIR publication source is ready. Run build_figures.bat to refresh the figure.
endlocal
