@echo off
setlocal
cd /d "%~dp0\..\.."
python -m tools.setup.r_figures
if errorlevel 1 (
  echo.
  echo R and ggplot2 setup failed. Review the error above.
  exit /b 1
)
echo.
echo R and ggplot2 are ready for publication figures.
endlocal
