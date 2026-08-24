@echo off
setlocal
cd /d "%~dp0\..\.."
python -m tools.publication.figures
if errorlevel 1 (
  echo.
  echo Figure generation failed. Review the error above.
  exit /b 1
)
echo.
echo Publication figures are ready under results\RGB\summary\figures.
endlocal
