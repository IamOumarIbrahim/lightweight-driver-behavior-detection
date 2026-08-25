@echo off
setlocal
cd /d "%~dp0\..\.."
set "R_SCRIPT=%CD%\third_party\R-4.6.1\bin\Rscript.exe"
set "R_LIBS_USER=%CD%\third_party\R-library-4.6"
set "LC_ALL=English_United States.utf8"
if not exist "%R_SCRIPT%" (
  echo Project-local R is missing. Running the pinned figure setup first.
  call scripts\setup\05_setup_r_figures.bat
  if errorlevel 1 exit /b 1
)
"%R_SCRIPT%" --vanilla tools\publication\figures.R
if errorlevel 1 (
  echo.
  echo Figure generation failed. Review the error above.
  exit /b 1
)
echo.
echo Publication figures are ready under results\summary and results\{RGB,NIR}\summary.
endlocal
