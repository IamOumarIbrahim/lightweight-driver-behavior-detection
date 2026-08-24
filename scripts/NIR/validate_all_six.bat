@echo off
setlocal
cd /d "%~dp0\..\.."
set /p CONFIRM=Type VALIDATE_ALL_SIX to run validation:
if /I not "%CONFIRM%"=="VALIDATE_ALL_SIX" echo Cancelled.& exit /b 3
.venv\Scripts\python.exe -m tools.workflow.evaluate_all_nir validate --execute
