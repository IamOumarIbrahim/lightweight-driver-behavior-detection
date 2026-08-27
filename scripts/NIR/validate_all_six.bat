@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" echo Run scripts\setup\01_create_environment.bat first.& exit /b 2
if /I "%~1"=="--yes" goto RUN
set /p CONFIRM=Type VALIDATE_ALL_SIX to run validation:
if /I not "%CONFIRM%"=="VALIDATE_ALL_SIX" echo Cancelled.& exit /b 3
:RUN
.venv\Scripts\python.exe -m tools.workflow.evaluate_all_nir validate --execute
exit /b %ERRORLEVEL%
