@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" echo Run scripts\setup\01_create_environment.bat first.& exit /b 2
echo This runs all ten NIR jobs sequentially with safe resume and per-run logs.
if /I "%~1"=="--yes" goto RUN
set /p CONFIRM=Type TRAIN_ALL_TEN to continue:
if /I not "%CONFIRM%"=="TRAIN_ALL_TEN" echo Cancelled.& exit /b 3
:RUN
.venv\Scripts\python.exe -m tools.workflow.train_all_nir --execute-training
exit /b %ERRORLEVEL%
