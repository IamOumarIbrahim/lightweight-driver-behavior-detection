@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" echo Run scripts\setup\01_create_environment.bat first.& exit /b 2
echo This will start or safely resume: %~1 / %~2 / %~3
set /p CONFIRM=Type TRAIN to continue:
if /I not "%CONFIRM%"=="TRAIN" echo Cancelled.& exit /b 3
if /I "%~1"=="RGB" .venv\Scripts\python.exe -m tools.workflow.train --track RGB --model %~2 --seed %~3 --execute-training
if /I "%~1"=="NIR" .venv\Scripts\python.exe -m tools.workflow.train --track NIR --model %~2 --ratio %~3 --execute-training
