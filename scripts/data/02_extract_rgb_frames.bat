@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" echo Run scripts\setup\01_create_environment.bat first.& exit /b 2
if not exist "data\DMD" echo Put the DMD folder inside data\DMD first.& exit /b 2
.venv\Scripts\python.exe -m tools.data.extract_rgb || exit /b 1
.venv\Scripts\python.exe -m tools.data.prepare_rgb || exit /b 1
