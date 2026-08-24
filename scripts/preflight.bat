@echo off
setlocal
cd /d "%~dp0\.."
if not exist ".venv\Scripts\python.exe" echo Run scripts\setup\01_create_environment.bat first.& exit /b 2
.venv\Scripts\python.exe -m tools.reproducibility.preflight --track repository
