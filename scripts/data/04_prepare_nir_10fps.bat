@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" echo Run scripts\setup\01_create_environment.bat first.& exit /b 2
if not exist "data\Drive&Act" echo Put the Drive^&Act folder inside data\Drive^&Act first.& exit /b 2
.venv\Scripts\python.exe -m tools.data.prepare_nir || exit /b 1
