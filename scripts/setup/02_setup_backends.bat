@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" echo Run 01_create_environment.bat first.& exit /b 2
docker version >nul 2>&1
if errorlevel 1 echo Start Docker Desktop, wait until its engine is ready, then rerun this setup.& exit /b 2
.venv\Scripts\python.exe -m tools.setup.backends --install
