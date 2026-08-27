@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" echo Run 01_create_environment.bat first.& exit /b 2
.venv\Scripts\python.exe -c "from tools.backends.docker_backend import ensure_engine; print('Docker engine:', ensure_engine())" || exit /b 2
.venv\Scripts\python.exe -m tools.setup.backends --install
