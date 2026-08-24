@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" py -3.12 -m venv .venv || exit /b 1
if not exist "third_party\packages" mkdir "third_party\packages"
.venv\Scripts\python.exe -m pip download --dest "third_party\packages" -r requirements.lock.txt || exit /b 1
.venv\Scripts\python.exe -m pip install --no-index --find-links "third_party\packages" -r requirements.lock.txt || exit /b 1
.venv\Scripts\python.exe -m pytest tools\tests -q || exit /b 1
echo Environment ready: .venv
