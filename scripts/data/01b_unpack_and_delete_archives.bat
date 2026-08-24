@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" echo Run scripts\setup\01_create_environment.bat first.& exit /b 2
echo Successfully extracted archives under data\DMD and data\Drive^&Act will be permanently deleted.
set /p CONFIRM=Type DELETE_ARCHIVES to continue:
if /I not "%CONFIRM%"=="DELETE_ARCHIVES" echo Cancelled.& exit /b 3
.venv\Scripts\python.exe -m tools.data.unpack_sources --delete-archives || exit /b 1
