@echo off
setlocal
cd /d "%~dp0"

call scripts\setup\01_create_environment.bat || exit /b 1
call scripts\setup\02_setup_backends.bat || exit /b 1
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\NIR\train_all_sixteen.ps1 -Yes
exit /b %ERRORLEVEL%
