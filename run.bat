@echo off
setlocal
cd /d "%~dp0"

call scripts\setup\01_create_environment.bat || exit /b 1
call scripts\setup\02_setup_backends.bat || exit /b 1
call scripts\NIR\train_all_six.bat --yes
exit /b %ERRORLEVEL%
