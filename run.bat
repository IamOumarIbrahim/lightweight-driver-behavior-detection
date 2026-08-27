@echo off
setlocal
cd /d "C:\Dev\repos\Public repos\lightweight-driver-behavior-detection"

call scripts\setup\01_create_environment.bat || exit /b 1
call scripts\setup\02_setup_backends.bat || exit /b 1
call scripts\NIR\train_all_ten.bat --yes
exit /b %ERRORLEVEL%
