@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" echo Run scripts\setup\01_create_environment.bat first.& exit /b 2
echo This starts the single protected test pass for every validated NIR run.
set /p CONFIRM=Type RUN_ALL_PROTECTED_TESTS to continue:
if /I not "%CONFIRM%"=="RUN_ALL_PROTECTED_TESTS" echo Cancelled.& exit /b 3
.venv\Scripts\python.exe -m tools.workflow.evaluate_all_nir test --execute
exit /b %ERRORLEVEL%
