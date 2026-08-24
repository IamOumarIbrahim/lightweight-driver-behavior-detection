@echo off
setlocal
cd /d "%~dp0\..\.."
echo This starts the protected test pass for every validated NIR run.
set /p CONFIRM=Type RUN_ALL_PROTECTED_TESTS to continue:
if /I not "%CONFIRM%"=="RUN_ALL_PROTECTED_TESTS" echo Cancelled.& exit /b 3
.venv\Scripts\python.exe -m tools.workflow.evaluate_all_nir test --execute
