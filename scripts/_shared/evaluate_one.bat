@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv\Scripts\python.exe" echo Run scripts\setup\01_create_environment.bat first.& exit /b 2
if /I "%~1"=="validate" set WORD=VALIDATE
if /I "%~1"=="test" set WORD=RUN_PROTECTED_TEST
echo Phase: %~1  Run: %~2 / %~3 / %~4
set /p CONFIRM=Type %WORD% to continue:
if /I not "%CONFIRM%"=="%WORD%" echo Cancelled.& exit /b 3
if /I "%~2"=="RGB" goto RGB
if /I "%~1"=="validate" .venv\Scripts\python.exe -m tools.workflow.evaluate validate --track NIR --model %~3 --ratio %~4 --execute-validation
if /I "%~1"=="test" .venv\Scripts\python.exe -m tools.workflow.evaluate test --track NIR --model %~3 --ratio %~4 --execute-test --confirm RUN_PROTECTED_TEST
exit /b %ERRORLEVEL%
:RGB
if /I "%~1"=="validate" .venv\Scripts\python.exe -m tools.workflow.evaluate validate --track RGB --model %~3 --seed %~4 --execute-validation
if /I "%~1"=="test" .venv\Scripts\python.exe -m tools.workflow.evaluate test --track RGB --model %~3 --seed %~4 --execute-test --confirm RUN_PROTECTED_TEST
