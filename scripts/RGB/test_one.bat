@echo off
set /p MODEL=Model [yolo11n/yolo26n/dfine_n]:
set /p SEED=Seed [13/37/73]:
call "%~dp0\..\_shared\evaluate_one.bat" test RGB %MODEL% %SEED%
