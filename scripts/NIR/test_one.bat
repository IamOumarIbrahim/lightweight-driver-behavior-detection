@echo off
set /p MODEL=Model [yolo11n/yolo26n/rtmdet_tiny/efficientdet_d1/dfine_n]:
set /p RATIO=Ratio [1to2/1to6]:
call "%~dp0\..\_shared\evaluate_one.bat" test NIR %MODEL% %RATIO%
