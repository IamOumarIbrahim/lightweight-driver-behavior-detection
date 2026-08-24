@echo off
setlocal
cd /d "%~dp0\..\.."
if not exist ".venv-labelstudio\Scripts\label-studio.exe" echo Run 03_setup_label_studio.bat first.& exit /b 2
set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=%CD%\data\label_studio
.venv-labelstudio\Scripts\label-studio.exe start data\label_studio
