@echo off
setlocal
cd /d "%~dp0\..\.."

set "MIKTEX_BIN=%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64"
where pdflatex.exe >nul 2>nul
if not errorlevel 1 goto :build
if exist "%MIKTEX_BIN%\pdflatex.exe" (
  set "PATH=%MIKTEX_BIN%;%PATH%"
  goto :build
)

echo MiKTeX pdflatex was not found.
echo Install MiKTeX, then rerun this file. Perl and latexmk are not required.
exit /b 1

:build
pushd docs\manuscript
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if errorlevel 1 goto :failed
bibtex main
if errorlevel 1 goto :failed
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if errorlevel 1 goto :failed
pdflatex -interaction=nonstopmode -halt-on-error main.tex
if errorlevel 1 goto :failed
popd

echo.
echo Manuscript ready: docs\manuscript\main.pdf
endlocal
exit /b 0

:failed
popd
echo.
echo Manuscript build failed. Review docs\manuscript\main.log.
endlocal
exit /b 1
