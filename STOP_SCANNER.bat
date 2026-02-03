@echo off
title Stop Voice Scanner
echo Stopping all Python voice scanner processes...
tasklist /fi "imagename eq python.exe" /fo csv 2>NUL | find /i "python" >NUL
if "%ERRORLEVEL%"=="0" (
    echo Found running Python processes
    echo Please close the scanner window manually or press Ctrl+C in the scanner window
) else (
    echo No Python processes found
)
pause
