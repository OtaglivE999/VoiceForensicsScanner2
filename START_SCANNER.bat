@echo off
title Voice Forensics Scanner
color 0A

echo.
echo ===============================================================
echo   VOICE FORENSICS SCANNER
echo   Started: %date% %time%
echo ===============================================================
echo.

cd /d "%~dp0"

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [!] No virtual environment found, using system Python
    echo     Recommend: python -m venv venv
)

echo.
echo Verifying required modules...
echo.
python verify_install.py
if errorlevel 1 (
    echo.
    echo [!] Missing required modules detected.
    set /p INSTALL_CHOICE="    Install missing modules now? (y/n): "
    if /i "%INSTALL_CHOICE%"=="y" (
        echo.
        python verify_install.py --install
        if errorlevel 1 (
            echo.
            echo [ERROR] Module installation failed. Please install manually:
            echo         pip install -r requirements.txt
            pause
            goto :end
        )
        echo.
    ) else (
        echo.
        echo Please install dependencies manually:
        echo     pip install -r requirements.txt
        pause
        goto :end
    )
)

echo.
echo Listing available audio devices...
echo.
python src\soft_voice_scanner.py --list-devices

echo.
echo ===============================================================
echo   Enter device number to start scanning (or 'q' to quit):
echo ===============================================================
set /p DEVICE_ID="Device ID: "

if /i "%DEVICE_ID%"=="q" goto :end

echo.
echo Starting scanner on device %DEVICE_ID%...
echo Press Ctrl+C to stop
echo.

python src\soft_voice_scanner.py --device %DEVICE_ID% --ultra-sensitive

:end
echo.
echo Scanner stopped.
pause
