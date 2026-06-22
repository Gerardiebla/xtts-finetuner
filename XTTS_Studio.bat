@echo off
setlocal enabledelayedexpansion

REM XTTS Studio Launcher
REM Double-click this file to start the app

cd /d "%~dp0"

REM Check if venv exists
if not exist "venv_tts\Scripts\python.exe" (
    echo ERROR: venv_tts not found.
    echo Please ensure you've run setup.ps1 first to install dependencies.
    echo.
    pause
    exit /b 1
)

REM Set environment variable for Coqui
set COQUI_TOS_AGREED=1

REM Launch the GUI
"%~dp0venv_tts\Scripts\python.exe" "%~dp0audio_prep_gui.py"
