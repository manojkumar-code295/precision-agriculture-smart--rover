@echo off
rem =========================================================================
rem Plant Disease Detection AI - Windows Desktop Launcher
rem Double-click this file to launch the AI Camera Tester on Windows.
rem =========================================================================

cd /d "%~dp0"
title Plant Disease Detection AI - Camera Tester

echo ===================================================
echo Starting Plant Disease Detection AI Camera Tester...
echo ===================================================

rem Check Python
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    where py >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Python is not found in PATH!
        echo Please install Python 3.10, 3.11 or 3.12 from python.org
        pause
        exit /b 1
    )
    set PY_CMD=py
) else (
    set PY_CMD=python
)

rem Check virtual environment
if exist venv\Scripts\python.exe (
    set PY_CMD=venv\Scripts\python.exe
)

rem Launch Application
start "" %PY_CMD% camera_tester.py
exit /b 0
