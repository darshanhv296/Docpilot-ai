@echo off
setlocal EnableDelayedExpansion

title DocPilot AI / VISION AI - Local Standalone Server
color 0a

echo ==============================================================================
echo        DocPilot AI / VISION AI - Launching Local Application
echo ==============================================================================
echo.

:: 1. Activate Virtual Environment if available
if exist ".venv\Scripts\activate.bat" (
    echo [*] Activating isolated virtual environment (.venv)...
    call .venv\Scripts\activate.bat
) else (
    echo [i] No .venv found. Using system Python...
)

:: 2. Verify Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python not found in system PATH.
    echo Please run setup.bat first to set up your environment.
    echo.
    pause
    exit /b 1
)

:: 3. Launch Application
echo [*] Starting DocPilot AI local server...
echo [*] Browser will open automatically at http://localhost:8000
echo [*] Press Ctrl+C in this console to stop the server at any time.
echo.

python main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application exited with code %errorlevel%.
    echo Check logs\docpilot.log for technical details.
    echo.
    pause
)
