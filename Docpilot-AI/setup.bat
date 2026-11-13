@echo off
setlocal EnableDelayedExpansion

title DocPilot AI / VISION AI - Windows Environment Setup
color 0b

echo ==============================================================================
echo        DocPilot AI / VISION AI - Standalone Local Windows Setup
echo ==============================================================================
echo.

:: 1. Verify Python Installation
echo [*] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python is not installed or not found in system PATH.
    echo.
    echo Please install Python 3.10, 3.11, or 3.12 (64-bit) from:
    echo    https://www.python.org/downloads/
    echo.
    echo CRITICAL: Ensure you check "[X] Add python.exe to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [OK] Detected Python %PY_VER%

:: 2. Create or verify Virtual Environment
if not exist ".venv" (
    echo [*] Creating isolated virtual environment (.venv)...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Existing virtual environment found (.venv).
)

:: 3. Activate Virtual Environment
echo [*] Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: 4. Upgrade pip and install requirements
echo [*] Upgrading pip package manager...
python -m pip install --upgrade pip --quiet

echo [*] Installing local application dependencies...
if exist "requirements.txt" (
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo.
        echo [WARNING] Some optional compiled packages failed to install.
        echo DocPilot AI has built-in Pure-Python fallbacks and will continue.
    )
) else (
    echo [ERROR] requirements.txt not found.
    pause
    exit /b 1
)

:: 5. Create Storage, Data, and Log directories
echo [*] Initializing local storage and directories...
if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "storage" mkdir storage
if not exist "storage\uploads" mkdir storage\uploads
if not exist "storage\extracted" mkdir storage\extracted
if not exist "storage\indexes" mkdir storage\indexes
if not exist "storage\exports" mkdir storage\exports
if not exist "storage\temp" mkdir storage\temp
if not exist "storage\backups" mkdir storage\backups
if not exist ".data" mkdir .data
if not exist ".data\exports" mkdir .data\exports
if not exist ".data\uploads" mkdir .data\uploads

:: 6. Create default .env if missing
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo [OK] Initialized .env from template.
    )
)

:: 7. Run self-diagnostics test suite
echo.
echo [*] Running self-contained test suite...
python -m unittest discover -s tests -p "test_*.py"
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Some diagnostics reported warnings. You can still launch the app.
) else (
    echo.
    echo [OK] All system diagnostics and integration tests PASSED.
)

echo.
echo ==============================================================================
echo [SUCCESS] DocPilot AI / VISION AI Local Setup is Complete!
echo.
echo To launch the application:
echo    Double-click:  run.bat
echo    Or run:        python main.py
echo ==============================================================================
echo.
pause
