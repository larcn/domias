@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: Domino AI UI (Release / Rust Runtime)
:: - Creates/uses .venv
:: - Installs deps (flask/numpy/zstandard/maturin)
:: - Builds domino_rs via maturin if missing
:: - Forces operational runtime defaults

:: 1) Python detection
where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3.13) else (set PY=python)

:: 2) Venv
set VENV=.venv
if not exist "%VENV%\Scripts\python.exe" (
  echo [INFO] Creating venv...
  %PY% -m venv "%VENV%" || (echo [FAIL] venv create failed & pause & exit /b 1)
)
set VPY=%VENV%\Scripts\python.exe

:: 3) Deps
"%VPY%" -m pip install --upgrade pip >nul 2>nul
"%VPY%" -m pip install flask numpy zstandard maturin >nul 2>nul

:: 4) Runtime env (Release)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

set DOMINO_INFER=0
set DOMINO_AVOID_BETA=0

:: 5) Ensure domino_rs is importable; build if missing
"%VPY%" -c "import domino_rs; print(domino_rs.version())" >nul 2>nul
if %errorlevel% neq 0 (
  echo [INFO] domino_rs not found. Building with maturin...
  where cargo >nul 2>nul
  if %errorlevel% neq 0 (
    echo [FAIL] cargo not found in PATH. Install Rust toolchain first.
    echo        https://www.rust-lang.org/tools/install
    pause
    exit /b 1
  )
  "%VPY%" -m maturin develop --release
  if %errorlevel% neq 0 (
    echo [FAIL] maturin build failed.
    pause
    exit /b 1
  )
)

:: 6) Launch
echo.
echo [INFO] Starting Domino AI UI (Rust default)...
start "" http://127.0.0.1:5000
"%VPY%" app.py

pause