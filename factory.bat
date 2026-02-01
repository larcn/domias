:: FILE: factory.bat | version: 2026-01-17.im1_noblocks
:: Windows helper for Domino AI
::
:: Commands:
::   setup      : create venv + install deps + build domino_rs + smoke import
::   build      : build domino_rs + smoke import
::   test       : cargo test + optional pytest (if tests\ exists)
::   gates      : tools\domino_tool.py check
::   check      : import domino_rs + gates
::   gen        : factory_min.py generate (pass-through args)
::   cycle      : factory_min.py cycle (pass-through args)
::   pilot_inf  : generate preset small (3000) + tools\infer_tool.py pilot
::   help       : show help
::
:: Notes:
:: - After ANY Rust src\*.rs change you MUST run: factory.bat build (or setup)
:: - Rust reads DOMINO_* env vars via OnceLock; each run of this .bat is a new process

@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "VENV=.venv"
set "VPY=%VENV%\Scripts\python.exe"

set "CMD=%~1"
if "%CMD%"=="" set "CMD=help"
if not "%CMD%"=="" shift

echo.
echo === Domino AI - Factory : %CMD% ===
echo.

:: Default runtime env (safe per-process)
if "%DOMINO_AVOID_BETA%"=="" set "DOMINO_AVOID_BETA=0"
if "%PYTHONUNBUFFERED%"=="" set "PYTHONUNBUFFERED=1"
if "%PYTHONUTF8%"=="" set "PYTHONUTF8=1"
if "%PYTHONIOENCODING%"=="" set "PYTHONIOENCODING=utf-8"

:: Ensure venv
if exist "%VPY%" goto :VENV_OK
echo [INFO] Creating venv...
where py >nul 2>nul
if errorlevel 1 goto :VENV_PY
py -3.13 -m venv "%VENV%" 2>nul
if errorlevel 1 py -3 -m venv "%VENV%"
goto :VENV_OK

:VENV_PY
python -m venv "%VENV%"

:VENV_OK
if exist "%VPY%" goto :PIP_DEPS
echo [ERROR] venv creation failed. Expected: %VPY%
goto :END_FAIL

:PIP_DEPS
echo [INFO] Python: %VPY%
echo [INFO] Installing python deps...
"%VPY%" -m pip install --upgrade pip >nul 2>nul
"%VPY%" -m pip install numpy maturin zstandard pytest >nul 2>nul
if errorlevel 1 goto :PIP_FAIL

:: Route command
if /I "%CMD%"=="help" goto :DO_HELP
if /I "%CMD%"=="setup" goto :DO_SETUP
if /I "%CMD%"=="build" goto :DO_BUILD
if /I "%CMD%"=="test" goto :DO_TEST
if /I "%CMD%"=="gates" goto :DO_GATES
if /I "%CMD%"=="check" goto :DO_CHECK
if /I "%CMD%"=="gen" goto :DO_GEN
if /I "%CMD%"=="cycle" goto :DO_CYCLE
if /I "%CMD%"=="pilot_inf" goto :DO_PILOT_INF

echo [ERROR] Unknown command: %CMD%
echo.
goto :DO_HELP

:PIP_FAIL
echo [ERROR] pip install failed.
echo Run manually:
echo   "%VPY%" -m pip install numpy maturin zstandard pytest
goto :END_FAIL


:: =============================================================================
:: Commands
:: =============================================================================

:DO_SETUP
call :VCVARS
if errorlevel 1 goto :END_FAIL
call :BUILD_EXT
if errorlevel 1 goto :END_FAIL
goto :END_OK

:DO_BUILD
call :VCVARS
if errorlevel 1 goto :END_FAIL
call :BUILD_EXT
if errorlevel 1 goto :END_FAIL
goto :END_OK

:DO_TEST
call :VCVARS
if errorlevel 1 goto :END_FAIL

echo [INFO] cargo test...
where cargo >nul 2>nul
if errorlevel 1 goto :NO_CARGO
cargo test
if errorlevel 1 goto :END_FAIL

if exist "tests\" goto :DO_PYTEST
echo [INFO] pytest skipped (no tests\ directory).
goto :END_OK

:DO_PYTEST
echo [INFO] pytest...
"%VPY%" -m pytest -q
if errorlevel 1 goto :END_FAIL
goto :END_OK

:NO_CARGO
echo [ERROR] cargo not found. Install Rust toolchain and reopen terminal.
goto :END_FAIL

:DO_GATES
if exist "tools\domino_tool.py" goto :RUN_GATES
echo [ERROR] tools\domino_tool.py not found.
echo Expected: %cd%\tools\domino_tool.py
goto :END_FAIL

:RUN_GATES
echo [INFO] Running gates: tools\domino_tool.py check
"%VPY%" tools\domino_tool.py check
if errorlevel 1 goto :END_FAIL
goto :END_OK

:DO_CHECK
echo [INFO] Checking: import domino_rs...
"%VPY%" -c "import domino_rs; print('domino_rs import OK'); print(domino_rs.version())"
if errorlevel 1 goto :NEED_SETUP
goto :DO_GATES

:NEED_SETUP
echo [ERROR] domino_rs not installed. Run: factory.bat setup
goto :END_FAIL

:DO_GEN
if exist "factory_min.py" goto :RUN_GEN
echo [ERROR] factory_min.py not found.
goto :END_FAIL

:RUN_GEN
echo [INFO] Running: factory_min.py generate %*
echo [INFO] NOTE: DOMINO_AVOID_BETA=%DOMINO_AVOID_BETA%
"%VPY%" factory_min.py generate %*
if errorlevel 1 goto :END_FAIL
goto :END_OK

:DO_CYCLE
if exist "factory_min.py" goto :RUN_CYCLE
echo [ERROR] factory_min.py not found.
goto :END_FAIL

:RUN_CYCLE
echo [INFO] Running: factory_min.py cycle %*
echo [INFO] NOTE: DOMINO_AVOID_BETA=%DOMINO_AVOID_BETA%
"%VPY%" factory_min.py cycle %*
if errorlevel 1 goto :END_FAIL
goto :END_OK

:DO_PILOT_INF
set "OUTDIR=runs_inf_pilot"
set "SEED=99999"
if not "%~1"=="" set "OUTDIR=%~1"
if not "%~2"=="" set "SEED=%~2"

if exist "factory_min.py" goto :PILOT_HAVE_FACTORY
echo [ERROR] factory_min.py not found.
goto :END_FAIL

:PILOT_HAVE_FACTORY
if exist "tools\infer_tool.py" goto :PILOT_RUN
echo [ERROR] tools\infer_tool.py not found.
echo Expected: %cd%\tools\infer_tool.py
goto :END_FAIL

:PILOT_RUN
echo [INFO] Pilot INF1:
echo   out_dir=%OUTDIR%
echo   seed=%SEED%
echo   DOMINO_AVOID_BETA=%DOMINO_AVOID_BETA%

echo [INFO] 1) Generate (preset small = 3000 matches)...
"%VPY%" factory_min.py generate --preset small --out "%OUTDIR%" --seed %SEED%
if errorlevel 1 goto :END_FAIL

set "MANIFEST="
for /f "usebackq delims=" %%F in (`dir /b /a:-d /o:-n "%OUTDIR%\run_*.manifest.json" 2^>nul`) do (
  set "MANIFEST=%OUTDIR%\%%F"
  goto :PILOT_HAVE_MANIFEST
)

:PILOT_HAVE_MANIFEST
if not "%MANIFEST%"=="" goto :PILOT_TRAIN
echo [ERROR] Could not locate manifest in %OUTDIR%\run_*.manifest.json
goto :END_FAIL

:PILOT_TRAIN
echo [INFO] Using manifest: %MANIFEST%
echo [INFO] 2) INF1 pilot (check + train)...
"%VPY%" tools\infer_tool.py pilot --manifest "%MANIFEST%" --sample_limit 500 --steps 400 --batch 256 --hidden 256 --lr 0.001 --l2 0.0001 --val_samples 2000 --out inference_model.json
if errorlevel 1 goto :END_FAIL

echo [OK] Pilot complete. Output: inference_model.json
goto :END_OK


:DO_HELP
echo Usage:
echo   factory.bat setup
echo   factory.bat build
echo   factory.bat test
echo   factory.bat check
echo   factory.bat gates
echo   factory.bat gen   [factory_min generate args...]
echo   factory.bat cycle [factory_min cycle args...]
echo   factory.bat pilot_inf [out_dir] [seed]
echo.
echo Examples:
echo   factory.bat setup
echo   factory.bat test
echo   factory.bat gates
echo   factory.bat gen --preset small --out runs_smoke --seed 99999
echo   factory.bat cycle --preset small --n 1 --eval_matches 200 --eval_jobs 1
echo   factory.bat pilot_inf runs_inf_pilot 99999
echo.
goto :END_OK


:: =============================================================================
:: Helpers
:: =============================================================================

:VCVARS
echo [INFO] Checking MSVC toolchain (link.exe)...
where link.exe >nul 2>nul
if not errorlevel 1 goto :VCVARS_OK

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" goto :VCVARS_FIND
echo [ERROR] vswhere.exe not found. Install Build Tools for Visual Studio 2022.
exit /b 1

:VCVARS_FIND
set "VSROOT="
for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSROOT=%%I"
if not "%VSROOT%"=="" goto :VCVARS_CALL
echo [ERROR] MSVC Build Tools not found.
exit /b 1

:VCVARS_CALL
set "VCVARS64=%VSROOT%\VC\Auxiliary\Build\vcvars64.bat"
if exist "%VCVARS64%" goto :VCVARS_CALL2
echo [ERROR] vcvars64.bat not found.
echo Path: %VCVARS64%
exit /b 1

:VCVARS_CALL2
call "%VCVARS64%" >nul
where link.exe >nul 2>nul
if errorlevel 1 (
  echo [ERROR] link.exe still not found after vcvars64.
  exit /b 1
)

:VCVARS_OK
echo [OK] MSVC environment ready.
exit /b 0

:BUILD_EXT
echo [INFO] Building Rust extension (maturin develop --release)...
if exist "Cargo.toml" goto :BUILD_HAVE_TOML
echo [ERROR] Cargo.toml not found in: %cd%
exit /b 1

:BUILD_HAVE_TOML
where cargo >nul 2>nul
if errorlevel 1 goto :NO_CARGO2

"%VPY%" -m maturin develop --release --manifest-path "%cd%\Cargo.toml"
if errorlevel 1 goto :MATURIN_FAIL

echo [OK] Build complete.
echo [INFO] Smoke import...
"%VPY%" -c "import domino_rs; print('domino_rs import OK'); print(domino_rs.version())"
if errorlevel 1 goto :IMPORT_FAIL
exit /b 0

:NO_CARGO2
echo [ERROR] cargo not found. Install Rust toolchain and reopen terminal.
exit /b 1

:MATURIN_FAIL
echo [ERROR] maturin develop failed.
exit /b 1

:IMPORT_FAIL
echo [ERROR] import domino_rs failed after build.
exit /b 1


:: =============================================================================
:: End
:: =============================================================================

:END_OK
echo.
echo === DONE (OK) ===
echo.
pause
exit /b 0

:END_FAIL
echo.
echo === DONE (FAILED) ===
echo.
pause
exit /b 1