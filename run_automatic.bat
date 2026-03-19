@echo off
REM ═══════════════════════════════════════════════════════════════════
REM Automatic Run Script - Full Pipeline Execution
REM ═══════════════════════════════════════════════════════════════════
REM This script will:
REM   1. Activate virtual environment
REM   2. Verify datasets are present
REM   3. Run complete pipeline with all models
REM ═══════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

echo ╔══════════════════════════════════════════════════════════════════╗
echo ║   BREAST CANCER CLASSIFICATION - AUTOMATIC RUN                  ║
echo ╚══════════════════════════════════════════════════════════════════╝
echo.

REM ── Activate virtual environment ───────────────────────────────────
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo ✓ Virtual environment activated
echo.

REM ── Check GPU availability ─────────────────────────────────────────
echo Checking GPU availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo.

REM ── Verify datasets ────────────────────────────────────────────────
echo Checking datasets...
if not exist "data\BreaKHis_v1" (
    echo WARNING: BreakHis dataset not found
    echo Downloading BreakHis dataset...
    python scripts\download_datasets.py --dataset breakhis
) else (
    echo ✓ BreakHis dataset found
)

if not exist "data\WBCD" (
    echo INFO: WBCD dataset not found - will download if enabled in config
) else (
    echo ✓ WBCD dataset found
)

if not exist "data\SEER" (
    echo INFO: SEER dataset not found - will download if enabled in config
) else (
    echo ✓ SEER dataset found
)
echo.

REM ── Run the full pipeline ──────────────────────────────────────────
echo ═══════════════════════════════════════════════════════════════════
echo Starting full pipeline execution...
echo This may take several hours depending on enabled models
echo ═══════════════════════════════════════════════════════════════════
echo.

python run_pipeline.py --config config.yaml

if errorlevel 1 (
    echo.
    echo ERROR: Pipeline execution failed
    echo Check the error messages above
    pause
    exit /b 1
)

echo.
echo ╔══════════════════════════════════════════════════════════════════╗
echo ║   PIPELINE COMPLETE!                                            ║
echo ╠══════════════════════════════════════════════════════════════════╣
echo ║  Results saved to: outputs\                                      ║
echo ║  Comparison table: outputs\comparison_binary.csv                 ║
echo ╚══════════════════════════════════════════════════════════════════╝
echo.
pause
