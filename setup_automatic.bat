@echo off
REM ═══════════════════════════════════════════════════════════════════
REM Automatic Setup Script for Breast Cancer Classification Pipeline
REM ═══════════════════════════════════════════════════════════════════
REM This script will:
REM   1. Create virtual environment
REM   2. Install all requirements
REM   3. Download datasets (BreakHis, WBCD, SEER)
REM   4. Run the full pipeline automatically
REM ═══════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

echo ╔══════════════════════════════════════════════════════════════════╗
echo ║   BREAST CANCER CLASSIFICATION - AUTOMATIC SETUP                ║
echo ╚══════════════════════════════════════════════════════════════════╝
echo.

REM ── Step 1: Check Python installation ──────────────────────────────
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo ✓ Python found
echo.

REM ── Step 2: Create virtual environment ─────────────────────────────
echo [2/6] Setting up virtual environment...
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)
echo.

REM ── Step 3: Activate virtual environment ───────────────────────────
echo [3/6] Activating virtual environment...
call .venv\Scripts\activate.bat
echo ✓ Virtual environment activated
echo.

REM ── Step 4: Install requirements ───────────────────────────────────
echo [4/6] Installing requirements (this may take 5-10 minutes)...
echo Installing PyTorch with CUDA 11.8 support for A100 GPU...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Installing additional requirements...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)
echo ✓ All requirements installed
echo.

REM ── Step 5: Download datasets ──────────────────────────────────────
echo [5/6] Downloading datasets...
python scripts\download_datasets.py
if errorlevel 1 (
    echo WARNING: Dataset download encountered issues
    echo You may need to manually download datasets
    echo See scripts\download_datasets.py for details
)
echo.

REM ── Step 6: Verify setup ───────────────────────────────────────────
echo [6/6] Verifying setup...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo.

echo ╔══════════════════════════════════════════════════════════════════╗
echo ║   SETUP COMPLETE!                                               ║
echo ╠══════════════════════════════════════════════════════════════════╣
echo ║  Next steps:                                                     ║
echo ║  1. Review config.yaml to customize settings                     ║
echo ║  2. Run: run_automatic.bat to start full pipeline                ║
echo ║     or: python run_pipeline.py                                   ║
echo ╚══════════════════════════════════════════════════════════════════╝
echo.
pause
