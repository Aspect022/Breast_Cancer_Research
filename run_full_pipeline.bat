@echo off
REM ═══════════════════════════════════════════════════════════════════
REM BREAST CANCER CLASSIFICATION - FULLY AUTOMATED PIPELINE
REM ═══════════════════════════════════════════════════════════════════
REM This script will:
REM   1. Create/activate virtual environment
REM   2. Install all requirements (A100 GPU optimized)
REM   3. Download datasets (BreakHis, WBCD, SEER)
REM   4. Run complete pipeline with all enabled models
REM   5. Generate results summary and statistical analysis
REM
REM Usage: run_full_pipeline.bat [--quick] [--dataset DATASET]
REM   --quick    : Run with subset=200 and epochs=3 for testing
REM   --dataset  : Specify dataset (breakhis/wbcd/seer, default: breakhis)
REM ═══════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

echo ╔══════════════════════════════════════════════════════════════════╗
echo ║   BREAST CANCER CLASSIFICATION - AUTOMATED PIPELINE             ║
echo ║   Transformers + Quantum-Enhanced Neural Networks               ║
echo ╚══════════════════════════════════════════════════════════════════╝
echo.

REM ── Parse arguments ────────────────────────────────────────────────
set QUICK_MODE=0
set DATASET=breakhis

:parse_args
if "%~1"=="" goto :end_parse_args
if /i "%~1"=="--quick" (
    set QUICK_MODE=1
    shift
    goto :parse_args
)
if /i "%~1"=="--dataset" (
    set DATASET=%~2
    shift
    shift
    goto :parse_args
)
shift
goto :parse_args
:end_parse_args

echo Configuration:
echo   Dataset: %DATASET%
if %QUICK_MODE%==1 (
    echo   Mode: QUICK TEST (subset=200, epochs=3)
) else (
    echo   Mode: FULL TRAINING
)
echo.

REM ── Step 1: Check Python ───────────────────────────────────────────
echo [1/7] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9+
    pause
    exit /b 1
)
echo ✓ Python found
echo.

REM ── Step 2: Create virtual environment ─────────────────────────────
echo [2/7] Setting up virtual environment...
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
) else (
    echo ✓ Virtual environment already exists
)
echo.

REM ── Step 3: Activate venv ──────────────────────────────────────────
echo [3/7] Activating virtual environment...
call .venv\Scripts\activate.bat
echo ✓ Activated
echo.

REM ── Step 4: Install requirements ───────────────────────────────────
echo [4/7] Installing requirements (A100 GPU optimized)...
echo This may take 5-10 minutes on first run...
echo.

echo Installing PyTorch with CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q

echo Installing timm and other core packages...
pip install timm>=1.0.0 scipy>=1.11.0 opencv-python>=4.8.0 -q

echo Installing remaining requirements...
pip install -r requirements.txt -q

echo ✓ All packages installed
echo.

REM ── Step 5: Check GPU ──────────────────────────────────────────────
echo [5/7] Checking GPU availability...
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo.

REM ── Step 6: Download datasets ──────────────────────────────────────
echo [6/7] Checking/downloading datasets...
python scripts\download_datasets.py --dataset %DATASET%
echo.

REM ── Step 7: Configure for quick or full run ────────────────────────
if %QUICK_MODE%==1 (
    echo [7/7] Preparing quick test configuration...
    
    REM Create temporary quick config
    (
        echo data:
        echo   dataset: "%DATASET%"
        echo   data_dir: "data/%DATASET%"
        echo   task: "binary"
        echo   subset: 200
        echo   num_workers: 4
        echo   seed: 42
        echo.
        echo cv:
        echo   n_folds: 2
        echo.
        echo training:
        echo   epochs: 3
        echo   patience: 2
        echo   grad_clip: 1.0
        echo.
        echo models:
        echo   efficientnet:
        echo     enabled: true
        echo     batch_size: 32
        echo     lr: 1.0e-4
        echo   swin_tiny:
        echo     enabled: true
        echo     batch_size: 32
        echo     lr: 2.0e-5
        echo   convnext_tiny:
        echo     enabled: true
        echo     batch_size: 32
        echo     lr: 2.0e-5
        echo   dual_branch_fusion:
        echo     enabled: true
        echo     batch_size: 16
        echo     lr: 2.0e-5
        echo   qenn_ry:
        echo     enabled: true
        echo     batch_size: 16
        echo     n_qubits: 4
        echo     n_layers: 1
        echo.
        echo output:
        echo   output_dir: "outputs_quick_test"
    ) > config_quick.yaml
    
    set CONFIG_FILE=config_quick.yaml
    echo ✓ Quick test config created
) else (
    set CONFIG_FILE=config.yaml
    echo ✓ Using full configuration
)
echo.

REM ── Run pipeline ───────────────────────────────────────────────────
echo ═══════════════════════════════════════════════════════════════════
echo Starting full pipeline execution...
echo Configuration: %CONFIG_FILE%
echo Dataset: %DATASET%
echo ═══════════════════════════════════════════════════════════════════
echo.

python run_pipeline.py --config %CONFIG_FILE% --dataset %DATASET%

if errorlevel 1 (
    echo.
    echo ╔══════════════════════════════════════════════════════════════════╗
    echo ║   PIPELINE EXECUTION FAILED                                     ║
    echo ╚══════════════════════════════════════════════════════════════════╝
    echo Check error messages above for details.
    pause
    exit /b 1
)

REM ── Cleanup quick config ────────────────────────────────────────────
if %QUICK_MODE%==1 (
    if exist config_quick.yaml del config_quick.yaml
)

REM ── Summary ────────────────────────────────────────────────────────
echo.
echo ╔══════════════════════════════════════════════════════════════════╗
echo ║   PIPELINE COMPLETE!                                            ║
echo ╠══════════════════════════════════════════════════════════════════╣
echo ║  Results saved to: outputs\                                      ║
if %QUICK_MODE%==1 (
    echo ║  Quick test results: outputs_quick_test\comparison_binary.csv    ║
) else (
    echo ║  Full results: outputs\comparison_binary.csv                     ║
)
echo ╚══════════════════════════════════════════════════════════════════╝
echo.
echo Next steps:
echo   1. Review results in outputs\comparison_binary.csv
echo   2. Check individual model results in outputs\^<Model^>_<task>\
echo   3. Run statistical analysis: python scripts\analyze_results.py
echo.
pause
