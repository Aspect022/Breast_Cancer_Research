@echo off
REM ═══════════════════════════════════════════════════════════════════
REM Breast Cancer Classification - One-Click Run Script (Windows)
REM ═══════════════════════════════════════════════════════════════════
REM 
REM This script runs the complete training pipeline with automatic
REM environment and dataset checks.
REM
REM Features:
REM - Auto-creates venv if missing
REM - Auto-downloads dataset if missing
REM - Activates environment automatically
REM - Runs full pipeline with all models
REM - Logs all output to logs\training_log.txt
REM - Shows estimated time remaining
REM - Handles errors gracefully
REM
REM Usage:
REM   run_all.bat
REM   run_all.bat --models efficientnet spiking
REM   run_all.bat --config my_config.yaml
REM   run_all.bat --quick  REM Quick test with subset
REM ═══════════════════════════════════════════════════════════════════

setlocal EnableDelayedExpansion

REM ═══════════════════════════════════════════════════════════════════
REM Configuration
REM ═══════════════════════════════════════════════════════════════════

set "PROJECT_ROOT=%~dp0"
set "VENV_DIR=%PROJECT_ROOT%.venv"
set "DATA_DIR=%DATA_DIR:%PROJECT_ROOT%data%"
set "LOG_DIR=%PROJECT_ROOT%logs"
set "CONFIG_FILE=%PROJECT_ROOT%config.yaml"
set "PIPELINE_SCRIPT=%PROJECT_ROOT%run_pipeline.py"

REM Generate timestamp for log file
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value') do set "dt=%%i"
set "TIMESTAMP=%dt:~0,4%%dt:~4,2%%dt:~6,2%_%dt:~8,2%%dt:~10,2%%dt:~12,2%"
set "LOG_FILE=%LOG_DIR%\training_log_%TIMESTAMP%.txt"

REM Pipeline arguments
set "PIPELINE_ARGS="
set "QUICK_MODE=false"

REM ═══════════════════════════════════════════════════════════════════
REM Utility Functions
REM ═══════════════════════════════════════════════════════════════════

:log_info
echo [INFO] %~1
echo [INFO] %~1 >> "%LOG_FILE%"
goto :eof

:log_success
echo [OK] %~1
echo [OK] %~1 >> "%LOG_FILE%"
goto :eof

:log_warning
echo [WARN] %~1
echo [WARN] %~1 >> "%LOG_FILE%"
goto :eof

:log_error
echo [ERROR] %~1
echo [ERROR] %~1 >> "%LOG_FILE%"
goto :eof

:log_step
echo.
echo ═════════════════════════════════════════════════════════
echo %~1
echo ═════════════════════════════════════════════════════════
echo.
echo ═════════════════════════════════════════════════════════ >> "%LOG_FILE%"
echo %~1 >> "%LOG_FILE%"
echo ═════════════════════════════════════════════════════════ >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"
goto :eof

REM ═══════════════════════════════════════════════════════════════════
REM Setup Check Functions
REM ═══════════════════════════════════════════════════════════════════

:check_venv
call :log_step "Checking Virtual Environment"

if not exist "%VENV_DIR%" (
    call :log_warning "Virtual environment not found at %VENV_DIR%"
    call :log_info "Running auto_setup.bat to create environment..."
    
    if exist "%PROJECT_ROOT%scripts\auto_setup.bat" (
        call "%PROJECT_ROOT%scripts\auto_setup.bat" --skip-dataset
    ) else (
        call :log_error "auto_setup.bat not found. Please run setup first."
        exit /b 1
    )
) else (
    call :log_success "Virtual environment found"
)

goto :eof

:activate_venv
call :log_info "Activating virtual environment..."

call "%VENV_DIR%\Scripts\activate.bat"

if %errorlevel% equ 0 (
    call :log_success "Virtual environment activated"
) else (
    call :log_error "Failed to activate virtual environment"
    exit /b 1
)

goto :eof

:check_dataset
call :log_step "Checking Dataset"

set "DATASET_PATH=%DATA_DIR%\BreaKHis_v1"

if not exist "%DATASET_PATH%" (
    call :log_warning "Dataset not found at %DATASET_PATH%"
    call :log_info "Downloading dataset..."
    
    if exist "%PROJECT_ROOT%scripts\download_dataset.py" (
        python "%PROJECT_ROOT%scripts\download_dataset.py" --output_dir "%DATA_DIR%" --resume
    ) else (
        call :log_error "download_dataset.py not found."
        call :log_info "Please download BreakHis dataset manually from:"
        call :log_info "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/"
        exit /b 1
    )
) else (
    call :log_success "Dataset found at %DATASET_PATH%"
    
    REM Quick validation - count PNG files
    for /f %%i in ('dir /b /s "%DATASET_PATH%\*.png" 2^>nul ^| find /c ".png"') do set "IMAGE_COUNT=%%i"
    if defined IMAGE_COUNT (
        call :log_success "Found !IMAGE_COUNT! images"
    ) else (
        call :log_warning "Could not count images"
    )
)

goto :eof

:check_gpu
call :log_step "Checking GPU Status"

REM Check nvidia-smi
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    call :log_success "NVIDIA GPU detected"
    echo.
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    echo.
) else (
    call :log_warning "No NVIDIA GPU detected (running on CPU)"
)

REM Check PyTorch CUDA
call :log_info "Checking PyTorch CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>&1
if %errorlevel% equ 0 (
    call :log_success "PyTorch CUDA support enabled"
) else (
    call :log_warning "PyTorch CUDA not available or PyTorch not installed"
)

goto :eof

:check_config
call :log_step "Checking Configuration"

if not exist "%CONFIG_FILE%" (
    call :log_error "Config file not found: %CONFIG_FILE%"
    exit /b 1
)

call :log_success "Config file: %CONFIG_FILE%"

REM Show key settings
call :log_info "Key settings:"
python -c "import yaml; cfg=yaml.safe_load(open('%CONFIG_FILE%')); print(f\"  Task: {cfg['data']['task']}\"); print(f\"  Folds: {cfg['cv']['n_folds']}\"); print(f\"  Epochs: {cfg['training']['epochs']}\"); models=[k for k,v in cfg['models'].items() if v.get('enabled',False)]; print(f\"  Models: {', '.join(models)}\")" 2>&1

goto :eof

REM ═══════════════════════════════════════════════════════════════════
REM Time Estimation
REM ═══════════════════════════════════════════════════════════════════

:estimate_time
call :log_step "Time Estimation"

python -c "
import yaml

with open('%CONFIG_FILE%') as f:
    cfg = yaml.safe_load(f)

epochs = cfg['training']['epochs']
folds = cfg['cv']['n_folds']
models = [k for k, v in cfg['models'].items() if v.get('enabled', False)]

# Rough estimates per fold (based on typical A100 performance)
time_per_fold = {
    'efficientnet': 300,    # ~5 min per fold
    'spiking': 400,         # ~6.5 min per fold
    'hybrid_vit': 350,      # ~6 min per fold
    'vit_tiny': 250,        # ~4 min per fold
    'quantum': 600,         # ~10 min per fold (CPU)
}

total_seconds = 0
print('Estimated time per model (all folds):')
for model in models:
    est = time_per_fold.get(model, 300) * folds
    total_seconds += est
    mins = est // 60
    print(f'  {model}: ~{mins} minutes')

total_mins = total_seconds // 60
total_hours = total_mins // 60
remaining_mins = total_mins %% 60

print()
if total_hours > 0:
    print(f'TOTAL ESTIMATE: ~{total_hours}h {remaining_mins}m')
else:
    print(f'TOTAL ESTIMATE: ~{total_mins} minutes')
print()
print('Note: Actual times vary based on GPU, dataset size, and settings.')
" 2>&1

goto :eof

REM ═══════════════════════════════════════════════════════════════════
REM Pipeline Execution
REM ═══════════════════════════════════════════════════════════════════

:run_pipeline
call :log_step "Starting Training Pipeline"

echo ═════════════════════════════════════════════════════════
echo   TRAINING STARTED: %DATE% %TIME%
echo ═════════════════════════════════════════════════════════

REM Record start time
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value') do set "dt=%%i"
set "START_TIME=%date% %time%"

REM Run pipeline
if "%QUICK_MODE%"=="true" (
    call :log_info "Running in QUICK mode (subset=200, epochs=3)"
    python "%PIPELINE_SCRIPT%" --config "%CONFIG_FILE%" %PIPELINE_ARGS%
) else (
    python "%PIPELINE_SCRIPT%" --config "%CONFIG_FILE%" %PIPELINE_ARGS%
)

set "PIPELINE_EXIT_CODE=%errorlevel%"

REM Record end time
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value') do set "dt=%%i"
set "END_TIME=%date% %time%"

echo.
echo ═════════════════════════════════════════════════════════
echo   TRAINING COMPLETED: %END_TIME%
echo ═════════════════════════════════════════════════════════

if %PIPELINE_EXIT_CODE% equ 0 (
    call :log_success "Pipeline completed successfully!"
) else (
    call :log_error "Pipeline failed with exit code: %PIPELINE_EXIT_CODE%"
    call :log_info "Check log file for details: %LOG_FILE%"
)

goto :eof

:show_results
call :log_step "Results Summary"

REM Find latest comparison CSV
for /f "delims=" %%i in ('dir /b /o-d "%PROJECT_ROOT%outputs\comparison_*.csv" 2^>nul') do (
    set "COMPARISON_FILE=%PROJECT_ROOT%outputs\%%i"
    goto :found_comparison
)

:found_comparison
if defined COMPARISON_FILE (
    if exist "%COMPARISON_FILE%" (
        call :log_success "Latest results: %COMPARISON_FILE%"
        echo.
        echo Results preview:
        python -c "import pandas as pd; df=pd.read_csv('%COMPARISON_FILE%'); cols=[c for c in df.columns if any(x in c for x in ['Model','Mean_Acc','Mean_F1','Mean_AUC','Mean_MCC'])]; print(df[cols].to_string(index=False)) if cols else print('No key columns found')" 2>&1
    )
) else (
    call :log_warning "No comparison results found yet"
)

echo.
call :log_info "Full log saved to: %LOG_FILE%"
call :log_info "Outputs directory: %PROJECT_ROOT%outputs"

goto :eof

REM ═══════════════════════════════════════════════════════════════════
REM Parse Arguments
REM ═══════════════════════════════════════════════════════════════════

:parse_args
if "%~1"=="" goto :main

if /i "%~1"=="--models" (
    set "PIPELINE_ARGS=%PIPELINE_ARGS% --models %~2"
    shift
    shift
    goto :parse_args
)

if /i "%~1"=="--config" (
    set "CONFIG_FILE=%~2"
    shift
    shift
    goto :parse_args
)

if /i "%~1"=="--quick" (
    set "QUICK_MODE=true"
    shift
    goto :parse_args
)

if /i "%~1"=="--help" (
    echo Usage: %~nx0 [OPTIONS]
    echo.
    echo Options:
    echo   --models M1 M2    Run specific models only
    echo   --config FILE     Use custom config file
    echo   --quick           Quick test mode (subset, fewer epochs)
    echo   --help            Show this help message
    exit /b 0
)

call :log_error "Unknown option: %~1"
exit /b 1

REM ═══════════════════════════════════════════════════════════════════
REM Main
REM ═══════════════════════════════════════════════════════════════════

:main
cd /d "%PROJECT_ROOT%"

REM Create log directory
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Initialize log file
echo ═════════════════════════════════════════════════════════ > "%LOG_FILE%"
echo   BREAST CANCER CLASSIFICATION - TRAINING LOG >> "%LOG_FILE%"
echo   Started: %DATE% %TIME% >> "%LOG_FILE%"
echo ═════════════════════════════════════════════════════════ >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

call :parse_args %*

call :log_step "Breast Cancer Classification - One-Click Runner"
call :log_info "Project: %PROJECT_ROOT%"
call :log_info "Log file: %LOG_FILE%"

REM Pre-flight checks
call :check_venv
call :activate_venv
call :check_dataset
call :check_gpu
call :check_config
call :estimate_time

REM Confirm before starting
echo.
set /p CONFIRM="Start training? (y/N): "
if /i not "!CONFIRM!"=="y" (
    call :log_info "Training cancelled by user"
    exit /b 0
)

REM Run pipeline
call :run_pipeline
set "RESULT=%errorlevel%"

REM Show results
call :show_results

exit /b %RESULT%
