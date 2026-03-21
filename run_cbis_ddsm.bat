@echo off
REM ════════════════════════════════════════════════════════════════════════════════
REM CBIS-DDSM Generalization Validation Pipeline (Windows)
REM ════════════════════════════════════════════════════════════════════════════════
REM 
REM This script:
REM 1. Auto-downloads CBIS-DDSM dataset from Kaggle
REM 2. Runs all enabled models from config.yaml
REM 3. Compares performance with BreakHis results
REM
REM Usage:
REM   run_cbis_ddsm.bat                    REM Full pipeline
REM   run_cbis_ddsm.bat --quick            REM Quick test with 200 samples
REM   run_cbis_ddsm.bat --models swin_tiny REM Specific models only
REM
REM Prerequisites:
REM   - Kaggle CLI installed: pip install kaggle
REM   - Kaggle credentials set:
REM     set KAGGLE_USERNAME=your_username
REM     set KAGGLE_KEY=your_api_key
REM ════════════════════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

REM ════════════════════════════════════════════════════════════════════════════════
REM Configuration
REM ════════════════════════════════════════════════════════════════════════════════

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Default parameters
set "DATASET=cbis_ddsm"
set "DATA_DIR=data\CBIS-DDSM"
set "CONFIG_FILE=config.yaml"
set "QUICK_MODE=false"
set "MODELS="
set "SUBSET_SIZE="

REM ════════════════════════════════════════════════════════════════════════════════
REM Parse Command Line Arguments
REM ════════════════════════════════════════════════════════════════════════════════

:parse_args
if "%~1"=="" goto :end_parse_args
if /i "%~1"=="--quick" (
    set "QUICK_MODE=true"
    set "SUBSET_SIZE=200"
    shift
    goto :parse_args
)
if /i "%~1"=="--models" (
    set "MODELS=%~2"
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
if /i "%~1"=="--data-dir" (
    set "DATA_DIR=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="-h" goto :help
if /i "%~1"=="--help" goto :help
echo [ERROR] Unknown option: %~1
exit /b 1

:help
echo Usage: %~nx0 [OPTIONS]
echo.
echo Options:
echo   --quick           Run quick test with 200 samples
echo   --models MODELS   Comma-separated list of models to run
echo   --config FILE     Use custom config file (default: config.yaml)
echo   --data-dir DIR    Custom data directory (default: data\CBIS-DDSM)
echo   -h, --help        Show this help message
exit /b 0

:end_parse_args

REM ════════════════════════════════════════════════════════════════════════════════
REM Helper Functions
REM ════════════════════════════════════════════════════════════════════════════════

:print_header
echo.
echo ════════════════════════════════════════════════════════════════
echo %~1
echo ════════════════════════════════════════════════════════════════
echo.
goto :eof

:print_success
echo [OK] %~1
goto :eof

:print_warning
echo [WARN] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

REM ════════════════════════════════════════════════════════════════════════════════
REM Check Prerequisites
REM ════════════════════════════════════════════════════════════════════════════════

call :print_header "Checking Prerequisites"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    call :print_error "Python not found. Please install Python 3.9+"
    exit /b 1
)
call :print_success "Python found"

REM Check Kaggle CLI
where kaggle >nul 2>&1
if errorlevel 1 (
    call :print_warning "Kaggle CLI not installed. Installing..."
    pip install kaggle
)
call :print_success "Kaggle CLI found"

REM Check Kaggle credentials
if "%KAGGLE_USERNAME%"=="" (
    call :print_error "Kaggle credentials not set!"
    echo.
    echo Please set the following environment variables:
    echo   set KAGGLE_USERNAME=your_username
    echo   set KAGGLE_KEY=your_api_key
    echo.
    echo Get your API key from: https://www.kaggle.com/account
    exit /b 1
)
call :print_success "Kaggle credentials found"

REM Check requirements
if not exist "requirements.txt" (
    call :print_error "requirements.txt not found!"
    exit /b 1
)

call :print_success "Installing/verifying requirements..."
pip install -q -r requirements.txt

REM ════════════════════════════════════════════════════════════════════════════════
REM Download Dataset
REM ════════════════════════════════════════════════════════════════════════════════

call :print_header "Downloading CBIS-DDSM Dataset"

if exist "%DATA_DIR%\metadata.csv" (
    call :print_success "Dataset already exists at %DATA_DIR%"
    echo Skipping download...
) else (
    echo Creating data directory: %DATA_DIR%
    mkdir "%DATA_DIR%" 2>nul
    
    echo Downloading CBIS-DDSM from Kaggle...
    echo This may take 10-30 minutes depending on your connection speed.
    
    kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset -p "%DATA_DIR%"
    
    echo Extracting dataset...
    powershell -Command "Expand-Archive -Path '%DATA_DIR%\cbis-ddsm-breast-cancer-image-dataset.zip' -DestinationPath '%DATA_DIR%' -Force"
    del "%DATA_DIR%\cbis-ddsm-breast-cancer-image-dataset.zip"
    
    call :print_success "Dataset downloaded and extracted to %DATA_DIR%"
)

REM ════════════════════════════════════════════════════════════════════════════════
REM Update Configuration
REM ════════════════════════════════════════════════════════════════════════════════

call :print_header "Updating Configuration"

set "TEMP_CONFIG=config_cbis_ddsm_temp.yaml"

echo Creating temporary config: %TEMP_CONFIG%

REM Get subset value
set "SUBSET_VALUE=null"
if not "%SUBSET_SIZE%"=="" set "SUBSET_VALUE=%SUBSET_SIZE%"

REM Create temporary config
(
    echo # Auto-generated config for CBIS-DDSM validation
    echo # Generated: %date% %time%
    echo.
    echo data:
    echo   dataset: "%DATASET%"
    echo   data_dir: "%DATA_DIR%"
    echo   task: "binary"
    echo   subset: %SUBSET_VALUE%
    echo   num_workers: 4
    echo   seed: 42
    echo.
    echo   cbis_ddsm:
    echo     data_dir: "%DATA_DIR%"
    echo     auto_download: false
    echo.
    echo # Include all other settings from original config
    echo.
) > "%TEMP_CONFIG%"

REM Append rest of original config
findstr /n "^" "%CONFIG_FILE%" | findstr /v "^1:" | findstr /v "^2:" | findstr /v "^3:" | findstr /v "^4:" | findstr /v "^5:" | findstr /v "^6:" | findstr /v "^7:" | findstr /v "^8:" | findstr /v "^9:" | findstr /v "^10:" | findstr /v "^11:" | findstr /v "^12:" | findstr /v "^13:" | findstr /v "^14:" | findstr /v "^15:" | findstr /v "^16:" | findstr /v "^17:" >> "%TEMP_CONFIG%"

call :print_success "Configuration updated"

echo.
echo Configuration summary:
echo   Dataset: %DATASET%
echo   Data Directory: %DATA_DIR%
echo   Subset Size: %SUBSET_VALUE%
echo   Config File: %TEMP_CONFIG%

REM ════════════════════════════════════════════════════════════════════════════════
REM Run Training Pipeline
REM ════════════════════════════════════════════════════════════════════════════════

call :print_header "Running Training Pipeline"

set "CMD=python run_pipeline.py --config %TEMP_CONFIG%"

if not "%MODELS%"=="" (
    set "CMD=!CMD! --models %MODELS%"
)

echo Running: !CMD!
echo.

!CMD!
if errorlevel 1 (
    call :print_error "Training pipeline failed!"
    exit /b 1
)

REM ════════════════════════════════════════════════════════════════════════════════
REM Cleanup
REM ════════════════════════════════════════════════════════════════════════════════

call :print_header "Cleanup"

if exist "%TEMP_CONFIG%" (
    del "%TEMP_CONFIG%"
    call :print_success "Removed temporary config file"
)

REM ════════════════════════════════════════════════════════════════════════════════
REM Summary
REM ════════════════════════════════════════════════════════════════════════════════

call :print_header "Pipeline Complete!"

echo Results saved to: outputs/
echo.
echo Next steps:
echo   1. Review results: type outputs\comparison_*.csv
echo   2. Check W&B for training curves
echo.
echo ════════════════════════════════════════════════════════════════════════════════

endlocal
