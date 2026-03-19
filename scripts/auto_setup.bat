@echo off
REM ═══════════════════════════════════════════════════════════════════
REM Breast Cancer Classification - Automated Setup Script (Windows)
REM ═══════════════════════════════════════════════════════════════════
REM 
REM This script automates the complete setup process:
REM 1. Creates and activates virtual environment
REM 2. Installs all dependencies
REM 3. Downloads the BreakHis dataset
REM 4. Verifies GPU availability
REM 5. Creates output directories
REM 6. Generates config if missing
REM
REM Usage:
REM   scripts\auto_setup.bat
REM   scripts\auto_setup.bat --skip-dataset
REM   scripts\auto_setup.bat --data-dir D:\path\to\data
REM ═══════════════════════════════════════════════════════════════════

setlocal EnableDelayedExpansion

REM ═══════════════════════════════════════════════════════════════════
REM Configuration
REM ═══════════════════════════════════════════════════════════════════

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "VENV_DIR=%PROJECT_ROOT%\.venv"
set "DATA_DIR=%DATA_DIR:%PROJECT_ROOT%\data%"
set "CONFIG_FILE=%PROJECT_ROOT%\config.yaml"
set "LOG_FILE=%PROJECT_ROOT%\logs\setup.log"
set "PYTHON_VERSION=3.10"

REM Skip dataset flag (default: false)
set "SKIP_DATASET=false"

REM ═══════════════════════════════════════════════════════════════════
REM Utility Functions
REM ═══════════════════════════════════════════════════════════════════

:log_info
echo [INFO] %~1
goto :eof

:log_success
echo [OK] %~1
goto :eof

:log_warning
echo [WARN] %~1
goto :eof

:log_error
echo [ERROR] %~1
goto :eof

:log_step
echo.
echo ═════════════════════════════════════════════════════════
echo %~1
echo ═════════════════════════════════════════════════════════
echo.
goto :eof

:check_python
where python >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python"
    goto :eof
)

where python3 >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=python3"
    goto :eof
)

set "PYTHON_CMD="
goto :eof

:check_pip
where pip >nul 2>&1
if %errorlevel% equ 0 (
    set "PIP_CMD=pip"
    goto :eof
)

where pip3 >nul 2>&1
if %errorlevel% equ 0 (
    set "PIP_CMD=pip3"
    goto :eof
)

set "PIP_CMD="
goto :eof

REM ═══════════════════════════════════════════════════════════════════
REM Setup Functions
REM ═══════════════════════════════════════════════════════════════════

:check_prerequisites
call :log_step "Checking Prerequisites"

call :check_python
if "%PYTHON_CMD%"=="" (
    call :log_error "Python not found. Please install Python %PYTHON_VERSION% or higher."
    echo.
    echo Download from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    exit /b 1
)

for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set "PYTHON_VER=%%i"
call :log_success "Python found: %PYTHON_VER%"

call :check_pip
if "%PIP_CMD%"=="" (
    call :log_error "pip not found"
    exit /b 1
)
call :log_success "pip found"

REM Check git (optional)
where git >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=3" %%i in ('git --version') do set "GIT_VER=%%i"
    call :log_success "Git found: %GIT_VER%"
) else (
    call :log_warning "Git not found (optional)"
)

goto :eof

:create_virtual_environment
call :log_step "Creating Virtual Environment"

if exist "%VENV_DIR%" (
    call :log_warning "Virtual environment already exists at %VENV_DIR%"
    echo.
    set /p RECREATE="Recreate? (y/N): "
    if /i not "!RECREATE!"=="y" (
        call :log_info "Using existing virtual environment"
        goto :eof
    )
    call :log_info "Removing old virtual environment..."
    rmdir /s /q "%VENV_DIR%"
)

call :log_info "Creating virtual environment in %VENV_DIR%..."
%PYTHON_CMD% -m venv "%VENV_DIR%"

if %errorlevel% equ 0 (
    call :log_success "Virtual environment created successfully"
) else (
    call :log_error "Failed to create virtual environment"
    exit /b 1
)

goto :eof

:activate_virtual_environment
call :log_info "Activating virtual environment..."

call "%VENV_DIR%\Scripts\activate.bat"

if %errorlevel% equ 0 (
    call :log_success "Virtual environment activated"
    call :log_info "Python: %PYTHON_CMD%"
) else (
    call :log_error "Failed to activate virtual environment"
    exit /b 1
)

goto :eof

:install_dependencies
call :log_step "Installing Dependencies"

set "REQUIREMENTS_FILE=%PROJECT_ROOT%\requirements.txt"

if not exist "%REQUIREMENTS_FILE%" (
    call :log_error "requirements.txt not found at %REQUIREMENTS_FILE%"
    exit /b 1
)

call :log_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel

call :log_info "Installing requirements from %REQUIREMENTS_FILE%..."
pip install -r "%REQUIREMENTS_FILE%"

if %errorlevel% equ 0 (
    call :log_success "Dependencies installed successfully"
) else (
    call :log_error "Failed to install dependencies"
    exit /b 1
)

REM Check for NVIDIA GPU
call :log_info "Checking for NVIDIA GPU..."
where nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    call :log_success "NVIDIA GPU detected"
    echo.
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    echo.
) else (
    call :log_warning "nvidia-smi not found - NVIDIA driver may not be installed"
)

goto :eof

:download_dataset
call :log_step "Downloading BreakHis Dataset"

if "%SKIP_DATASET%"=="true" (
    call :log_warning "Skipping dataset download (--skip-dataset flag set)"
    goto :eof
)

REM Check if dataset already exists
if exist "%DATA_DIR%\BreaKHis_v1" (
    call :log_success "Dataset already exists at %DATA_DIR%\BreaKHis_v1"
    goto :eof
)

call :log_info "Running dataset downloader..."

REM Create data directory
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"

REM Run the Python download script
python "%SCRIPT_DIR%download_dataset.py" --output_dir "%DATA_DIR%" --resume

if %errorlevel% equ 0 (
    call :log_success "Dataset downloaded successfully"
) else (
    call :log_warning "Dataset download failed or was interrupted"
    call :log_info "You can run it manually later:"
    call :log_info "  python scripts\download_dataset.py"
)

goto :eof

:verify_gpu
call :log_step "Verifying GPU Availability"

call :log_info "Checking PyTorch CUDA support..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul
if %errorlevel% equ 0 (
    call :log_success "PyTorch CUDA support enabled"
) else (
    call :log_warning "PyTorch CUDA not available or PyTorch not installed"
)

goto :eof

:create_directories
call :log_step "Creating Output Directories"

set "DIRS=%PROJECT_ROOT%\outputs %PROJECT_ROOT%\logs %PROJECT_ROOT%\checkpoints %DATA_DIR%"

for %%D in (%DIRS%) do (
    if not exist "%%D" (
        mkdir "%%D"
        call :log_success "Created: %%D"
    ) else (
        call :log_info "Exists: %%D"
    )
)

goto :eof

:generate_config
call :log_step "Checking Configuration"

if exist "%CONFIG_FILE%" (
    call :log_success "Config file exists: %CONFIG_FILE%"
    goto :eof
)

call :log_warning "Config file not found. Creating default config..."

(
echo # BreakHis Breast Cancer Classification - Default Config
echo # Generated by auto_setup.bat
echo.
echo data:
echo   data_dir: "data/BreaKHis_v1"
echo   task: "binary"
echo   num_workers: 8
echo   seed: 42
echo.
echo cv:
echo   n_folds: 5
echo.
echo training:
echo   epochs: 30
echo   patience: 7
echo   grad_clip: 1.0
echo.
echo models:
echo   efficientnet:
echo     enabled: true
echo     batch_size: 64
echo     lr: 1.0e-4
echo     use_amp: true
echo.
echo   spiking:
echo     enabled: true
echo     batch_size: 32
echo     lr: 8.0e-4
echo     use_amp: false
echo.
echo   hybrid_vit:
echo     enabled: true
echo     batch_size: 64
echo     lr: 1.0e-4
echo     use_amp: true
echo.
echo   vit_tiny:
echo     enabled: true
echo     batch_size: 64
echo     lr: 1.0e-4
echo     use_amp: true
echo.
echo   quantum:
echo     enabled: true
echo     batch_size: 16
echo     lr: 1.0e-4
echo     use_amp: false
echo.
echo output:
echo   output_dir: "outputs"
echo   save_best_model: true
echo   save_plots: true
echo   save_csv: true
) > "%CONFIG_FILE%"

call :log_success "Default config created: %CONFIG_FILE%"

goto :eof

:print_summary
call :log_step "Setup Complete!"

echo.
echo ┌─────────────────────────────────────────────────────────────┐
echo │  SETUP SUMMARY                                              │
echo ├─────────────────────────────────────────────────────────────┤
echo │  Project Root:    %PROJECT_ROOT%
echo │  Virtual Env:     %VENV_DIR%
echo │  Data Directory:  %DATA_DIR%
echo │  Config File:     %CONFIG_FILE%
echo │  Logs:            %LOG_FILE%
echo ├─────────────────────────────────────────────────────────────┤
echo │  Next Steps:                                                │
echo │  1. Activate venv:  %VENV_DIR%\Scripts\activate
echo │  2. Verify data:    dir %DATA_DIR%
echo │  3. Edit config:    notepad %CONFIG_FILE%
echo │  4. Run training:   python run_pipeline.py
echo └─────────────────────────────────────────────────────────────┘
echo.

goto :eof

REM ═══════════════════════════════════════════════════════════════════
REM Parse Arguments
REM ═══════════════════════════════════════════════════════════════════

:parse_args
if "%~1"=="" goto :main

if /i "%~1"=="--skip-dataset" (
    set "SKIP_DATASET=true"
    shift
    goto :parse_args
)

if /i "%~1"=="--data-dir" (
    set "DATA_DIR=%~2"
    shift
    shift
    goto :parse_args
)

if /i "%~1"=="--help" (
    echo Usage: %~nx0 [OPTIONS]
    echo.
    echo Options:
    echo   --skip-dataset    Skip dataset download
    echo   --data-dir DIR    Set custom data directory
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
if not exist "%PROJECT_ROOT%\logs" mkdir "%PROJECT_ROOT%\logs"

call :parse_args %*
call :check_prerequisites
call :create_virtual_environment
call :activate_virtual_environment
call :install_dependencies
call :create_directories
call :download_dataset
call :verify_gpu
call :generate_config
call :print_summary

call :log_success "All setup steps completed!"

endlocal
exit /b 0
