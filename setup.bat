@echo off
echo =======================================================
echo Breast Cancer QML vs SNN Baseline - Setup Environment
echo =======================================================

echo.
echo [1/3] Creating Python Virtual Environment (.venv)...
python -m venv .venv
if %errorlevel% neq 0 (
    echo [Error] Failed to create virtual environment. Ensure Python is installed and in your PATH.
    pause
    exit /b %errorlevel%
)

echo.
echo [2/3] Activating Virtual Environment and Upgrading PIP...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo.
echo [3/3] Installing Dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo =======================================================
echo Setup Complete!
echo =======================================================
echo.
echo TO START WORKING:
echo 1. Activate the environment by running:
echo    .venv\Scripts\activate
echo.
echo 2. Dataset Setup:
echo    - Download the BreakHis dataset.
echo    - Extract it inside the 'data/' folder in this project directory.
echo    - (e.g., d:\Projects\AI-Projects\Brest_cancer\data\BreaKHis_v1)
echo.
echo 3. Run the sanity check:
echo    python src/train.py --task binary --data_dir data/BreaKHis_v1 --subset 200 --epochs 3
echo =======================================================
pause
