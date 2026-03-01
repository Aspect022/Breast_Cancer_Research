#!/bin/bash

echo "======================================================="
echo "Breast Cancer QML vs SNN Baseline - Setup Environment"
echo "======================================================="

echo ""
echo "[1/3] Creating Python Virtual Environment (.venv)..."
python3 -m venv .venv
if [ $? -ne 0 ]; then
    echo "[Error] Failed to create virtual environment. Ensure Python3 and python3-venv are installed."
    exit 1
fi

echo ""
echo "[2/3] Activating Virtual Environment and Upgrading PIP..."
source .venv/bin/activate
pip install --upgrade pip

echo ""
echo "[3/3] Installing Dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "======================================================="
echo "Setup Complete!"
echo "======================================================="
echo ""
echo "TO START WORKING:"
echo "1. Activate the environment by running:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Dataset Setup:"
echo "   - Download the BreakHis dataset."
echo "   - Extract it inside the 'data/' folder in this project directory."
echo "   - (e.g., data/BreaKHis_v1)"
echo ""
echo "3. Run the sanity check:"
echo "   python src/train.py --task binary --data_dir data/BreaKHis_v1 --subset 200 --epochs 3"
echo "======================================================="
