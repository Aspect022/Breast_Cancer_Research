#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/final_runs/logs

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="outputs/final_runs/logs/breast_cancer_final_${TIMESTAMP}.log"

nohup python run_pipeline.py --config config_final.yaml > "$LOG_PATH" 2>&1 &
PID=$!

echo "Started breast-cancer-final run"
echo "PID: $PID"
echo "Log: $LOG_PATH"
