#!/usr/bin/env bash
set -euo pipefail

# Unattended server pipeline for this repository.
#
# One-command usage from the project root:
#   mkdir -p logs
#   nohup bash run_server_auto_pipeline.sh > logs/server_auto_pipeline.out 2>&1 &
#   tail -f logs/server_auto_pipeline.out
#
# Useful optional overrides:
#   RUN_GENOMICS=0 bash run_server_auto_pipeline.sh
#   RUN_HISTOLOGY=0 bash run_server_auto_pipeline.sh
#   REQUIRE_CUDA=0 bash run_server_auto_pipeline.sh
#   GENOMICS_CONFIG=config_genomics_a100.yaml bash run_server_auto_pipeline.sh
#   HISTOLOGY_MODELS="cnn resnet18 tbca" bash run_server_auto_pipeline.sh
#
# The script is intentionally strict. It downloads datasets on the server
# whenever possible and never creates or accepts simulated production data.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs
RUN_ID="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="logs/server_auto_pipeline_${RUN_ID}.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

fail() {
  printf '\nERROR: %s\n' "$*" >&2
  printf 'Master log: %s\n' "$MASTER_LOG" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

count_images() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    printf '0'
    return
  fi
  find "$path" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l | tr -d ' '
}

download_url() {
  local url="$1"
  local output="$2"
  local label="$3"

  mkdir -p "$(dirname "$output")"
  if have curl; then
    curl -L --fail --retry 5 --retry-delay 10 --connect-timeout 30 -o "$output" "$url"
  elif have wget; then
    wget --tries=5 --timeout=30 -O "$output" "$url"
  else
    fail "Neither curl nor wget is available for downloading ${label}."
  fi
}

extract_zip() {
  local archive="$1"
  local destination="$2"

  mkdir -p "$destination"
  python - "$archive" "$destination" <<'PY'
import sys
import zipfile

archive, destination = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(archive, "r") as zf:
    zf.extractall(destination)
print(f"Extracted {archive} into {destination}")
PY
}

RUN_GENOMICS="${RUN_GENOMICS:-1}"
RUN_HISTOLOGY="${RUN_HISTOLOGY:-1}"
RUN_BREAKHIS="${RUN_BREAKHIS:-1}"
RUN_WBCD="${RUN_WBCD:-1}"
AUTO_DOWNLOAD_GEO="${AUTO_DOWNLOAD_GEO:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"

TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
GENOMICS_CONFIG="${GENOMICS_CONFIG:-config_genomics_rtx5050.yaml}"
HISTOLOGY_CONFIG="${HISTOLOGY_CONFIG:-config.yaml}"

USER_BREAKHIS_ROOT_SET="${BREAKHIS_ROOT+x}"
USER_GEO_ROOT_SET="${GEO_ROOT+x}"
BREAKHIS_ROOT="${BREAKHIS_ROOT:-data/BreaKHis_v1}"
BREAKHIS_MIN_IMAGES="${BREAKHIS_MIN_IMAGES:-1000}"
BREAKHIS_KAGGLE_DATASET="${BREAKHIS_KAGGLE_DATASET:-ambarish/breakhis}"
BREAKHIS_DIRECT_URL="${BREAKHIS_DIRECT_URL:-https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/BreaKHis_v1.zip}"
BREAKHIS_ARCHIVE="${BREAKHIS_ARCHIVE:-data/BreaKHis_v1.zip}"
GEO_ROOT="${GEO_ROOT:-data/geo_tnbc}"
GEO_COHORTS="${GEO_COHORTS:-GSE25066 GSE20271 GSE20194 GSE32646}"
GENOMICS_MODELS="${GENOMICS_MODELS:-g_baseline_mlp g_baseline_trees}"
HISTOLOGY_MODELS="${HISTOLOGY_MODELS:-}"

WBCD_URL="${WBCD_URL:-https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data}"
WBCD_OUTPUT="${WBCD_OUTPUT:-data/WBCD/wbcd.csv}"

PYTHON_BIN="${PYTHON_BIN:-}"

# Keep unattended server runs from blocking on W&B login. Set WANDB_MODE=online
# before running the script if the server already has a valid W&B setup.
export WANDB_MODE="${WANDB_MODE:-offline}"

select_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    return
  fi

  if have python3.11; then
    PYTHON_BIN="python3.11"
  elif have python3.10; then
    PYTHON_BIN="python3.10"
  elif have python3; then
    PYTHON_BIN="python3"
  else
    fail "Could not find python3.11, python3.10, or python3 on PATH."
  fi
}

setup_environment() {
  log "Preparing Python environment"
  select_python
  "$PYTHON_BIN" --version

  if [[ ! -d ".venv" ]]; then
    log "Creating .venv"
    "$PYTHON_BIN" -m venv .venv
  else
    log "Using existing .venv"
  fi

  # shellcheck source=/dev/null
  source .venv/bin/activate

  python -m pip install --upgrade pip setuptools wheel

  log "Installing PyTorch from ${TORCH_INDEX_URL}"
  python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"

  log "Installing repository requirements"
  python -m pip install -r requirements.txt

  log "Installing genomics dependencies"
  python -m pip install GEOparse gseapy xgboost imbalanced-learn lifelines
  python -m pip install pycombat || log "Optional pycombat install failed; continuing because current implemented genomics baselines do not require it."
}

load_configured_dataset_paths() {
  log "Reading dataset paths from config files"

  if [[ -z "$USER_BREAKHIS_ROOT_SET" && -f "$HISTOLOGY_CONFIG" ]]; then
    local configured_breakhis
    configured_breakhis="$(python - "$HISTOLOGY_CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
print((cfg.get("data") or {}).get("data_dir") or "")
PY
)"
    if [[ -n "$configured_breakhis" ]]; then
      BREAKHIS_ROOT="$configured_breakhis"
    fi
  fi

  if [[ -z "$USER_GEO_ROOT_SET" && -f "$GENOMICS_CONFIG" ]]; then
    local configured_geo
    configured_geo="$(python - "$GENOMICS_CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}
print((cfg.get("data") or {}).get("data_dir") or "")
PY
)"
    if [[ -n "$configured_geo" ]]; then
      GEO_ROOT="$configured_geo"
    fi
  fi

  log "Histology dataset path: ${BREAKHIS_ROOT}"
  log "Genomics dataset path: ${GEO_ROOT}"
}

verify_cuda() {
  log "Checking CUDA visibility"
  set +e
  python - <<'PY'
import sys
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device count:", torch.cuda.device_count())
    print("cuda device 0:", torch.cuda.get_device_name(0))
else:
    print("cuda device count: 0")

sys.exit(0 if torch.cuda.is_available() else 2)
PY
  local status=$?
  set -e
  if [[ "$status" -ne 0 && "$REQUIRE_CUDA" == "1" ]]; then
    fail "CUDA is not available. Set REQUIRE_CUDA=0 only for CPU smoke runs."
  fi
  if [[ "$status" -ne 0 ]]; then
    log "CUDA is unavailable, but REQUIRE_CUDA=0 was set. Continuing."
  fi
}

download_breakhis_with_kaggle() {
  log "Attempting BreakHis download with Kaggle dataset: ${BREAKHIS_KAGGLE_DATASET}"
  python - "$BREAKHIS_KAGGLE_DATASET" <<'PY'
import sys
from pathlib import Path

dataset = sys.argv[1]
download_root = Path("data")
download_root.mkdir(parents=True, exist_ok=True)

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception as exc:
    print(f"Kaggle Python package is unavailable: {exc}")
    sys.exit(2)

api = KaggleApi()
api.authenticate()
api.dataset_download_files(dataset, path=str(download_root), unzip=True, quiet=False)
print(f"Kaggle download finished for {dataset}")
PY
}

download_breakhis_with_direct_url() {
  log "Attempting BreakHis direct URL download"
  download_url "$BREAKHIS_DIRECT_URL" "$BREAKHIS_ARCHIVE" "BreakHis archive"
  extract_zip "$BREAKHIS_ARCHIVE" "data"
}

ensure_breakhis_dataset() {
  if [[ "$RUN_BREAKHIS" != "1" && "$RUN_HISTOLOGY" != "1" ]]; then
    log "Skipping BreakHis dataset check"
    return
  fi

  log "Checking BreakHis dataset at ${BREAKHIS_ROOT}"
  mkdir -p "$BREAKHIS_ROOT"

  local before_count
  before_count="$(count_images "$BREAKHIS_ROOT")"
  log "BreakHis image count before download: ${before_count}"

  if [[ "$before_count" -lt "$BREAKHIS_MIN_IMAGES" ]]; then
    log "BreakHis is missing or incomplete. Downloading on this server."
    if ! download_breakhis_with_kaggle; then
      log "Kaggle BreakHis download failed. Trying direct URL fallback."
      download_breakhis_with_direct_url || log "Direct BreakHis URL download failed."
    fi
  fi

  local after_count
  after_count="$(count_images "$BREAKHIS_ROOT")"
  log "BreakHis image count after check: ${after_count}"

  if [[ "$after_count" -lt "$BREAKHIS_MIN_IMAGES" ]]; then
    local data_count
    data_count="$(count_images "data")"
    log "Total image count anywhere under data/: ${data_count}"
    fail "BreakHis still looks incomplete at ${BREAKHIS_ROOT}. Expected at least ${BREAKHIS_MIN_IMAGES} images. Check Kaggle credentials, BREAKHIS_KAGGLE_DATASET, or set BREAKHIS_ROOT to the extracted BreakHis path."
  fi
}

ensure_wbcd_dataset() {
  if [[ "$RUN_WBCD" != "1" ]]; then
    log "Skipping WBCD dataset check"
    return
  fi

  log "Checking WBCD dataset"
  if [[ ! -f "$WBCD_OUTPUT" ]]; then
    log "Downloading WBCD directly from UCI"
    download_url "$WBCD_URL" "$WBCD_OUTPUT" "WBCD"
  fi
}

download_geo_raw_if_needed() {
  mkdir -p "${GEO_ROOT}/raw"

  local missing_cohorts=()
  local cohort
  for cohort in $GEO_COHORTS; do
    if ! find "${GEO_ROOT}/raw" -type f -iname "*${cohort}*" 2>/dev/null | grep -q .; then
      missing_cohorts+=("$cohort")
    fi
  done

  if [[ "${#missing_cohorts[@]}" -eq 0 ]]; then
    log "All configured GEO raw cohorts already exist under ${GEO_ROOT}/raw"
    return
  fi

  if [[ "$AUTO_DOWNLOAD_GEO" != "1" ]]; then
    log "AUTO_DOWNLOAD_GEO=0, so raw GEO download is skipped"
    return
  fi

  log "Downloading missing GEO cohorts: ${missing_cohorts[*]}"
  for cohort in "${missing_cohorts[@]}"; do
    local prefix
    prefix="$(printf '%s' "$cohort" | sed -E 's/[0-9]{3}$/nnn/')"
    local url="https://ftp.ncbi.nlm.nih.gov/geo/series/${prefix}/${cohort}/soft/${cohort}_family.soft.gz"
    local output="${GEO_ROOT}/raw/${cohort}_family.soft.gz"
    log "Downloading ${cohort} from ${url}"
    download_url "$url" "$output" "$cohort GEO SOFT"
  done
}

missing_genomics_processed_files() {
  local missing=0
  for file in \
    "${GEO_ROOT}/processed/combined_325_genes_100.csv" \
    "${GEO_ROOT}/processed/combined_325_genes_500.csv" \
    "${GEO_ROOT}/processed/labels_pcr_rd.csv"; do
    if [[ ! -f "$file" ]]; then
      printf '%s\n' "$file"
      missing=1
    fi
  done
  return "$missing"
}

ensure_genomics_dataset() {
  if [[ "$RUN_GENOMICS" != "1" ]]; then
    log "Skipping genomics dataset check"
    return
  fi

  log "Checking processed TNBC genomics files"
  mkdir -p "${GEO_ROOT}/processed"

  local missing_file_list="logs/genomics_missing_files_${RUN_ID}.txt"

  if missing_genomics_processed_files >"$missing_file_list"; then
    log "Processed TNBC genomics files are present"
    return
  fi

  download_geo_raw_if_needed

  if missing_genomics_processed_files >"$missing_file_list"; then
    log "Processed TNBC genomics files are present after raw download"
    return
  fi

  log "Raw GEO data may now be present, but processed model-ready CSV files are still missing."
  cat "$missing_file_list"
  fail "Genomics training needs the processed CSV files listed above. Add them under ${GEO_ROOT}/processed or run with RUN_GENOMICS=0. The script will not generate synthetic genomics results in production mode."
}

run_genomics_models() {
  if [[ "$RUN_GENOMICS" != "1" ]]; then
    log "Skipping genomics models"
    return
  fi

  log "Running genomics models from ${GENOMICS_CONFIG}"
  [[ -f "$GENOMICS_CONFIG" ]] || fail "Genomics config not found: ${GENOMICS_CONFIG}"
  read -r -a genomics_model_args <<< "$GENOMICS_MODELS"
  local genomics_log="logs/genomics_${RUN_ID}.log"
  python run_genomic.py \
    --config "$GENOMICS_CONFIG" \
    --models "${genomics_model_args[@]}" \
    2>&1 | tee "$genomics_log"

  if grep -q "ERROR running" "$genomics_log"; then
    fail "At least one genomics model failed. Check ${genomics_log}."
  fi
}

run_histology_models() {
  if [[ "$RUN_HISTOLOGY" != "1" ]]; then
    log "Skipping histopathology models"
    return
  fi

  log "Running histopathology models from ${HISTOLOGY_CONFIG}"
  [[ -f "$HISTOLOGY_CONFIG" ]] || fail "Histopathology config not found: ${HISTOLOGY_CONFIG}"
  local histology_log="logs/histology_${RUN_ID}.log"
  if [[ -z "$HISTOLOGY_MODELS" ]]; then
    python run_pipeline.py \
      --config "$HISTOLOGY_CONFIG" \
      2>&1 | tee "$histology_log"
  else
    read -r -a histology_model_args <<< "$HISTOLOGY_MODELS"
    python run_pipeline.py \
      --config "$HISTOLOGY_CONFIG" \
      --models "${histology_model_args[@]}" \
      2>&1 | tee "$histology_log"
  fi

  if grep -q "ERROR running" "$histology_log"; then
    fail "At least one histopathology model failed. Check ${histology_log}."
  fi
}

validate_outputs() {
  log "Validating expected result files"

  local failures=0
  if [[ "$RUN_GENOMICS" == "1" && ! -f "outputs_genomics/comparison_geo_tnbc.csv" ]]; then
    printf 'Missing expected genomics comparison: outputs_genomics/comparison_geo_tnbc.csv\n' >&2
    failures=1
  fi

  if [[ "$RUN_HISTOLOGY" == "1" && ! -f "outputs/comparison_binary.csv" ]]; then
    printf 'Missing expected histology comparison: outputs/comparison_binary.csv\n' >&2
    failures=1
  fi

  if [[ "$failures" -ne 0 ]]; then
    fail "One or more expected result files were not created."
  fi

  log "Pipeline finished successfully"
  printf '\nSaved logs:\n'
  printf '  Master: %s\n' "$MASTER_LOG"
  [[ "$RUN_GENOMICS" == "1" ]] && printf '  Genomics: logs/genomics_%s.log\n' "$RUN_ID"
  [[ "$RUN_HISTOLOGY" == "1" ]] && printf '  Histology: logs/histology_%s.log\n' "$RUN_ID"

  printf '\nSaved result directories:\n'
  [[ "$RUN_GENOMICS" == "1" ]] && printf '  outputs_genomics/\n'
  [[ "$RUN_HISTOLOGY" == "1" ]] && printf '  outputs/\n'
}

main() {
  log "Starting unattended server pipeline"
  log "Project root: ${PROJECT_ROOT}"
  log "Run id: ${RUN_ID}"

  setup_environment
  load_configured_dataset_paths
  verify_cuda
  ensure_breakhis_dataset
  ensure_wbcd_dataset
  ensure_genomics_dataset
  run_genomics_models
  run_histology_models
  validate_outputs
}

main "$@"
