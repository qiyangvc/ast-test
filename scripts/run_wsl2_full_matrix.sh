#!/usr/bin/env bash
# One-command WSL2 runner for the full 32-model AST training matrix.
#
# Default run:
#   bash scripts/run_wsl2_full_matrix.sh
#
# Useful overrides:
#   PROFILE=mild bash scripts/run_wsl2_full_matrix.sh
#   OUTPUT_DIR=output/submission_strong_full_matrix bash scripts/run_wsl2_full_matrix.sh
#   SKIP_DATA_PREP=1 SKIP_DATA_BUILD=1 bash scripts/run_wsl2_full_matrix.sh

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PROFILE="${PROFILE:-strong}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-.venv_submit}"
VENV_PY="${VENV_DIR}/bin/python"

VECTOR_SIZE="${VECTOR_SIZE:-200}"
MAX_VOCAB="${MAX_VOCAB:-50000}"
MAX_LEN="${MAX_LEN:-64}"
W2V_EPOCHS="${W2V_EPOCHS:-20}"
CLF_EPOCHS="${CLF_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-512}"
CONFIDENCE_ATTACK_LIMIT="${CONFIDENCE_ATTACK_LIMIT:-0}"
CONFIDENCE_ATTACK_STRENGTH="${CONFIDENCE_ATTACK_STRENGTH:-strong}"
REVIEW_SAMPLE_SIZE="${REVIEW_SAMPLE_SIZE:-0}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc 2>/dev/null || echo 4)}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"

mkdir -p output/logs
LOG_FILE="${LOG_FILE:-output/logs/wsl2_full_matrix_${PROFILE}_${RUN_ID}.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "== AST full-matrix WSL2 runner =="
echo "Project root: ${PROJECT_ROOT}"
echo "Profile: ${PROFILE}"
echo "Run id: ${RUN_ID}"
echo "Log file: ${LOG_FILE}"

if ! grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then
  echo "Warning: this script is designed for WSL2, but /proc/version does not look like WSL."
fi

case "${PROJECT_ROOT}" in
  /mnt/*)
    echo "Warning: project is under /mnt/*. WSL2 training is usually much faster under the Linux filesystem, for example ~/big_data/-111."
    ;;
esac

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is required for dataset preparation."
  echo "Install it in WSL2 with: sudo apt update && sudo apt install -y git"
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Warning: ${PYTHON_BIN} was not found; falling back to python3."
  PYTHON_BIN="python3"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: Python was not found. Install Python 3.12 in WSL2 first."
  exit 1
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required; Python 3.12 is recommended.")
print("Python:", sys.version.replace("\n", " "))
PY

if [[ ! -x "${VENV_PY}" ]]; then
  echo "Creating virtual environment: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

echo "Installing Python dependencies..."
"${VENV_PY}" -m pip install -U pip
"${VENV_PY}" -m pip install -r requirements.txt

if [[ "${SKIP_DATA_PREP:-0}" != "1" ]]; then
  echo "Preparing external datasets..."
  PREP_ARGS=()
  if [[ "${FORCE_DATA:-0}" == "1" ]]; then
    PREP_ARGS+=(--force)
  fi
  "${VENV_PY}" scripts/prepare_external_datasets.py "${PREP_ARGS[@]}"
else
  echo "Skipping external dataset preparation because SKIP_DATA_PREP=1."
fi

build_mild_dataset() {
  echo "Building mild AST dataset and manifest..."
  "${VENV_PY}" scripts/build_ast_dataset.py \
    --input-dir tensorlayer_text_antispam=data/external/raw/tensorlayer_text_antispam/msglog \
    --canonical-jsonl spam_messages_lr=data/external/canonical/spam_messages_lr.jsonl \
    --canonical-jsonl fbs_sms_dataset=data/external/canonical/fbs_sms_dataset.jsonl \
    --output-dir data/ast_experiment \
    --ast-strength mild \
    --max-variants-spam 2 \
    --max-variants-normal 1
}

if [[ "${SKIP_DATA_BUILD:-0}" != "1" ]]; then
  if [[ ! -f data/ast_experiment/manifest.json || "${REBUILD_MILD:-0}" == "1" ]]; then
    build_mild_dataset
  else
    echo "Mild AST manifest already exists: data/ast_experiment/manifest.json"
  fi
else
  echo "Skipping dataset build because SKIP_DATA_BUILD=1."
fi

case "${PROFILE}" in
  strong)
    DATASET_DIR="${DATASET_DIR:-data/ast_experiment_strong}"
    OUTPUT_DIR="${OUTPUT_DIR:-output/submission_strong_full_matrix_${RUN_ID}}"
    TRAIN_CMD=(
      "${VENV_PY}" scripts/run_strong_ast_experiment.py
      --full-matrix
      --base-manifest data/ast_experiment/manifest.json
      --dataset-dir "${DATASET_DIR}"
      --output-dir "${OUTPUT_DIR}"
      --vector-size "${VECTOR_SIZE}"
      --max-vocab "${MAX_VOCAB}"
      --max-len "${MAX_LEN}"
      --w2v-epochs "${W2V_EPOCHS}"
      --clf-epochs "${CLF_EPOCHS}"
      --batch-size "${BATCH_SIZE}"
      --confidence-attack-limit "${CONFIDENCE_ATTACK_LIMIT}"
      --confidence-attack-strength "${CONFIDENCE_ATTACK_STRENGTH}"
      --review-sample-size "${REVIEW_SAMPLE_SIZE}"
    )
    if [[ "${SKIP_DATA_BUILD:-0}" == "1" ]]; then
      TRAIN_CMD+=(--skip-build)
    fi
    ;;
  mild)
    DATASET_DIR="${DATASET_DIR:-data/ast_experiment}"
    OUTPUT_DIR="${OUTPUT_DIR:-output/submission_mild_full_matrix_${RUN_ID}}"
    TRAIN_CMD=(
      "${VENV_PY}" scripts/submission_pipeline.py
      --full-matrix
      --dataset-dir "${DATASET_DIR}"
      --output-dir "${OUTPUT_DIR}"
      --vector-size "${VECTOR_SIZE}"
      --max-vocab "${MAX_VOCAB}"
      --max-len "${MAX_LEN}"
      --w2v-epochs "${W2V_EPOCHS}"
      --clf-epochs "${CLF_EPOCHS}"
      --batch-size "${BATCH_SIZE}"
      --confidence-attack-limit "${CONFIDENCE_ATTACK_LIMIT}"
      --confidence-attack-strength mild
      --review-sample-size "${REVIEW_SAMPLE_SIZE}"
    )
    ;;
  *)
    echo "Error: unsupported PROFILE=${PROFILE}. Use PROFILE=strong or PROFILE=mild."
    exit 1
    ;;
esac

echo "Dataset dir: ${DATASET_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Training matrix: 8 modes x 4 base models = 32 trainings, plus ensemble_vote evaluation."
echo "Starting training..."
printf 'Command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'

"${TRAIN_CMD[@]}"

echo "Training completed."
echo "Output dir: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
