#!/usr/bin/env bash
set -euo pipefail

# ===== Defaults (aligned with codediff/train_scripts/run.sh) =====
PYTHON_BIN_DEFAULT="/mnt/sda/wanght19/anaconda3/envs/FVD-DPM/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"

DATASET="activemq"
DATA_DIR="codediff/data/lineDP_dataset"
TOKENIZER_JSON="codediff/data/lineDP_dataset/activemq_bpe/tokenizer.json"
EXP_NAME="activemq_codebert_10e_$(date +%Y%m%d_%H%M%S)"
LOG_FILE=""

# Fixed 10 epochs for this script
NUM_EPOCHS=10
BATCH_SIZE=32
LR="1e-4"
WEIGHT_DECAY="0.0"
NUM_WORKERS=0

MAX_TOKENS_PER_LINE=48
MAX_LINES_PER_FILE=-1
LINE_WINDOW_SIZE=256
LINE_WINDOW_STRIDE=64

DIFFUSION_STEPS=50
CLS_WEIGHT="1"
SEED=0
DEVICE="cuda:1"

# CodeBERT options (local-only)
LINE_ENCODER="codebert"
CODEBERT_LOCAL_PATH="/mnt/sda/wanght19/code/huggingface/codebert-base"
FREEZE_CODEBERT=0
CODEBERT_UNFREEZE_EPOCH=-1

DO_PREDICT=1
PREDICT_EPOCH=-1
DROP_COMMENT=0
DROP_BLANK=0

# OOM auto-backoff options
AUTO_REDUCE_BATCH_ON_OOM=1
MIN_BATCH_SIZE=1
OOM_MAX_RETRIES=6

EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash codediff/train_scripts/run_codebert_10.sh [options] [-- extra args for codediff/main.py]

Common options:
  --dataset NAME
  --data-dir PATH
  --tokenizer-json PATH
  --exp-name NAME
  --batch-size N
  --device DEVICE                (e.g. cuda:0 / cuda:1 / cpu)
  --line-window-size N
  --line-window-stride N
  --max-lines-per-file N         (<=0 disables this cap)
  --max-tokens-per-line N
  --diffusion-steps N
  --cls-weight V
  --codebert-local-path PATH
  --freeze-codebert
  --codebert-unfreeze-epoch N    (-1 auto half-freeze-half-tune; 0 means tune from epoch1)
  --predict-epoch N              (-1 means latest epoch)
  --no-predict                   (disable prediction export)
  --drop-comment
  --drop-blank
  --python-bin PATH
  --log-file PATH
  --disable-auto-batch-oom
  --min-batch-size N
  --oom-max-retries N
  -h, --help

Notes:
  - This script always trains for 10 epochs.
  - CodeBERT is loaded only from local path via --codebert-local-path.
  - On CUDA OOM, batch size is halved automatically until success/min batch.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --tokenizer-json) TOKENIZER_JSON="$2"; shift 2 ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --max-tokens-per-line) MAX_TOKENS_PER_LINE="$2"; shift 2 ;;
    --max-lines-per-file) MAX_LINES_PER_FILE="$2"; shift 2 ;;
    --line-window-size) LINE_WINDOW_SIZE="$2"; shift 2 ;;
    --line-window-stride) LINE_WINDOW_STRIDE="$2"; shift 2 ;;
    --diffusion-steps) DIFFUSION_STEPS="$2"; shift 2 ;;
    --cls-weight) CLS_WEIGHT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --codebert-local-path) CODEBERT_LOCAL_PATH="$2"; shift 2 ;;
    --freeze-codebert) FREEZE_CODEBERT=1; shift ;;
    --codebert-unfreeze-epoch) CODEBERT_UNFREEZE_EPOCH="$2"; shift 2 ;;
    --predict-epoch) PREDICT_EPOCH="$2"; shift 2 ;;
    --drop-comment) DROP_COMMENT=1; shift ;;
    --drop-blank) DROP_BLANK=1; shift ;;
    --no-predict) DO_PREDICT=0; shift ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --log-file) LOG_FILE="$2"; shift 2 ;;
    --disable-auto-batch-oom) AUTO_REDUCE_BATCH_ON_OOM=0; shift ;;
    --min-batch-size) MIN_BATCH_SIZE="$2"; shift 2 ;;
    --oom-max-retries) OOM_MAX_RETRIES="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] Python binary not executable: $PYTHON_BIN"
  echo "Set with --python-bin or env PYTHON_BIN"
  exit 1
fi

if [[ ! -f "$TOKENIZER_JSON" ]]; then
  echo "[ERROR] tokenizer.json not found: $TOKENIZER_JSON"
  exit 1
fi

if [[ ! -d "$CODEBERT_LOCAL_PATH" ]]; then
  echo "[ERROR] CodeBERT local path not found: $CODEBERT_LOCAL_PATH"
  exit 1
fi

if [[ -z "$LOG_FILE" ]]; then
  LOG_DIR="codediff/output/log/Codediff"
  mkdir -p "$LOG_DIR"
  LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
else
  mkdir -p "$(dirname "$LOG_FILE")"
fi

# Redirect all script output (including training command output) to one log file.
exec >> "$LOG_FILE" 2>&1
echo "[INFO] Log file: $LOG_FILE"

CMD=(
  "$PYTHON_BIN" codediff/main.py
  --dataset "$DATASET"
  --data-dir "$DATA_DIR"
  --tokenizer-json "$TOKENIZER_JSON"
  --exp-name "$EXP_NAME"
  --num-epochs "$NUM_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --num-workers "$NUM_WORKERS"
  --max-tokens-per-line "$MAX_TOKENS_PER_LINE"
  --max-lines-per-file "$MAX_LINES_PER_FILE"
  --line-window-size "$LINE_WINDOW_SIZE"
  --line-window-stride "$LINE_WINDOW_STRIDE"
  --diffusion-steps "$DIFFUSION_STEPS"
  --cls-weight "$CLS_WEIGHT"
  --line-encoder "$LINE_ENCODER"
  --codebert-local-path "$CODEBERT_LOCAL_PATH"
  --codebert-unfreeze-epoch "$CODEBERT_UNFREEZE_EPOCH"
  --seed "$SEED"
  --device "$DEVICE"
  --predict-epoch "$PREDICT_EPOCH"
)

if [[ "$FREEZE_CODEBERT" -eq 1 ]]; then
  CMD+=(--freeze-codebert)
fi
if [[ "$DO_PREDICT" -eq 1 ]]; then
  CMD+=(--do-predict)
fi
if [[ "$DROP_COMMENT" -eq 1 ]]; then
  CMD+=(--drop-comment)
fi
if [[ "$DROP_BLANK" -eq 1 ]]; then
  CMD+=(--drop-blank)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

CURRENT_BATCH_SIZE="$BATCH_SIZE"
ATTEMPT=0
OOM_LOG_FILE="/tmp/codediff_codebert_oom_retry_$$.log"

while true; do
  ATTEMPT=$((ATTEMPT + 1))

  # Rebuild command with current batch size.
  CMD=(
    "$PYTHON_BIN" codediff/main.py
    --dataset "$DATASET"
    --data-dir "$DATA_DIR"
    --tokenizer-json "$TOKENIZER_JSON"
    --exp-name "$EXP_NAME"
    --num-epochs "$NUM_EPOCHS"
    --batch-size "$CURRENT_BATCH_SIZE"
    --lr "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --num-workers "$NUM_WORKERS"
    --max-tokens-per-line "$MAX_TOKENS_PER_LINE"
    --max-lines-per-file "$MAX_LINES_PER_FILE"
    --line-window-size "$LINE_WINDOW_SIZE"
    --line-window-stride "$LINE_WINDOW_STRIDE"
    --diffusion-steps "$DIFFUSION_STEPS"
    --cls-weight "$CLS_WEIGHT"
    --line-encoder "$LINE_ENCODER"
    --codebert-local-path "$CODEBERT_LOCAL_PATH"
    --codebert-unfreeze-epoch "$CODEBERT_UNFREEZE_EPOCH"
    --seed "$SEED"
    --device "$DEVICE"
    --predict-epoch "$PREDICT_EPOCH"
  )

  if [[ "$FREEZE_CODEBERT" -eq 1 ]]; then
    CMD+=(--freeze-codebert)
  fi
  if [[ "$DO_PREDICT" -eq 1 ]]; then
    CMD+=(--do-predict)
  fi
  if [[ "$DROP_COMMENT" -eq 1 ]]; then
    CMD+=(--drop-comment)
  fi
  if [[ "$DROP_BLANK" -eq 1 ]]; then
    CMD+=(--drop-blank)
  fi
  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
  fi

  echo "[INFO] Attempt ${ATTEMPT}, batch_size=${CURRENT_BATCH_SIZE}, running command:"
  printf ' %q' "${CMD[@]}"
  echo

  set +e
  "${CMD[@]}" 2>&1 | tee "$OOM_LOG_FILE"
  RUN_STATUS=${PIPESTATUS[0]}
  set -e

  if [[ "$RUN_STATUS" -eq 0 ]]; then
    rm -f "$OOM_LOG_FILE"
    exit 0
  fi

  if [[ "$AUTO_REDUCE_BATCH_ON_OOM" -ne 1 ]]; then
    echo "[ERROR] Training failed (status=${RUN_STATUS}), auto OOM backoff disabled."
    rm -f "$OOM_LOG_FILE"
    exit "$RUN_STATUS"
  fi

  IS_OOM_TEXT=0
  if grep -Eiq "out of memory|cuda out of memory|cuda error: out of memory|oom-kill|killed process|\bkilled\b" "$OOM_LOG_FILE"; then
    IS_OOM_TEXT=1
  fi

  IS_OOM_EXIT=0
  if [[ "$RUN_STATUS" -eq 137 || "$RUN_STATUS" -eq 9 ]]; then
    IS_OOM_EXIT=1
  fi

  if [[ "$IS_OOM_TEXT" -ne 1 && "$IS_OOM_EXIT" -ne 1 ]]; then
    echo "[ERROR] Training failed (status=${RUN_STATUS}) but not an OOM-like error."
    rm -f "$OOM_LOG_FILE"
    exit "$RUN_STATUS"
  fi

  if [[ "$ATTEMPT" -ge "$OOM_MAX_RETRIES" ]]; then
    echo "[ERROR] Reached oom max retries (${OOM_MAX_RETRIES})."
    rm -f "$OOM_LOG_FILE"
    exit "$RUN_STATUS"
  fi

  if [[ "$CURRENT_BATCH_SIZE" -le "$MIN_BATCH_SIZE" ]]; then
    echo "[ERROR] OOM at minimum batch size (${MIN_BATCH_SIZE})."
    rm -f "$OOM_LOG_FILE"
    exit "$RUN_STATUS"
  fi

  NEXT_BATCH_SIZE=$((CURRENT_BATCH_SIZE / 2))
  if [[ "$NEXT_BATCH_SIZE" -lt "$MIN_BATCH_SIZE" ]]; then
    NEXT_BATCH_SIZE="$MIN_BATCH_SIZE"
  fi
  echo "[WARN] CUDA OOM detected, reducing batch size: ${CURRENT_BATCH_SIZE} -> ${NEXT_BATCH_SIZE}"
  CURRENT_BATCH_SIZE="$NEXT_BATCH_SIZE"
done
