#!/usr/bin/env bash
set -euo pipefail

# ===== Defaults (edit here if you prefer) =====
PYTHON_BIN_DEFAULT="/mnt/sda/wanght19/anaconda3/envs/FVD-DPM/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_BIN_DEFAULT}"

DATASET="activemq"
DATA_DIR="codediff/data/lineDP_dataset"
TOKENIZER_JSON="codediff/data/lineDP_dataset/activemq_bpe/tokenizer.json"
EXP_NAME="activemq_sw_$(date +%Y%m%d_%H%M%S)"

NUM_EPOCHS=100
BATCH_SIZE=32
LR="1e-4"
WEIGHT_DECAY="0.0"
NUM_WORKERS=0
EVAL_EVERY_EPOCHS=5
SAVE_BEST_ONLY=1
BEST_METRIC="valid_f1"
LOG_EVERY_BATCHES=2

MAX_TOKENS_PER_LINE=48
MAX_LINES_PER_FILE=-1
LINE_WINDOW_SIZE=256
LINE_WINDOW_STRIDE=64

DIFFUSION_STEPS=20
CLS_WEIGHT="1"
CONSISTENCY_WEIGHT="0.1"
DISABLE_CLS_HEAD=0
SEED=0
DEVICE="cuda:1"

DO_PREDICT=1
PREDICT_EPOCH=-1
DROP_COMMENT=0
DROP_BLANK=0

EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash codediff/train_scripts/run.sh [options] [-- extra args for codediff/main.py]

Common options:
  --dataset NAME
  --data-dir PATH
  --tokenizer-json PATH
  --exp-name NAME
  --num-epochs N
  --batch-size N
  --device DEVICE                (e.g. cuda:0 / cuda:1 / cpu)
  --line-window-size N
  --line-window-stride N
  --max-lines-per-file N         (<=0 disables this cap)
  --max-tokens-per-line N
  --diffusion-steps N
  --eval-every-epochs N
  --consistency-weight W
  --disable-cls-head
  --best-metric NAME             (valid_f1 | valid_acc | valid_loss)
  --save-best-only               (default on)
  --no-save-best-only
  --log-every-batches N
  --predict-epoch N              (-1 means latest epoch)
  --no-predict                   (disable testing/prediction export)
  --drop-comment
  --drop-blank
  --python-bin PATH
  -h, --help

Examples:
  bash codediff/train_scripts/run.sh
  bash codediff/train_scripts/run.sh --num-epochs 5 --batch-size 2 --device cuda:0
  bash codediff/train_scripts/run.sh --line-window-size 256 --line-window-stride 128
  bash codediff/train_scripts/run.sh -- --log-every-batches 10 --log-first-batch-shape
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --tokenizer-json) TOKENIZER_JSON="$2"; shift 2 ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --num-epochs) NUM_EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --max-tokens-per-line) MAX_TOKENS_PER_LINE="$2"; shift 2 ;;
    --max-lines-per-file) MAX_LINES_PER_FILE="$2"; shift 2 ;;
    --line-window-size) LINE_WINDOW_SIZE="$2"; shift 2 ;;
    --line-window-stride) LINE_WINDOW_STRIDE="$2"; shift 2 ;;
    --diffusion-steps) DIFFUSION_STEPS="$2"; shift 2 ;;
    --eval-every-epochs) EVAL_EVERY_EPOCHS="$2"; shift 2 ;;
    --cls-weight) CLS_WEIGHT="$2"; shift 2 ;;
    --consistency-weight) CONSISTENCY_WEIGHT="$2"; shift 2 ;;
    --disable-cls-head) DISABLE_CLS_HEAD=1; shift ;;
    --seed) SEED="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --best-metric) BEST_METRIC="$2"; shift 2 ;;
    --save-best-only) SAVE_BEST_ONLY=1; shift ;;
    --no-save-best-only) SAVE_BEST_ONLY=0; shift ;;
    --log-every-batches) LOG_EVERY_BATCHES="$2"; shift 2 ;;
    --predict-epoch) PREDICT_EPOCH="$2"; shift 2 ;;
    --drop-comment) DROP_COMMENT=1; shift ;;
    --drop-blank) DROP_BLANK=1; shift ;;
    --no-predict) DO_PREDICT=0; shift ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
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
  echo "Example build command:"
  echo "  $PYTHON_BIN codediff/tokenizer_utils.py train-bpe --csv codediff/data/lineDP_dataset/${DATASET}-5.0.0.csv --out codediff/data/lineDP_dataset/${DATASET}_bpe --vocab-size 8000 --min-frequency 2 --split train --drop-comment"
  exit 1
fi

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
  --eval-every-epochs "$EVAL_EVERY_EPOCHS"
  --best-metric "$BEST_METRIC"
  --log-every-batches "$LOG_EVERY_BATCHES"
  --cls-weight "$CLS_WEIGHT"
  --consistency-weight "$CONSISTENCY_WEIGHT"
  --seed "$SEED"
  --device "$DEVICE"
  --predict-epoch "$PREDICT_EPOCH"
)

if [[ "$SAVE_BEST_ONLY" -eq 1 ]]; then
  CMD+=(--save-best-only)
fi

if [[ "$DISABLE_CLS_HEAD" -eq 1 ]]; then
  CMD+=(--disable-cls-head)
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

echo "[INFO] Running command:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"
