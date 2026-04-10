#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sda/wanght19/code/SVD/DiffuModel/SeqDiffuSeq

PY_BIN="/mnt/sda/wanght19/anaconda3/envs/FVD-DPM/bin/python"
DATA_DIR="codediff/data/lineDP_dataset"
TOKENIZER_JSON="codediff/data/lineDP_dataset/activemq_bpe/tokenizer.json"
RUN_TAG="$(date +%Y%m%d-%H%M%S)"
LOG_DIR="codediffpro/output/loss/all_projects_T1_ep100_${RUN_TAG}"
mkdir -p "${LOG_DIR}"

DATASETS_GPU0=(
  camel
  derby
  groovy
  hbase
)

DATASETS_GPU1=(
  hive
  jruby
  lucene
  wicket
)

train_dataset() {
  local ds="$1"
  local device="$2"
  local exp_name="${ds}_codediffpro_T1_ep100_${RUN_TAG}"
  local log_file="${LOG_DIR}/${ds}_${device}.log"

  echo "[start] dataset=${ds} device=${device} exp=${exp_name}" | tee -a "${log_file}"

  "${PY_BIN}" codediffpro/train_minimal.py \
    --dataset "${ds}" \
    --data-dir "${DATA_DIR}" \
    --tokenizer-json "${TOKENIZER_JSON}" \
    --exp-name "${exp_name}" \
    --batch-size 8 \
    --num-epochs 100 \
    --eval-every 1 \
    --save-every 1 \
    --num-workers 0 \
    --lr 0.0001 \
    --weight-decay 0.0 \
    --max-tokens-per-line 48 \
    --max-lines-per-file -1 \
    --line-window-size 256 \
    --line-window-stride 64 \
    --balanced-sampling \
    --diffusion-steps 1 \
    --token-emb-dim 128 \
    --line-hidden-dim 128 \
    --cond-dim 256 \
    --label-emb-dim 256 \
    --num-cond-layers 2 \
    --num-denoise-layers 2 \
    --nhead 8 \
    --dropout 0.1 \
    --consistency-weight 0.1 \
    --proto-weight 1.2 \
    --proto-temperature 0.07 \
    --proto-margin 0.2 \
    --proto-pos-weight 3.0 \
    --proto-neg-weight 1.0 \
    --ranking-weight 1.0 \
    --ranking-margin 0.4 \
    --hard-negative-ratio 0.3 \
    --hard-negative-weight 0.7 \
    --hard-negative-margin 0.1 \
    --tau 0.0 \
    --tau-mode auto \
    --tau-candidates 81 \
    --seed 0 \
    --device "${device}" \
    --save-all-checkpoints \
    >> "${log_file}" 2>&1

  echo "[done] dataset=${ds} device=${device} exp=${exp_name}" | tee -a "${log_file}"
}

run_group() {
  local device="$1"
  shift
  for ds in "$@"; do
    train_dataset "${ds}" "${device}"
  done
}

run_group cuda:0 "${DATASETS_GPU0[@]}" &
PID_GPU0=$!

run_group cuda:1 "${DATASETS_GPU1[@]}" &
PID_GPU1=$!

wait "${PID_GPU0}"
wait "${PID_GPU1}"

echo "[all_done] all datasets finished on both GPUs. logs=${LOG_DIR}"
