#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sda/wanght19/code/SVD/DiffuModel/SeqDiffuSeq

PY_BIN="/mnt/sda/wanght19/anaconda3/envs/FVD-DPM/bin/python"
DATA_DIR="codediff/data/lineDP_dataset"
TOKENIZER_JSON="codediff/data/lineDP_dataset/activemq_bpe/tokenizer.json"
RUN_TAG="$(date +%Y%m%d-%H%M%S)"
OUT_ROOT="codediffpro/output/loss/drift_conservative_${RUN_TAG}"
mkdir -p "${OUT_ROOT}"

DATASETS=(hbase jruby)
SEEDS=(0)
MAX_JOBS=2

train_release() {
  case "$1" in
    hbase) echo "hbase-0.94.0" ;;
    jruby) echo "jruby-1.1" ;;
    *) echo "unknown" ; return 1 ;;
  esac
}

val_release() {
  case "$1" in
    hbase) echo "hbase-0.95.0" ;;
    jruby) echo "jruby-1.4.0" ;;
    *) echo "unknown" ; return 1 ;;
  esac
}

device_for() {
  case "$1" in
    hbase) echo "cuda:0" ;;
    jruby) echo "cuda:1" ;;
    *) echo "cpu" ; return 1 ;;
  esac
}

run_one() {
  local ds="$1"
  local seed="$2"

  local device
  device="$(device_for "${ds}")"
  local tr_rel
  tr_rel="$(train_release "${ds}")"
  local va_rel
  va_rel="$(val_release "${ds}")"

  local exp_name="${ds}_conservative_seed${seed}_50ep_${RUN_TAG}"
  local log_file="${OUT_ROOT}/${exp_name}.log"
  local model_dir="codediffpro/output/model/${ds}/${exp_name}"
  local pred_dir="${OUT_ROOT}/predictions/${exp_name}"
  local metrics_dir="${OUT_ROOT}/metrics/${exp_name}"
  local summary_json="${OUT_ROOT}/${exp_name}.json"

  mkdir -p "${pred_dir}" "${metrics_dir}"

  echo "[start] dataset=${ds} seed=${seed} device=${device} exp=${exp_name}" | tee -a "${log_file}"

  "${PY_BIN}" codediffpro/train_minimal.py \
    --dataset "${ds}" \
    --data-dir "${DATA_DIR}" \
    --tokenizer-json "${TOKENIZER_JSON}" \
    --exp-name "${exp_name}" \
    --batch-size 8 \
    --num-epochs 50 \
    --num-workers 0 \
    --lr 0.0001 \
    --weight-decay 0.0001 \
    --max-tokens-per-line 48 \
    --max-lines-per-file -1 \
    --line-window-size 256 \
    --line-window-stride 128 \
    --dropout 0.2 \
    --consistency-weight 0.2 \
    --proto-weight 1.2 \
    --proto-temperature 0.07 \
    --proto-margin 0.2 \
    --loss-mode classify_focus \
    --tau 0.0 \
    --tau-mode auto \
    --tau-candidates 81 \
    --seed "${seed}" \
    --device "${device}" \
    --save-all-checkpoints \
    >> "${log_file}" 2>&1

  local ckpt_path="${model_dir}/checkpoint_50epochs.pth"
  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[error] missing checkpoint ${ckpt_path}" | tee -a "${log_file}"
    return 1
  fi

  for release in "${tr_rel}" "${va_rel}"; do
    local pred_csv="${pred_dir}/${release}.csv"
    local metrics_json="${metrics_dir}/${release}.json"
    "${PY_BIN}" codediffpro/infer_prototype.py \
      --model-dir "${model_dir}" \
      --checkpoint "checkpoint_50epochs.pth" \
      --release "${release}" \
      --output "${pred_csv}" \
      --metrics-output "${metrics_json}" \
      >> "${log_file}" 2>&1
  done

  "${PY_BIN}" - <<PY
import json
from pathlib import Path

train_metrics = json.loads(Path("${metrics_dir}/${tr_rel}.json").read_text(encoding="utf-8"))
val_metrics = json.loads(Path("${metrics_dir}/${va_rel}.json").read_text(encoding="utf-8"))
summary = {
    "dataset": "${ds}",
    "seed": ${seed},
    "train_release": "${tr_rel}",
    "val_release": "${va_rel}",
    "dropout": 0.2,
    "weight_decay": 0.0001,
    "line_window_stride": 128,
    "loss_mode": "classify_focus",
    "train_line_auc": round(float(train_metrics["line_auc"]), 6),
    "val_line_auc": round(float(val_metrics["line_auc"]), 6),
    "auc_drop": round(float(train_metrics["line_auc"]) - float(val_metrics["line_auc"]), 6),
    "train_line_best_f1": round(float(train_metrics["line_best_f1"]), 6),
    "val_line_best_f1": round(float(val_metrics["line_best_f1"]), 6),
    "f1_drop": round(float(train_metrics["line_best_f1"]) - float(val_metrics["line_best_f1"]), 6),
    "train_recall20": round(float(train_metrics["recall20"]), 6),
    "val_recall20": round(float(val_metrics["recall20"]), 6),
    "train_effort20": round(float(train_metrics["effort20"]), 6),
    "val_effort20": round(float(val_metrics["effort20"]), 6)
}
Path("${summary_json}").write_text(json.dumps(summary, indent=2), encoding="utf-8")
PY

  echo "[done] dataset=${ds} seed=${seed} exp=${exp_name}" | tee -a "${log_file}"
}

for seed in "${SEEDS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    while [[ $(jobs -pr | wc -l) -ge ${MAX_JOBS} ]]; do
      wait -n
    done
    run_one "${ds}" "${seed}" &
  done
done

wait

"${PY_BIN}" - <<PY
import csv
import json
from pathlib import Path

out = Path("${OUT_ROOT}")
rows = []
for p in sorted(out.glob("*.json")):
    rows.append(json.loads(p.read_text(encoding="utf-8")))
rows.sort(key=lambda r: (r["dataset"], int(r["seed"])))
summary_path = out / "drift_conservative_summary.csv"
if rows:
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
print(summary_path)
PY

echo "[all_done] out_root=${OUT_ROOT}"