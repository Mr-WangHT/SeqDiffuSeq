#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sda/wanght19/code/SVD/DiffuModel/SeqDiffuSeq

/mnt/sda/wanght19/anaconda3/envs/FVD-DPM/bin/python codediffpro/train_minimal.py \
  --dataset activemq \
  --data-dir codediff/data/lineDP_dataset \
  --tokenizer-json codediff/data/lineDP_dataset/activemq_bpe/tokenizer.json \
  --exp-name activemq_codediffpro_rank_boostA_ep8_20260326 \
  --batch-size 8 \
  --num-epochs 8 \
  --num-workers 0 \
  --lr 0.0001 \
  --weight-decay 0.0 \
  --max-tokens-per-line 48 \
  --max-lines-per-file -1 \
  --line-window-size 256 \
  --line-window-stride 64 \
  --diffusion-steps 20 \
  --token-emb-dim 128 \
  --line-hidden-dim 128 \
  --cond-dim 256 \
  --label-emb-dim 256 \
  --num-cond-layers 2 \
  --num-denoise-layers 2 \
  --nhead 8 \
  --dropout 0.1 \
  --balanced-sampling \
  --consistency-weight 0.1 \
  --proto-weight 1.2 \
  --proto-temperature 0.07 \
  --proto-margin 0.2 \
  --proto-pos-weight 3.0 \
  --proto-neg-weight 1.0 \
  --ranking-weight 1.0 \
  --ranking-margin 0.5 \
  --ranking-focal-gamma 2.0 \
  --hard-negative-ratio 0.5 \
  --hard-negative-weight 0.7 \
  --hard-negative-margin 0.1 \
  --positive-margin-weight 0.6 \
  --positive-margin-target 0.5 \
  --loss-mode rank_focus \
  --tau 0.0 \
  --tau-mode auto \
  --tau-candidates 81 \
  --best-model-metric rank_score \
  --best-model-alpha 0.6 \
  --seed 0 \
  --save-all-checkpoints \
  --device cuda:1
