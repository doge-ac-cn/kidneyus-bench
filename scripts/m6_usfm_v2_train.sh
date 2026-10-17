#!/usr/bin/env bash
# M6 USFM official-recipe fine-tune x3 seeds -> usfm E1/E2 eval (p3)
set -u
ROOT="${KIDNEYUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="$ROOT/outputs/logs"
source ~/anaconda/etc/profile.d/conda.sh
conda activate kidneyus2
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT/scripts"

echo "[$(date -Is)] M6 USFM official-recipe chain start" >> "$LOG/m6_v2.log"
for seed in 42 43 44; do
  echo "[$(date -Is)] usfm v2 train seed=$seed start" >> "$LOG/m6_v2.log"
  CUDA_VISIBLE_DEVICES=0 python m6_usfm.py --mode train --seed $seed --gpu 0 \
    >> "$LOG/usfm_seed${seed}.log" 2>&1
  rc=$?
  echo "[$(date -Is)] usfm v2 train seed=$seed exit=$rc" >> "$LOG/m6_v2.log"
done

echo "[$(date -Is)] usfm eval E1/E2 (p3) start" >> "$LOG/m6_v2.log"
CUDA_VISIBLE_DEVICES=0 python m3_run_eval.py --gpu 0 \
  --models usfm --seeds 42,43,44 --datasets okd,trusted --prompts p3 \
  >> "$LOG/m3_run_eval_usfm.log" 2>&1
rc=$?
echo "[$(date -Is)] usfm eval exit=$rc" >> "$LOG/m6_v2.log"
echo "[$(date -Is)] M6 USFM v2 chain DONE" >> "$LOG/m6_v2.log"
