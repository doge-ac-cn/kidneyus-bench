#!/usr/bin/env bash
# M6 USFM training queue: fires after the best-primary nnU-Net chain drains GPU0.
# Trains USFM 3 seeds (50 ep each), then runs its E1/E2 evaluation (prompt-free).
set -u
ROOT="${KIDNEYUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="$ROOT/outputs/logs"
source ~/anaconda/etc/profile.d/conda.sh
conda activate kidneyus2
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT/scripts"

echo "[$(date -Is)] waiting for best-primary chain done..." >> "$LOG/m6_queue.log"
while ! grep -q "nnunet best primary scoring done" "$LOG/m3_nnunet_best.log" 2>/dev/null; do
  sleep 300
done
sleep 30

for seed in 42 43 44; do
  echo "[$(date -Is)] usfm train seed=$seed start" >> "$LOG/m6_queue.log"
  CUDA_VISIBLE_DEVICES=0 python m6_usfm.py --mode train --seed $seed --gpu 0 \
    >> "$LOG/usfm_seed${seed}.log" 2>&1
  rc=$?
  echo "[$(date -Is)] usfm train seed=$seed exit=$rc" >> "$LOG/m6_queue.log"
done

echo "[$(date -Is)] usfm eval E1/E2 (p3) start" >> "$LOG/m6_queue.log"
CUDA_VISIBLE_DEVICES=0 python m3_run_eval.py --gpu 0 \
  --models usfm --seeds 42,43,44 --datasets okd,trusted --prompts p3 \
  >> "$LOG/m3_run_eval_usfm.log" 2>&1
rc=$?
echo "[$(date -Is)] usfm eval exit=$rc" >> "$LOG/m6_queue.log"
echo "[$(date -Is)] M6 usfm queue DONE" >> "$LOG/m6_queue.log"
