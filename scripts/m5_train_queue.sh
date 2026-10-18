#!/usr/bin/env bash
# M5: SAMUS decoder-only light FT x3 seeds — waits for the medsam rerun to drain.
set -u
ROOT="${KIDNEYUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="$ROOT/outputs/logs"
RERUNLOG="$LOG/m2_rerun.log"
source ~/anaconda/etc/profile.d/conda.sh
conda activate kidneyus2
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT/scripts"

echo "[$(date -Is)] waiting for medsam rerun to finish..." >> "$LOG/m5_queue.log"
while ! grep -q "medsam rerun done" "$RERUNLOG" 2>/dev/null; do
  sleep 300
done
sleep 30

for seed in 42 43 44; do
  echo "[$(date -Is)] samus train seed=$seed start" >> "$LOG/m5_queue.log"
  CUDA_VISIBLE_DEVICES=0 python m5_samus.py --mode train --seed $seed --gpu 0 \
    >> "$LOG/samus_seed${seed}.log" 2>&1
  rc=$?
  echo "[$(date -Is)] samus train seed=$seed exit=$rc" >> "$LOG/m5_queue.log"
done
echo "[$(date -Is)] M5 samus queue done" >> "$LOG/m5_queue.log"
