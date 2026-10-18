#!/usr/bin/env bash
# Re-run MedSAM seeds 42/43 (invalid: seed42 killed after ep01, seed43 OOM race).
# Waits for the GPU0 foundation queue to finish first. Exit codes recorded correctly.
set -u
ROOT="${KIDNEYUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="$ROOT/outputs/logs"
QUEUELOG="$LOG/m2_launch.log"
source ~/anaconda/etc/profile.d/conda.sh
conda activate kidneyus2
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT/scripts"

echo "[$(date -Is)] waiting for GPU0 queue to finish..." >> "$LOG/m2_rerun.log"
while ! grep -q "GPU0 foundation queue done" "$QUEUELOG" 2>/dev/null; do
  sleep 300
done
sleep 30  # let GPU0 memory fully release

for seed in 42 43; do
  echo "[$(date -Is)] medsam rerun seed=$seed start" >> "$LOG/m2_rerun.log"
  CUDA_VISIBLE_DEVICES=0 python m2_train_foundation.py --model medsam --seed $seed --gpu 0 --bs 1 \
    >> "$LOG/medsam_seed${seed}.log" 2>&1
  rc=$?
  echo "[$(date -Is)] medsam rerun seed=$seed exit=$rc" >> "$LOG/m2_rerun.log"
done
echo "[$(date -Is)] medsam rerun done" >> "$LOG/m2_rerun.log"
