#!/usr/bin/env bash
# Re-run SAMUS light FT x3 seeds after the unet512 lr3e-4 retake drains GPU0.
set -u
ROOT="${KIDNEYUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="$ROOT/outputs/logs"
source ~/anaconda/etc/profile.d/conda.sh
conda activate kidneyus2
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT/scripts"
echo "[$(date -Is)] waiting for unet512 retake to finish..." >> "$LOG/m5_requeue.log"
while ! grep -q "unet512 retake done" "$LOG/m2_unet512_retake.log" 2>/dev/null; do
  sleep 180
done
sleep 30
for seed in 42 43 44; do
  echo "[$(date -Is)] samus requeue seed=$seed start" >> "$LOG/m5_requeue.log"
  CUDA_VISIBLE_DEVICES=0 python m5_samus.py --mode train --seed $seed --gpu 0 \
    >> "$LOG/samus_seed${seed}.log" 2>&1
  rc=$?
  echo "[$(date -Is)] samus requeue seed=$seed exit=$rc" >> "$LOG/m5_requeue.log"
done
echo "[$(date -Is)] samus requeue done" >> "$LOG/m5_requeue.log"
