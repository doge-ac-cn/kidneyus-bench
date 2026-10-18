#!/usr/bin/env bash
# E1++ retake: U-Net 512 with lr 3e-4 (the 1e-3 recipe proved unstable at 512;
# lr selected on val for this auxiliary check only — documented in RUN_REPORT).
set -u
ROOT="${KIDNEYUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="$ROOT/outputs/logs"
source ~/anaconda/etc/profile.d/conda.sh
conda activate kidneyus2
export PYTHONNOUSERSITE=1
cd "$ROOT/scripts"
echo "[$(date -Is)] waiting for M5 samus queue to finish..." >> "$LOG/m2_unet512_retake.log"
while ! grep -q "M5 samus queue done" "$LOG/m5_queue.log" 2>/dev/null; do
  sleep 300
done
sleep 30
for seed in 42 43 44; do
  echo "[$(date -Is)] unet512 lr3e-4 seed=$seed start" >> "$LOG/m2_unet512_retake.log"
  CUDA_VISIBLE_DEVICES=0 python m2_train_unet.py --seed $seed --gpu 0 --res 512 --lr 3e-4 \
    >> "$LOG/unet512lr3e-4_seed${seed}.log" 2>&1
  rc=$?
  echo "[$(date -Is)] unet512 lr3e-4 seed=$seed exit=$rc" >> "$LOG/m2_unet512_retake.log"
done
echo "[$(date -Is)] unet512 retake done" >> "$LOG/m2_unet512_retake.log"
