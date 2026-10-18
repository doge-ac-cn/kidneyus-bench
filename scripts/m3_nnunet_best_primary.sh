#!/usr/bin/env bash
# T1-2 primary arm: nnU-Net predictions with checkpoint_best scored into the
# PRIMARY CSVs. Runs after the M5 v2 chain drains GPU0.
set -u
ROOT="${KIDNEYUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="$ROOT/outputs/logs"
source ~/anaconda/etc/profile.d/conda.sh
conda activate kidneyus2
export PYTHONNOUSERSITE=1
export nnUNet_raw="$ROOT/data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$ROOT/data/nnunet/nnUNet_preprocessed"
export nnUNet_results="$ROOT/data/nnunet/nnUNetResults"
export nnUNetRaw="$nnUNet_raw" nnUNetPreprocessed="$nnUNet_preprocessed" nnUNetResults="$nnUNet_results"
cd "$ROOT/scripts"

echo "[$(date -Is)] waiting for M5 v2 chain DONE..." >> "$LOG/m3_nnunet_best.log"
while ! grep -q "M5 v2 chain DONE" "$LOG/m5_v2.log" 2>/dev/null; do
  sleep 300
done
sleep 30

for seed in 42 43 44; do
  for ds in okd trusted; do
    IN="$ROOT/scratch/nnunet_eval_input/$ds"
    OUT="$ROOT/outputs/predictions/nnunet_seed${seed}_best/$ds"
    mkdir -p "$OUT"
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i "$IN" -o "$OUT" -d 001 -c 2d \
      -tr nnUNetTrainerSeed$seed -f 0 -chk checkpoint_best.pth \
      >> "$LOG/nnunet_predict_best_seed${seed}_${ds}.log" 2>&1
    rc=$?
    echo "[$(date -Is)] nnunet best seed=$seed $ds exit=$rc" >> "$LOG/m3_nnunet_best.log"
  done
done
python m3_score_nnunet.py --seeds 42,43,44 --datasets okd \
  --chk checkpoint_best --tag best \
  --pred-base "$ROOT/outputs/predictions/nnunet_seed{seed}_best" >> "$LOG/m3_nnunet_best.log" 2>&1
python m3_score_nnunet.py --seeds 42,43,44 --datasets trusted \
  --chk checkpoint_best --tag best \
  --pred-base "$ROOT/outputs/predictions/nnunet_seed{seed}_best" >> "$LOG/m3_nnunet_best.log" 2>&1
echo "[$(date -Is)] nnunet best primary scoring done" >> "$LOG/m3_nnunet_best.log"
