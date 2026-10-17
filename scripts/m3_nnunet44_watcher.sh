#!/usr/bin/env bash
# nnU-Net seed44: predict + score when its training finishes (checkpoint_final appears).
set -u
ROOT="${KIDNEYUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="$ROOT/outputs/logs"
CKPT_DIR="$ROOT/data/nnunet/nnUNetResults/Dataset001_KidneyUS/nnUNetTrainerSeed44__nnUNetPlans__2d/fold_0"
source ~/anaconda/etc/profile.d/conda.sh
conda activate kidneyus2
export PYTHONNOUSERSITE=1
export nnUNet_raw="$ROOT/data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$ROOT/data/nnunet/nnUNet_preprocessed"
export nnUNet_results="$ROOT/data/nnunet/nnUNetResults"
export nnUNetRaw="$nnUNet_raw" nnUNetPreprocessed="$nnUNet_preprocessed" nnUNetResults="$nnUNet_results"
cd "$ROOT/scripts"

echo "[$(date -Is)] waiting for nnU-Net seed44 checkpoint_final..." >> "$LOG/m3_nnunet44.log"
while [ ! -f "$CKPT_DIR/checkpoint_final.pth" ]; do sleep 600; done
sleep 60

for ds in okd trusted; do
  IN="$ROOT/scratch/nnunet_eval_input/$ds"
  OUT="$ROOT/outputs/predictions/nnunet_seed44/$ds"
  mkdir -p "$OUT"
  echo "[$(date -Is)] nnunet seed=44 $ds predict start" >> "$LOG/m3_nnunet44.log"
  CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict -i "$IN" -o "$OUT" -d 001 -c 2d \
    -tr nnUNetTrainerSeed44 -f 0 -chk checkpoint_final.pth \
    >> "$LOG/nnunet_predict_seed44_${ds}.log" 2>&1
  rc=$?
  echo "[$(date -Is)] nnunet seed=44 $ds exit=$rc" >> "$LOG/m3_nnunet44.log"
done
echo "[$(date -Is)] scoring seed44" >> "$LOG/m3_nnunet44.log"
python m3_score_nnunet.py --seeds 44 --datasets okd,trusted >> "$LOG/m3_nnunet44.log" 2>&1
echo "[$(date -Is)] nnunet44 chain done" >> "$LOG/m3_nnunet44.log"
