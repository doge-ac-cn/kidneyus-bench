#!/usr/bin/env bash
# nnU-Net v2 prediction over the exported eval inputs (run m3_export_nnunet_input.py first).
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
GPU=${GPU:-0}

for seed in 42 43 44; do
  for ds in okd trusted; do
    IN="$ROOT/scratch/nnunet_eval_input/$ds"
    OUT="$ROOT/outputs/predictions/nnunet_seed${seed}/$ds"
    mkdir -p "$OUT"
    echo "[$(date -Is)] nnunet seed=$seed $ds predict start" >> "$LOG/m3_nnunet.log"
    # fold 0 (canonical split); best checkpoint (final only exists after training ends)
    CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_predict -i "$IN" -o "$OUT" -d 001 -c 2d \
      -tr nnUNetTrainerSeed$seed -f 0 -chk checkpoint_best.pth \
      >> "$LOG/nnunet_predict_seed${seed}_${ds}.log" 2>&1
    rc=$?
    echo "[$(date -Is)] nnunet seed=$seed $ds exit=$rc" >> "$LOG/m3_nnunet.log"
  done
done
echo "[$(date -Is)] nnunet predictions done" >> "$LOG/m3_nnunet.log"
