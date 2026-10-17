#!/usr/bin/env bash
# M5 v2 chain: SAMUS fine-tune with the OFFICIAL recipe (encoder-side light
# modules + decoder trainable) x3 seeds -> SAMUS E1/E2 (P3) -> SAMUS E3
# (P1/P2/P4) -> nnU-Net 42/43 predict (checkpoint_final) + score.
set -u
ROOT="${KIDNEYUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="$ROOT/outputs/logs"
source ~/anaconda/etc/profile.d/conda.sh
conda activate kidneyus2
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export nnUNet_raw="$ROOT/data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$ROOT/data/nnunet/nnUNet_preprocessed"
export nnUNet_results="$ROOT/data/nnunet/nnUNetResults"
export nnUNetRaw="$nnUNet_raw" nnUNetPreprocessed="$nnUNet_preprocessed" nnUNetResults="$nnUNet_results"
cd "$ROOT/scripts"

for seed in 42 43 44; do
  echo "[$(date -Is)] samus v2 train seed=$seed start" >> "$LOG/m5_v2.log"
  CUDA_VISIBLE_DEVICES=0 python m5_samus.py --mode train --seed $seed --gpu 0 --bs 8 \
    >> "$LOG/samus_seed${seed}.log" 2>&1
  rc=$?
  echo "[$(date -Is)] samus v2 train seed=$seed exit=$rc" >> "$LOG/m5_v2.log"
done

echo "[$(date -Is)] samus eval E1/E2 (p3)" >> "$LOG/m5_v2.log"
CUDA_VISIBLE_DEVICES=0 python m3_run_eval.py --gpu 0 \
  --models samus --seeds 42,43,44 --datasets okd,trusted --prompts p3 \
  >> "$LOG/m3_run_eval_samus_e1e2.log" 2>&1
rc=$?
echo "[$(date -Is)] samus eval E1/E2 exit=$rc" >> "$LOG/m5_v2.log"

echo "[$(date -Is)] samus eval E3 ablation" >> "$LOG/m5_v2.log"
CUDA_VISIBLE_DEVICES=0 python m3_run_eval.py --gpu 0 \
  --models samus --seeds 42,43,44 --datasets okd,trusted \
  --prompts p1_box,p2_point,p4_box \
  >> "$LOG/m3_run_eval_samus_e3.log" 2>&1
rc=$?
echo "[$(date -Is)] samus eval E3 exit=$rc" >> "$LOG/m5_v2.log"

echo "[$(date -Is)] nnunet 42/43 predict (checkpoint_final) + score" >> "$LOG/m5_v2.log"
for seed in 42 43; do
  for ds in okd trusted; do
    IN="$ROOT/scratch/nnunet_eval_input/$ds"
    OUT="$ROOT/outputs/predictions/nnunet_seed${seed}/$ds"
    mkdir -p "$OUT"
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i "$IN" -o "$OUT" -d 001 -c 2d \
      -tr nnUNetTrainerSeed$seed -f 0 -chk checkpoint_final.pth \
      >> "$LOG/nnunet_predict_seed${seed}_${ds}.log" 2>&1
    rc=$?
    echo "[$(date -Is)] nnunet predict seed=$seed $ds exit=$rc" >> "$LOG/m5_v2.log"
  done
done
python m3_score_nnunet.py --seeds 42,43 --datasets okd >> "$LOG/m5_v2.log" 2>&1
python m3_score_nnunet.py --seeds 42,43 --datasets trusted >> "$LOG/m5_v2.log" 2>&1
echo "[$(date -Is)] M5 v2 chain DONE" >> "$LOG/m5_v2.log"
