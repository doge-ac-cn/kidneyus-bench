#!/usr/bin/env bash
# GPU0 follow-up chain after the SAMUS requeue drains:
#   1. SAMUS eval: E1/E2 (P3) + E3 ablation (P1/P2/P4) for 3 seeds
#   2. nnU-Net 42/43 predict (checkpoint_final) + score — the gpu2-watcher
#      attempts failed on missing env/argparse (fixed 2026-10-09)
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

echo "[$(date -Is)] waiting for samus requeue done..." >> "$LOG/m3_gpu0_followup.log"
while ! grep -q "samus requeue done" "$LOG/m5_requeue.log" 2>/dev/null; do
  sleep 180
done
sleep 30

echo "[$(date -Is)] stepA: samus eval E1/E2 (p3)" >> "$LOG/m3_gpu0_followup.log"
CUDA_VISIBLE_DEVICES=0 python m3_run_eval.py --gpu 0 \
  --models samus --seeds 42,43,44 --datasets okd,trusted --prompts p3 \
  >> "$LOG/m3_run_eval_samus_e1e2.log" 2>&1
rc=$?
echo "[$(date -Is)] stepA exit=$rc" >> "$LOG/m3_gpu0_followup.log"

echo "[$(date -Is)] stepB: samus eval E3 ablation" >> "$LOG/m3_gpu0_followup.log"
CUDA_VISIBLE_DEVICES=0 python m3_run_eval.py --gpu 0 \
  --models samus --seeds 42,43,44 --datasets okd,trusted \
  --prompts p1_box,p2_point,p4_box \
  >> "$LOG/m3_run_eval_samus_e3.log" 2>&1
rc=$?
echo "[$(date -Is)] stepB exit=$rc" >> "$LOG/m3_gpu0_followup.log"

echo "[$(date -Is)] stepC: nnunet 42/43 predict (checkpoint_final) + score" >> "$LOG/m3_gpu0_followup.log"
for seed in 42 43; do
  for ds in okd trusted; do
    IN="$ROOT/scratch/nnunet_eval_input/$ds"
    OUT="$ROOT/outputs/predictions/nnunet_seed${seed}/$ds"
    mkdir -p "$OUT"
    CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict -i "$IN" -o "$OUT" -d 001 -c 2d \
      -tr nnUNetTrainerSeed$seed -f 0 -chk checkpoint_final.pth \
      >> "$LOG/nnunet_predict_seed${seed}_${ds}.log" 2>&1
    rc=$?
    echo "[$(date -Is)] nnunet predict seed=$seed $ds exit=$rc" >> "$LOG/m3_gpu0_followup.log"
  done
done
python m3_score_nnunet.py --seeds 42,43 --datasets okd >> "$LOG/m3_gpu0_followup.log" 2>&1
python m3_score_nnunet.py --seeds 42,43 --datasets trusted >> "$LOG/m3_gpu0_followup.log" 2>&1
echo "[$(date -Is)] GPU0 follow-up DONE" >> "$LOG/m3_gpu0_followup.log"
