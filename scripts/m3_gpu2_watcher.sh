#!/usr/bin/env bash
# GPU2 M3 formal-eval watcher: fires when nnU-Net seed43 training exits,
# runs the ready-model M3 batches in GPU2's idle window:
#   1. nnU-Net predict for seeds 42,43 (OKD + TRUSTED inputs, best ckpt)
#   2. score nnU-Net predictions -> e1/e2 rows
#   3. m3_run_eval E1+E2 (P3) for unet/sam_lora/medsam 3 seeds
#   4. stage1 prompts (P1/P2/P4 derivation + D1 detection table)
#   5. m3_run_eval E3 ablation (P1/P2/P4) for sam_lora/medsam
# (nnU-Net seed44 + samus + unet512-retake are chained elsewhere)
set -u
ROOT="${KIDNEYUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="$ROOT/outputs/logs"
source ~/anaconda/etc/profile.d/conda.sh
conda activate kidneyus2
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT/scripts"
GPU=2

echo "[$(date -Is)] watching for nnU-Net seed43 (pid 1888596) exit..." >> "$LOG/m3_gpu2.log"
while kill -0 1888596 2>/dev/null; do sleep 60; done
sleep 60  # memory release

echo "[$(date -Is)] step1: nnunet predict 42,43" >> "$LOG/m3_gpu2.log"
for seed in 42 43; do
  for ds in okd trusted; do
    IN="$ROOT/scratch/nnunet_eval_input/$ds"
    OUT="$ROOT/outputs/predictions/nnunet_seed${seed}/$ds"
    mkdir -p "$OUT"
    CUDA_VISIBLE_DEVICES=$GPU nnUNetv2_predict -i "$IN" -o "$OUT" -d 001 -c 2d \
      -tr nnUNetTrainerSeed$seed -f 0 -chk checkpoint_best.pth \
      >> "$LOG/nnunet_predict_seed${seed}_${ds}.log" 2>&1
    rc=$?
    echo "[$(date -Is)] nnunet predict seed=$seed $ds exit=$rc" >> "$LOG/m3_gpu2.log"
  done
done

echo "[$(date -Is)] step2: score nnunet" >> "$LOG/m3_gpu2.log"
python m3_score_nnunet.py --seeds 42,43 --datasets okd,trusted >> "$LOG/m3_gpu2.log" 2>&1

echo "[$(date -Is)] step3: run_eval E1+E2 (p3)" >> "$LOG/m3_gpu2.log"
CUDA_VISIBLE_DEVICES=$GPU python m3_run_eval.py --gpu 0 \
  --models unet,sam_lora,medsam --seeds 42,43,44 --datasets okd,trusted --prompts p3 \
  >> "$LOG/m3_run_eval_e1e2.log" 2>&1
echo "[$(date -Is)] step3 exit=$?" >> "$LOG/m3_gpu2.log"

echo "[$(date -Is)] step4: stage1 prompts" >> "$LOG/m3_gpu2.log"
CUDA_VISIBLE_DEVICES=$GPU python m3_stage1_prompts.py --datasets okd,trusted --device cuda \
  >> "$LOG/m3_stage1.log" 2>&1
echo "[$(date -Is)] step4 exit=$?" >> "$LOG/m3_gpu2.log"

echo "[$(date -Is)] step5: run_eval E3 ablation" >> "$LOG/m3_gpu2.log"
CUDA_VISIBLE_DEVICES=$GPU python m3_run_eval.py --gpu 0 \
  --models sam_lora,medsam --seeds 42,43,44 --datasets okd,trusted \
  --prompts p1_box,p2_point,p4_box \
  >> "$LOG/m3_run_eval_e3.log" 2>&1
echo "[$(date -Is)] step5 exit=$?" >> "$LOG/m3_gpu2.log"
echo "[$(date -Is)] GPU2 M3 batch DONE" >> "$LOG/m3_gpu2.log"
