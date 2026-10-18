#!/usr/bin/env bash
# M2: internal training on OKD, 3 model families x 3 seeds, one GPU per family.
#   GPU0: D1 YOLOv8s detector (bbox auto-derived from capsule masks)
#   GPU1: M1 lightweight U-Net
#   GPU2: M2 nnU-Net v2 2d, canonical split imposed as fold 0
set -u
ROOT="${KIDNEYUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="$ROOT/outputs/logs"
mkdir -p "$LOG"
source ~/anaconda/etc/profile.d/conda.sh
conda activate kidneyus2
export PYTHONNOUSERSITE=1
export HF_ENDPOINT=https://hf-mirror.com
cd "$ROOT/scripts"

# nnU-Net env vars
export nnUNetRaw="$ROOT/data/nnunet/nnUNet_raw"
export nnUNetPreprocessed="$ROOT/data/nnunet/nnUNet_preprocessed"
export nnUNetResults="$ROOT/data/nnunet/nnUNetResults"

echo "[$(date -Is)] M2 launch start" | tee -a "$LOG/m2_launch.log"

# ---------- GPU0: D1 YOLO, 3 seeds ----------
(
  for seed in 42 43 44; do
    echo "[$(date -Is)] YOLO seed=$seed start" >> "$LOG/yolo_seed${seed}.log"
    CUDA_VISIBLE_DEVICES=0 yolo detect train \
      model="$ROOT/data/weights/yolov8s.pt" \
      data="$ROOT/data/yolo/okd.yaml" \
      epochs=50 seed=$seed project="$ROOT/outputs/yolo_runs" \
      name="d1_seed${seed}" exist_ok=True device=0 \
      >> "$LOG/yolo_seed${seed}.log" 2>&1
    echo "[$(date -Is)] YOLO seed=$seed exit=$?" >> "$LOG/yolo_seed${seed}.log"
  done
  echo "[$(date -Is)] GPU0 YOLO all seeds done" >> "$LOG/m2_launch.log"
) &

# ---------- GPU1: U-Net, 3 seeds ----------
(
  for seed in 42 43 44; do
    echo "[$(date -Is)] U-Net seed=$seed start" >> "$LOG/m2_launch.log"
    CUDA_VISIBLE_DEVICES=1 python m2_train_unet.py --seed $seed --gpu 0 \
      >> "$LOG/unet_seed${seed}.log" 2>&1
    echo "[$(date -Is)] U-Net seed=$seed exit=$?" >> "$LOG/m2_launch.log"
  done
  echo "[$(date -Is)] GPU1 U-Net all seeds done" >> "$LOG/m2_launch.log"
) &

# ---------- GPU2: nnU-Net ----------
(
  nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity \
    >> "$LOG/nnunet_preprocess.log" 2>&1
  echo "[$(date -Is)] nnU-Net preprocess exit=$?" >> "$LOG/m2_launch.log"
  # make custom seed trainers discoverable by -tr (must live inside the package)
  PKG_TR_DIR="$(python -c 'import nnunetv2, os; print(os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"))')"
  cp nnunet_trainers.py "$PKG_TR_DIR/nnUNetTrainerSeedVariants.py"
  for seed in 42 43 44; do
    echo "[$(date -Is)] nnU-Net seed=$seed start" >> "$LOG/m2_launch.log"
    CUDA_VISIBLE_DEVICES=2 nnUNetv2_train Dataset001_KidneyUS 2d 0 -tr nnUNetTrainerSeed$seed \
      >> "$LOG/nnunet_seed${seed}.log" 2>&1
    echo "[$(date -Is)] nnU-Net seed=$seed exit=$?" >> "$LOG/m2_launch.log"
  done
  echo "[$(date -Is)] GPU2 nnU-Net all seeds done" >> "$LOG/m2_launch.log"
) &

wait
echo "[$(date -Is)] M2 all done" | tee -a "$LOG/m2_launch.log"
