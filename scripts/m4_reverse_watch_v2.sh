#!/usr/bin/env bash
# M4 v2 continuation: fold0 trainers were orphaned after the original chain
# subshells were killed (they would have retrained fold1 from scratch, clobbering
# the GPU0 fold1 pair started 2026-10-11 01:59). Static assignment, collision-free:
#   GPU1 (2d)   : wait fold0 trainer -> folds 2,3,4        -> sweep fold1 if untrained
#   GPU2 (3d)   : wait fold0 trainer -> folds 2,3,4        -> sweep fold1 if untrained
#   GPU0        : 2d fold1 + 3d fold1 (already running, untouched)
set -u
ROOT="${KIDNEYUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="$ROOT/outputs/logs"
RES="$ROOT/data/nnunet/nnUNetResults/Dataset002_TRUSTED"
source ~/anaconda/etc/profile.d/conda.sh
conda activate kidneyus2
export PYTHONNOUSERSITE=1
export nnUNet_raw="$ROOT/data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="$ROOT/data/nnunet/nnUNet_preprocessed"
export nnUNet_results="$ROOT/data/nnunet/nnUNetResults"
export nnUNetRaw="$nnUNet_raw" nnUNetPreprocessed="$nnUNet_preprocessed" nnUNetResults="$nnUNet_results"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date -Is)] $*" >> "$LOG/m4_reverse_v2.log"; }

fold_done() { [ -f "$RES/nnUNetTrainer__nnUNetPlans__$1/fold_$2/checkpoint_final.pth" ]; }

run_fold() { # $1=cfg $2=fold $3=gpu
  local cfg=$1 fold=$2 gpu=$3 out="$RES/nnUNetTrainer__nnUNetPlans__$1/fold_$2"
  if fold_done "$cfg" "$fold"; then log "$cfg f$fold skip (final exists)"; return 0; fi
  local attempt rc
  for attempt in 1 2; do
    CUDA_VISIBLE_DEVICES=$gpu nnUNetv2_train Dataset002_TRUSTED "$cfg" "$fold" --npz --c \
      >> "$LOG/m4v2_${cfg}_f${fold}.log" 2>&1
    rc=$?
    log "$cfg fold=$fold attempt=$attempt exit=$rc"
    fold_done "$cfg" "$fold" && return 0
    log "$cfg f$fold no checkpoint_final, retrying" ; sleep 60
  done
  log "$cfg fold=$fold FAILED after 2 attempts"; return 1
}

sweep_fold1() { # $1=cfg $2=gpu — only if GPU0's fold1 never finished
  local cfg=$1 gpu=$2
  fold_done "$cfg" 1 && return 0
  if pgrep -f "nnUNetv2_train Dataset002_TRUSTED $cfg 1 " >/dev/null 2>&1; then
    log "$cfg f1 still training elsewhere, skip"; return 0
  fi
  log "$cfg f1 missing, training on GPU$gpu"
  run_fold "$cfg" 1 "$gpu"
}

watcher() { # $1=cfg $2=gpu $3=fold0_trainer_pid
  local cfg=$1 gpu=$2 pid=$3 f
  while kill -0 "$pid" 2>/dev/null; do sleep 120; done
  log "$cfg fold0 trainer ($pid) exited; fold0 done=$(fold_done "$cfg" 0 && echo yes || echo no)"
  fold_done "$cfg" 0 || run_fold "$cfg" 0 "$gpu"
  for f in 2 3 4; do run_fold "$cfg" "$f" "$gpu"; done
  sweep_fold1 "$cfg" "$gpu"
  log "$cfg chain COMPLETE: folds done = $(for f in 0 1 2 3 4; do fold_done "$cfg" $f && echo -n "$f "; done)"
}

log "M4 v2 watcher start (2d GPU1 <- pid $1 | 3d GPU2 <- pid $2)"
watcher 2d 1 "$1" &
P1=$!
watcher 3d_fullres 2 "$2" &
P2=$!
wait $P1 $P2
log "M4 v2 ALL DONE"
