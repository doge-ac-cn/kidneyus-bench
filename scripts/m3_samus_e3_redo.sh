#!/usr/bin/env bash
# SAMUS E3 redo: the v2 chain's E3 step fires before stage1 produces the prompt
# npz files (and crashes on np.load). This waits for BOTH the v2 chain marker
# AND the six npz files, then runs the SAMUS E3 ablation on GPU0.
set -u
ROOT="${KIDNEYUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG="$ROOT/outputs/logs"
source ~/anaconda/etc/profile.d/conda.sh
conda activate kidneyus2
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT/scripts"

echo "[$(date -Is)] waiting for v2 chain DONE + stage1 npz..." >> "$LOG/m3_samus_e3_redo.log"
while true; do
  done_marker=$(grep -c "M5 v2 chain DONE" "$LOG/m5_v2.log" 2>/dev/null)
  done_marker=${done_marker:-0}
  npz=$(ls "$ROOT/outputs/prompts/"p1_box_*.npz "$ROOT/outputs/prompts/"p2_point_*.npz "$ROOT/outputs/prompts/"p4_box_*.npz 2>/dev/null | wc -l)
  if [ "$done_marker" -ge 1 ] && [ "$npz" -ge 6 ]; then break; fi
  sleep 300
done
sleep 30

echo "[$(date -Is)] samus E3 redo start" >> "$LOG/m3_samus_e3_redo.log"
CUDA_VISIBLE_DEVICES=0 python m3_run_eval.py --gpu 0 \
  --models samus --seeds 42,43,44 --datasets okd,trusted \
  --prompts p1_box,p2_point,p4_box \
  >> "$LOG/m3_run_eval_samus_e3.log" 2>&1
rc=$?
echo "[$(date -Is)] samus E3 redo exit=$rc" >> "$LOG/m3_samus_e3_redo.log"

# 落表对账：预期 e1=3seed×3prompt×99=891 行、e2=3×3×18322=164898 行（samus 标签）
E1_EXPECT=891; E2_EXPECT=164898
E1_GOT=$(awk -F, '$1=="samus" && ($3=="p1"||$3=="p2"||$3=="p4")' "$ROOT/outputs/metrics/e1_internal.csv" 2>/dev/null | wc -l)
E2_GOT=$(awk -F, '$1=="samus" && ($3=="p1"||$3=="p2"||$3=="p4")' "$ROOT/outputs/metrics/e2_external.csv" 2>/dev/null | wc -l)
echo "[$(date -Is)] 对账: e1 samus消融行=$E1_GOT (预期$E1_EXPECT) | e2 samus消融行=$E2_GOT (预期$E2_EXPECT)" >> "$LOG/m3_samus_e3_redo.log"
if [ "$E1_GOT" -ne "$E1_EXPECT" ] || [ "$E2_GOT" -ne "$E2_EXPECT" ]; then
  echo "[$(date -Is)] ⚠ 对账不平——存在缺失单元，需排查" >> "$LOG/m3_samus_e3_redo.log"
fi
