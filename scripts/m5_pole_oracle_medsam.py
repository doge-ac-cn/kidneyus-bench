#!/usr/bin/env python3
"""P2-1b: MedSAM p3 (oracle box) per-slice predictions stratified by the
canonical pole-slice definition -> outputs/summary/pole_stratification_oracle.json.

Same shape as pole_stratification.json; pooled across the 3 registered seeds
(each seed uses its own fine-tuned weights, matching how the E2 p3 rows were
produced). Oracle box = largest-CC bbox of the GT (registered m3_run_eval
convention). Ground truth enters the registered scorer only.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from m3_common import iter_trusted, load_sam_family, score, \
    infer_sam_family, MODEL_RES, oracle_box  # noqa: E402

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (42, 43, 44)

canon = json.loads((ROOT / "outputs/summary/canonical_pole_slices.json").read_text())
pole = set()
for kid, zs in canon["canonical_pole_slices"].items():
    for z in zs:
        pole.add(f"{kid}_z{z:03d}")

out_p = ROOT / "outputs/summary/pole_stratification_oracle.json"
done = json.loads(out_p.read_text())["models"] if out_p.exists() else {}
out = {"definition": canon["definition"] + "; oracle box = registered p3 pathway",
       "models": dict(done)}  # 断点续跑：保留已完成的 seed 结果

for seed in SEEDS:
    if str(seed) in done.get("models", {}):
        print(f"[skip] seed{seed} already scored", flush=True)
        continue
    model = load_sam_family("medsam", seed, DEVICE)
    res = MODEL_RES["medsam"]
    dp, dr = [], []
    n = 0
    for cid, img, gt, spacing, _meta in iter_trusted():
        box = oracle_box(gt)
        pred = infer_sam_family(model, img, res, DEVICE, box=box)
        d, _h = score(pred.astype(bool), gt, spacing)
        (dp if cid in pole else dr).append(float(d))
        n += 1
        if n % 4000 == 0:
            print(f"  seed{seed}: {n} slices (pole so far {len(dp)})", flush=True)
    del model
    torch.cuda.empty_cache()
    out["models"][str(seed)] = {
        "pole_dice_mean": round(float(np.mean(dp)), 4), "pole_n": len(dp),
        "rest_dice_mean": round(float(np.mean(dr)), 4), "rest_n": len(dr)}
    out_p.write_text(json.dumps(out, indent=1))
    print(f"[oracle] seed{seed}: pole={out['models'][str(seed)]['pole_dice_mean']} "
          f"rest={out['models'][str(seed)]['rest_dice_mean']} (n={len(dp)}/{len(dr)})", flush=True)

pm = [out["models"][str(s)]["pole_dice_mean"] for s in SEEDS if str(s) in out["models"]]
rm = [out["models"][str(s)]["rest_dice_mean"] for s in SEEDS if str(s) in out["models"]]
if len(pm) == len(SEEDS):
    out["pooled_3seed"] = {"pole_dice_mean": round(float(np.mean(pm)), 4),
                           "rest_dice_mean": round(float(np.mean(rm)), 4)}
    out_p.write_text(json.dumps(out, indent=1))  # 仅三种子齐时才覆盖产物
print(json.dumps(out.get("pooled_3seed"), indent=1), flush=True)
