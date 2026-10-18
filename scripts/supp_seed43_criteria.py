#!/usr/bin/env python3
"""Seed-43 primary: §4.4 criteria recomputation (round-level means) + P5 numbers.

§4.4 (frozen rule, now on the registered model seed 43):
  (i)   external no-box rate < 25%           -> detection dumps (unchanged)
  (ii)  conditional Dice >= 0.75             -> per model, mean over the 3
                                                detector replicates
  (iii) strict Dice >= 0.55                  -> likewise
  (iv)  in-domain OKD non-regression         -> detector-level (unchanged)

P5 (amendment section 3, registered seed): nnU-Net seed-43 prior boxes ->
MedSAM seed-43 single pass; coverage / box-IoU-vs-oracle median / strict Dice.

Outputs supp_tta_criteria_seed43.json, supp_p5_priorbox_seed43.json.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
MET = ROOT / "outputs/metrics"
OUT = ROOT / "outputs/summary"

REPS = {"medsam": {42: "rev43_medsam43_tta42.csv",
                   43: "rev43_medsam43_tta.csv",
                   44: "rev43_medsam43_tta44.csv"},
        "sam_lora": {42: "rev43_samlora43_tta42.csv",
                     43: "rev43_samlora43_tta.csv",
                     44: "rev43_samlora43_tta44.csv"},
        "samus": {42: "rev43_samus43b_tta42.csv",
                  43: "rev_samus43_tta_r2s43.csv",
                  44: "rev43_samus43b_tta44.csv"}}


def rd(fn):
    df = pd.read_csv(MET / fn)
    assert len(df) == 18322, (fn, len(df))
    return df


def main():
    # ---- §4.4 (ii)/(iii) round-level means under seed 43 ----
    crit = {"caliber": "registered seed 43 (model weights); round-level means "
                       "across the three detector replicates",
            "per_model": {}}
    for m, reps in REPS.items():
        per_rep = {}
        for rep, fn in reps.items():
            df = rd(fn)
            det = df[df["pred_area_px"] > 0]
            per_rep[str(rep)] = {"strict": round(float(df["dice"].mean()), 4),
                                 "conditional": round(float(det["dice"].mean()), 4),
                                 "coverage": round(float(len(det) / len(df)), 4)}
        crit["per_model"][m] = {
            "replicates": per_rep,
            "round_mean": {k: round(float(np.mean([per_rep[str(r)][k] for r in reps])), 4)
                           for k in ("strict", "conditional", "coverage")}}
    # (i) detection no-box (detector seed fixed at registration; unchanged)
    det_j = json.loads((OUT / "tta_r2_detection.json").read_text())
    crit["i_no_box_rate"] = det_j["round2_mean_no_box"]
    crit["i_pass"] = crit["i_no_box_rate"] < 0.25
    crit["ii_pass_all_models"] = all(v["round_mean"]["conditional"] >= 0.75
                                     for v in crit["per_model"].values())
    crit["iii_pass_all_models"] = all(v["round_mean"]["strict"] >= 0.55
                                      for v in crit["per_model"].values())
    verdict = ("ADOPT" if (crit["i_pass"] and crit["ii_pass_all_models"]
                           and crit["iii_pass_all_models"]) else
               "criteria fail (see per-model values)")
    crit["verdict_on_i_ii_iii"] = verdict
    (OUT / "supp_tta_criteria_seed43.json").write_text(json.dumps(crit, indent=1))
    print("criteria:", json.dumps({m: crit["per_model"][m]["round_mean"]
                                   for m in REPS}, indent=1), flush=True)

    # ---- P5 numbers (seed 43) ----
    p5 = rd("rev43_medsam43_p5s43.csv")
    gt = np.load(ROOT / "outputs/prompts/p3_oracle_trusted.npz")
    gt_of = {str(c): np.asarray(b, float) for c, b in zip(gt["case_ids"], gt["boxes"])}
    p5b = np.load(ROOT / "outputs/prompts/p5_box_trusted_s43.npz")
    p5_of = {str(c): np.asarray(b, float) for c, b in zip(p5b["case_ids"], p5b["boxes"])}
    ious = []
    for cid, b in p5_of.items():
        if np.isnan(b).any() or cid not in gt_of:
            continue
        a, g = b, gt_of[cid]
        iw = max(min(a[2], g[2]) - max(a[0], g[0]), 0)
        ih = max(min(a[3], g[3]) - max(a[1], g[1]), 0)
        inter = iw * ih
        union = (a[2]-a[0])*(a[3]-a[1]) + (g[2]-g[0])*(g[3]-g[1]) - inter
        ious.append(inter / union if union > 0 else 0.0)
    coverage = float((p5["pred_area_px"] > 0).mean())
    iou_med = float(np.median(ious)) if ious else float("nan")
    dice = float(p5["dice"].mean())
    res = {"caliber": "registered seed 43: nnU-Net seed-43 prior boxes -> MedSAM seed-43",
           "n": len(p5), "coverage": round(coverage, 4),
           "box_iou_vs_oracle_median": round(iou_med, 4),
           "strict_dice_mean": round(dice, 4),
           "criterion": {"coverage>=0.75": coverage >= 0.75,
                         "iou_median>=0.70": bool(iou_med >= 0.70),
                         "dice>=0.55": dice >= 0.55},
           "verdict": "viable alternative prompt source" if (
               coverage >= 0.75 and iou_med >= 0.70 and dice >= 0.55)
               else "diagnostic/negative"}
    (OUT / "supp_p5_priorbox_seed43.json").write_text(json.dumps(res, indent=1))
    print("P5:", json.dumps(res, indent=1), flush=True)
    print("SEED43 CRITERIA DONE", flush=True)


if __name__ == "__main__":
    main()
