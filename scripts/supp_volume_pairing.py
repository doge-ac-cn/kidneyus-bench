#!/usr/bin/env python3
"""Item ②: US-vs-CT volumetric error, with explicit pairing rule, hierarchy,
and patient-cluster bootstrap CI on the bias.

Pairing rule (documented):
  - CT reference exists for 48 patients (both kidneys); the US sub-cohort is
    39 of them; every US kidney (59) maps to its same-patient, same-laterality
    CT component (R = image-left/smaller-x centroid, radiological convention).
  - Level: per-kidney pair (n=59); uncertainty resampled at the PATIENT level
    (39 clusters) with both kidneys resampled together.
  - US volume = summed per-slice predicted area x in-plane pixel area x
    slice spacing (spacing read from the mask headers: 0.3 x 0.3 x 0.3 mm).
  - Undetected slices contribute zero volume (registered empty-prompt rule);
    TTA-R2 no-box slice rate 8.3%, clustered near poles -> US volumes biased
    downward in that arm; the reported bias is therefore conservative for it.

Outputs: outputs/summary/volume_bland_altman.json (updated) +
         outputs/summary/supp_volume_pairing.md
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "outputs/summary"
N_BOOT = 2000


def boot_bias(pairs, n=N_BOOT, seed=42):
    """pairs: DataFrame with patient, us_ml, ct_ml; patient-cluster bootstrap."""
    rng = np.random.default_rng(seed)
    pats = pairs["patient"].unique()
    stats = np.empty(n)
    for it in range(n):
        pick = rng.choice(pats, len(pats), replace=True)
        sub = pairs[pairs["patient"].isin(pick)]
        stats[it] = (sub["us_ml"] - sub["ct_ml"]).mean()
    return [round(float(np.percentile(stats, 2.5)), 2),
            round(float(np.percentile(stats, 97.5)), 2)]


def pair_frame(us_ml_by_kidney, patient_of, ctv):
    rows = []
    for kid, us_ml in us_ml_by_kidney.items():
        pid = patient_of(kid)
        if pid not in ctv:
            continue
        side = kid[3]
        ref = ctv[pid]["ct_R_ml"] if side == "R" else ctv[pid]["ct_L_ml"]
        rows.append({"kidney_id": kid, "patient": pid, "us_ml": us_ml, "ct_ml": ref})
    return pd.DataFrame(rows)


def main():
    ctv = json.loads((OUT / "ct_reference_volumes.json").read_text())
    usv = json.loads((OUT / "us_volumes.json").read_text())
    vba = json.loads((OUT / "volume_bland_altman.json").read_text())

    # nnU-Net zero-shot pair frame
    f0 = pair_frame({k: v["pred_ml"] for k, v in usv.items()}, lambda k: k[:3], ctv)
    fg = pair_frame({k: v["gt_ml"] for k, v in usv.items()}, lambda k: k[:3], ctv)
    for f, key in ((f0, "nnunet_zeroshot_vs_ct"), (fg, "us_gt_vs_ct")):
        vba[key]["bias_ci95_ml_patient39"] = boot_bias(f)
        vba[key]["n_patients"] = int(f["patient"].nunique())
        vba[key]["n_kidney_pairs"] = int(len(f))
        vba[key]["level"] = "per-kidney pair; patient-clustered CI"

    # TTA-R2 pair frame from its per-kidney volumes in the existing entry
    r2 = ROOT / "outputs/metrics/rev_medsam_tta_r2s43.csv"
    if r2.exists():
        t = pd.read_csv(r2)
        per = t.groupby(["kidney_id", "patient"])["pred_area_px"].sum().reset_index()
        per["patient"] = per["patient"].astype(str)
        vol = {r["kidney_id"]: r["pred_area_px"] * 0.09 * 0.3 / 1000.0
               for _, r in per.iterrows()}
        f2 = pair_frame({k: v for k, v in vol.items()}, lambda k: str(k[:3]), ctv)
        if len(f2):
            d = f2["us_ml"] - f2["ct_ml"]
            bias = float(d.mean())
            loa = [round(bias - 1.96 * float(d.std(ddof=1)), 2),
                   round(bias + 1.96 * float(d.std(ddof=1)), 2)]
            vba["tta_r2_vs_ct"] = {"config": "TTA-R2 (MedSAM seed42, r2_s43 detector)",
                                   "n_kidney_pairs": int(len(f2)),
                                   "n_patients": int(f2["patient"].nunique()),
                                   "bias_ml": round(bias, 2),
                                   "bias_ci95_ml_patient39": boot_bias(f2),
                                   "loa95_ml": loa,
                                   "mean_us_pred_ml": round(float(f2["us_ml"].mean()), 2),
                                   "level": "per-kidney pair; patient-clustered CI",
                                   "note": vba.get("tta_r2_vs_ct", {}).get("note", "")}
    vba["pairing_rule"] = ("US kidney -> same-patient same-laterality CT component "
                           "(GT_estimated_masksCT, R = smaller-x centroid); US volume = "
                           "sum of per-slice predicted area x 0.09 mm2 x 0.3 mm; "
                           "level = per-kidney (n=59); CI = patient-39 cluster bootstrap; "
                           "US sub-cohort = 39 of the 48 CT patients")
    (OUT / "volume_bland_altman.json").write_text(json.dumps(vba, indent=1))

    md = ["# Volumetric error (Supplementary analysis (volumetric error))\n",
          vba["pairing_rule"], "",
          "| arm | n pairs | bias mL | 95% CI (patient) | 95% LoA |",
          "|---|---|---|---|---|"]
    for k in ("nnunet_zeroshot_vs_ct", "tta_r2_vs_ct", "us_gt_vs_ct"):
        v = vba.get(k, {})
        if not v:
            continue
        md.append(f"| {k} | {v.get('n_kidney_pairs', v.get('n'))} | {v.get('bias_ml')} | "
                  f"{v.get('bias_ci95_ml_patient39')} | {v.get('loa95_ml')} |")
    (OUT / "supp_volume_pairing.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
