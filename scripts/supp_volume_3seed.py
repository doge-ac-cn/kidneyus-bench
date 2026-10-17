#!/usr/bin/env python3
"""Item ② (extension): 3-seed nnU-Net US volume bias vs CT (the main volume
number was seed42; this reports the 3-seed caliber for the supplementary analyses).

Reads the registered best-checkpoint predictions of seeds 42/43/44
(outputs/predictions/nnunet_seed{N}_best/trusted), sums per-kidney predicted
area, converts with the header spacing, and reports per-seed and 3-seed-mean
bias vs the CT reference (patient-cluster CI on the 3-seed-mean bias).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "outputs/summary"


def main():
    sl = pd.read_csv(ROOT / "outputs/trusted_us_slices.csv")
    sk = {}
    for kid, g in sl.groupby("kidney_id", sort=True):
        sk[kid] = [int(z) for z in g["slice_idx"]]
    ctv = json.loads((OUT / "ct_reference_volumes.json").read_text())
    sp = sitk.ReadImage(str(ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA/US_masks/"
                                "GT_estimated_masksUS/200R_maskUS.nii.gz")).GetSpacing()
    mm3 = sp[0] * sp[1] * sp[2]
    rows = []
    for seed in (42, 43, 44):
        pd_dir = ROOT / f"outputs/predictions/nnunet_seed{seed}_best/trusted"
        vols = {}
        for kid, zs in sk.items():
            v = 0
            for z in zs:
                p = pd_dir / f"{kid}_z{z}.nii.gz"
                if p.exists():
                    v += int((sitk.GetArrayFromImage(sitk.ReadImage(str(p))) > 0).sum())
            vols[kid] = v * mm3 / 1000.0
        for kid, usml in vols.items():
            pid = kid[:3]
            if pid not in ctv:
                continue
            ref = ctv[pid]["ct_R_ml"] if kid[3] == "R" else ctv[pid]["ct_L_ml"]
            rows.append({"seed": seed, "kidney_id": kid, "patient": pid,
                         "us_ml": usml, "ct_ml": ref})
    df = pd.DataFrame(rows)
    per_seed = {int(s): round(float((g["us_ml"] - g["ct_ml"]).mean()), 2)
                for s, g in df.groupby("seed")}
    # 3-seed mean volume per kidney, then bias + patient-cluster CI
    kmean = df.groupby(["kidney_id", "patient"]).agg(
        us_ml=("us_ml", "mean"), ct_ml=("ct_ml", "first")).reset_index()
    d = kmean["us_ml"] - kmean["ct_ml"]
    bias = float(d.mean())
    rng = np.random.default_rng(42)
    pats = kmean["patient"].unique()
    stats = np.empty(2000)
    for it in range(2000):
        pick = rng.choice(pats, len(pats), replace=True)
        sub = kmean[kmean["patient"].isin(pick)]
        stats[it] = (sub["us_ml"] - sub["ct_ml"]).mean()
    res = {"per_seed_bias_ml": per_seed,
           "three_seed_mean_bias_ml": round(bias, 2),
           "three_seed_bias_ci95_patient39": [round(float(np.percentile(stats, 2.5)), 2),
                                              round(float(np.percentile(stats, 97.5)), 2)],
           "loa95_ml": [round(bias - 1.96 * float(d.std(ddof=1)), 2),
                        round(bias + 1.96 * float(d.std(ddof=1)), 2)],
           "n_kidney_pairs": int(len(kmean)), "n_patients": int(kmean["patient"].nunique()),
           "caliber": "per-kidney pair; 3-seed mean predicted volume; patient-clustered CI"}
    vba = json.loads((OUT / "volume_bland_altman.json").read_text())
    vba["nnunet_zeroshot_vs_ct_3seed"] = res
    (OUT / "volume_bland_altman.json").write_text(json.dumps(vba, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
