#!/usr/bin/env python3
"""B1: pole-slice annotator A/B consistency (the "convention vs sloppiness"
adjudication).

For every scored slice of every kidney: Dice(Annotator1_z > 0.5, Annotator2_z
> 0.5), split by the canonical pole definition (GT area < 15% of the kidney's
median scored-slice area, outputs/summary/canonical_pole_slices.json).
If A and B agree with each other at poles (IoD >= 0.6), the tiny pole
cross-sections are a shared annotation CONVENTION; if not, re-word.

By-product: per-slice A-vs-B Dice dump (outputs/metrics/iod_ab_per_slice.csv,
gitignored) reused by B3's joint bootstrap.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
MASKS = ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA/US_masks"


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else 1.0  # Dice = 2|A∩B|/(|A|+|B|)


def main():
    canon = json.loads((ROOT / "outputs/summary/canonical_pole_slices.json").read_text())
    pole = {(kid, int(z)) for kid, zs in canon["canonical_pole_slices"].items() for z in zs}

    sl = pd.read_csv(ROOT / "outputs/trusted_us_slices.csv")
    slices_by_kidney = {}
    for kid, z, area in zip(sl["kidney_id"], sl["slice_idx"], sl["area_gt"]):
        slices_by_kidney.setdefault(kid, []).append((int(z), float(area)))

    dump = open(ROOT / "outputs/metrics/iod_ab_per_slice.csv", "w")
    dump.write("kidney_id,patient,slice_idx,dice_ab,is_pole,area_gt\n")
    pole_d, rest_d = [], []
    per_kidney = {}
    n_kid = 0
    for kid, zlist in sorted(slices_by_kidney.items()):
        a_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(MASKS / "Annotator1" / f"{kid}1_maskUS.nii.gz"))) > 0.5
        b_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(MASKS / "Annotator2" / f"{kid}2_maskUS.nii.gz"))) > 0.5
        pk, rk = [], []
        for z, area in zlist:
            d = dice(a_vol[z], b_vol[z])
            isp = (kid, z) in pole
            dump.write(f"{kid},{kid[:3]},{z},{d:.6f},{int(isp)},{area:.0f}\n")
            (pk if isp else rk).append(d)
        pole_d += pk
        rest_d += rk
        per_kidney[kid] = {"n_pole": len(pk), "n_rest": len(rk),
                           "pole_mean": round(float(np.mean(pk)), 4) if pk else None,
                           "rest_mean": round(float(np.mean(rk)), 4) if rk else None}
        n_kid += 1
        if n_kid % 10 == 0:
            print(f"  {n_kid}/59 kidneys", flush=True)
    dump.close()

    def summ(vals):
        v = np.array(vals)
        return {"n": int(len(v)), "mean": round(float(v.mean()), 4),
                "median": round(float(np.median(v)), 4), "sd": round(float(v.std()), 4),
                "pct_ge_0.6": round(float((v >= 0.6).mean()), 4),
                "pct_ge_0.7": round(float((v >= 0.7).mean()), 4),
                "pct_ge_0.8": round(float((v >= 0.8).mean()), 4)}

    out = {"definition": "canonical pole slices (GT area < 15% of kidney median scored-slice area); "
                         "Dice(Annotator1, Annotator2) per slice, binarized >0.5",
           "pole": summ(pole_d), "non_pole": summ(rest_d),
           "verdict_rule": "pole A/B IoD >= 0.6 => shared annotation convention",
           "pole_ge_0.6_verdict": "annotation convention (A/B mutually consistent)"
                                  if np.mean(pole_d) >= 0.6 else
                                  "A/B inconsistent at poles - re-word required",
           "per_kidney": per_kidney}
    (ROOT / "outputs/summary/pole_annotator_consistency.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("pole", "non_pole", "pole_ge_0.6_verdict")}, indent=1))


if __name__ == "__main__":
    main()
