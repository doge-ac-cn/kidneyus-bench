#!/usr/bin/env python3
"""Item ④ (nnU-Net leg): score the registered best-checkpoint nnU-Net TRUSTED
predictions (nii.gz on disk, seed 42) per slice with the same column set as
supp_megapass.py: dice / dice_A / dice_B / HD95 / NSD@1mm / pred_area.

Output: outputs/metrics/rev_nnunet_best.csv (18,322 rows + header).
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent))
from supp_megapass import dice, hd95, nsd  # noqa: E402

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
PRED = ROOT / "outputs/predictions/nnunet_seed42_best/trusted"
RAW = ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA"
OUT = ROOT / "outputs/metrics/rev_nnunet_best.csv"


def main():
    import pandas as pd
    sl = pd.read_csv(ROOT / "outputs/trusted_us_slices.csv")
    n = 0
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "kidney_id", "patient", "dice", "dice_A", "dice_B",
                    "hd95", "nsd", "pred_area_px"])
        for kid, g in sl.groupby("kidney_id", sort=True):
            gt_img = sitk.ReadImage(str(RAW / f"US_masks/GT_estimated_masksUS/{kid}_maskUS.nii.gz"))
            gt_vol = sitk.GetArrayFromImage(gt_img) > 0.5
            A = sitk.GetArrayFromImage(sitk.ReadImage(
                str(RAW / f"US_masks/Annotator1/{kid}1_maskUS.nii.gz"))) > 0.5
            B = sitk.GetArrayFromImage(sitk.ReadImage(
                str(RAW / f"US_masks/Annotator2/{kid}2_maskUS.nii.gz"))) > 0.5
            sp = gt_img.GetSpacing()[1:3]  # (y,x) mm  (array order = z,y,x)
            for z in g["slice_idx"]:
                z = int(z)
                p = PRED / f"{kid}_z{z}.nii.gz"
                pred = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).squeeze() > 0
                pb = pred.astype(bool)
                gt = gt_vol[z]
                h = hd95(pb, gt, sp)
                w.writerow([f"{kid}_z{z:03d}", kid, kid[:3],
                            f"{dice(pb, gt):.6f}", f"{dice(pb, A[z]):.6f}",
                            f"{dice(pb, B[z]):.6f}",
                            "" if np.isnan(h) else f"{h:.4f}",
                            f"{nsd(pb, gt, sp):.6f}", int(pb.sum())])
                n += 1
            if (list(sl["kidney_id"]).index(kid) + 1) % 10 == 0:
                print(f"  {kid} done ({n} slices)", flush=True)
    lines = sum(1 for _ in open(OUT, errors="ignore"))
    print(f"nnU-Net scoring: {lines} lines (expect 18,323)", flush=True)
    assert lines == 18323, "row-count mismatch"


if __name__ == "__main__":
    main()
