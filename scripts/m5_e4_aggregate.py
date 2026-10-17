#!/usr/bin/env python3
"""M4/E4: aggregate the reverse-direction 5-fold CV (nnU-Net trained on
TRUSTED) into the E4 table.

For every fold of both configurations (2d, 3d_fullres): load the fold's
validation predictions (original geometry) and the matching ground truth from
labelsTr, compute per-kidney 3D Dice and HD95(mm) with the volume spacing,
then macro-average across the 59 kidneys (each kidney appears in exactly one
fold's validation set). Also reports per-patient means (39 clusters).

Output: outputs/summary/e4_reverse_cv.csv (per kidney) and printed aggregate.
"""
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__import__("os").environ.get("KIDNEYUS_ROOT",
          str(Path(__file__).resolve().parent.parent)))
RES = ROOT / "data/nnunet/nnUNetResults/Dataset002_TRUSTED"
LAB = ROOT / "data/nnunet/nnUNet_raw/Dataset002_TRUSTED/labelsTr"
OFFICIAL = {0: 12, 1: 10, 2: 13, 3: 12, 4: 12}  # US_cv1..5 sizes


def dice3d(a, b):
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return float(inter / s) if s else 1.0


def hd95_3d(a, b, spacing):
    from scipy import ndimage
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    d1 = ndimage.distance_transform_edt(~a, sampling=spacing)
    d2 = ndimage.distance_transform_edt(~b, sampling=spacing)
    surf_a = d1[b > 0] if (b > 0).any() else np.array([])
    surf_b = d2[a > 0] if (a > 0).any() else np.array([])
    all_d = np.concatenate([surf_a, surf_b])
    return float(np.percentile(all_d, 95)) if all_d.size else float("nan")


def main():
    rows = []
    for cfg in ("2d", "3d_fullres"):
        for fold in range(5):
            vdir = RES / f"nnUNetTrainer__nnUNetPlans__{cfg}/fold_{fold}/validation"
            preds = sorted(vdir.glob("TRUSTED_*.nii.gz")) if vdir.exists() else []
            if len(preds) != OFFICIAL[fold]:
                print(f"WARNING {cfg} f{fold}: {len(preds)} predictions (expected {OFFICIAL[fold]})")
            for p in preds:
                kid = p.stem.replace(".nii", "")
                gt_p = LAB / p.name
                pred = sitk.GetArrayFromImage(sitk.ReadImage(str(p))) > 0
                gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_p))) > 0
                sp = sitk.GetArrayFromImage.__self__ if False else None
                img_sitk = sitk.ReadImage(str(p))
                spacing = img_sitk.GetSpacing()[::-1]  # z,y,x
                rows.append({"config": cfg, "fold": fold, "kidney_id": kid,
                             "patient": kid[:3],
                             "dice3d": round(dice3d(pred, gt), 6),
                             "hd95_mm": round(hd95_3d(pred, gt, spacing), 4),
                             "n_slices": int(img_sitk.GetSize()[2])})
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs/summary/e4_reverse_cv.csv", index=False)
    for cfg in ("2d", "3d_fullres"):
        sub = df[df["config"] == cfg]
        if len(sub) == 0:
            continue
        pat = sub.groupby("patient")["dice3d"].mean()
        print(f"{cfg}: n_kidneys={len(sub)} dice3d={sub['dice3d'].mean():.4f}"
              f"±{sub['dice3d'].std():.4f} hd95={sub['hd95_mm'].mean():.2f}mm "
              f"| patient-level mean={pat.mean():.4f}±{pat.std():.4f} (n={len(pat)})")
    # fold-wise (must sum to 59 per config)
    for cfg in ("2d", "3d_fullres"):
        sub = df[df["config"] == cfg]
        per_fold = sub.groupby("fold")["dice3d"].agg(["count", "mean"])
        print(cfg, "per-fold:", {int(k): (int(v["count"]), round(v["mean"], 4))
                                  for k, v in per_fold.iterrows()})


if __name__ == "__main__":
    main()
