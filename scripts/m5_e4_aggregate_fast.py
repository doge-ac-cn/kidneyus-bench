#!/usr/bin/env python3
"""M4/E4 fast aggregation: surface-point + cKDTree HD95 (equivalent to the
distance-transform formulation, ~20x faster on 822x768x1024 volumes)."""
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__import__("os").environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
RES = ROOT / "data/nnunet/nnUNetResults/Dataset002_TRUSTED"
LAB = ROOT / "data/nnunet/nnUNet_raw/Dataset002_TRUSTED/labelsTr"
OFFICIAL = {0: 12, 1: 10, 2: 13, 3: 12, 4: 12}


def surface_points(mask, spacing):
    er = ndimage.binary_erosion(mask, iterations=1, border_value=0)
    surf = mask & ~er
    idx = np.argwhere(surf)
    return idx * np.asarray(spacing, dtype=np.float64)


def hd95_3d(a, b, spacing):
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    pa = surface_points(a, spacing)
    pb = surface_points(b, spacing)
    if len(pa) == 0 or len(pb) == 0:
        return float("nan")
    ta, tb = cKDTree(pa), cKDTree(pb)
    d_ab = ta.query(pb, k=1)[0]
    d_ba = tb.query(pa, k=1)[0]
    return float(np.percentile(np.concatenate([d_ab, d_ba]), 95))


def dice3d(a, b):
    inter = np.logical_and(a, b).sum()
    s = a.sum() + b.sum()
    return float(2.0 * inter / s) if s else 1.0  # Dice = 2|A∩B|/(|A|+|B|)


def main():
    import pandas as pd
    rows = []
    for cfg in ("2d", "3d_fullres"):
        for fold in range(5):
            vdir = RES / f"nnUNetTrainer__nnUNetPlans__{cfg}/fold_{fold}/validation"
            preds = sorted(vdir.glob("TRUSTED_*.nii.gz")) if vdir.exists() else []
            if len(preds) != OFFICIAL[fold]:
                print(f"WARNING {cfg} f{fold}: {len(preds)}/{OFFICIAL[fold]}", flush=True)
            for p in preds:
                kid = p.stem.replace(".nii", "")
                img = sitk.ReadImage(str(p))
                pred = sitk.GetArrayFromImage(img) > 0
                gt = sitk.GetArrayFromImage(sitk.ReadImage(str(LAB / p.name))) > 0
                spacing = img.GetSpacing()[::-1]
                kid_clean = kid.replace("TRUSTED_", "")
                rows.append({"config": cfg, "fold": fold, "kidney_id": kid_clean,
                             "patient": kid_clean[:3], "dice3d": round(dice3d(pred, gt), 6),
                             "hd95_mm": round(hd95_3d(pred, gt, spacing), 4),
                             "n_slices": int(img.GetSize()[2])})
                if len(rows) % 10 == 0:
                    print(f"  {len(rows)} kidneys scored", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs/summary/e4_reverse_cv.csv", index=False)
    for cfg in ("2d", "3d_fullres"):
        sub = df[df["config"] == cfg]
        if len(sub) == 0:
            continue
        pat = sub.groupby("patient")["dice3d"].mean()
        print(f"{cfg}: n={len(sub)} dice3d={sub['dice3d'].mean():.4f}±{sub['dice3d'].std():.4f} "
              f"hd95={sub['hd95_mm'].mean():.2f}mm | patient mean={pat.mean():.4f}±{pat.std():.4f} (n={len(pat)})")
        per_fold = sub.groupby("fold")["dice3d"].agg(["count", "mean"])
        print("  per-fold:", {int(k): (int(v["count"]), round(v["mean"], 4)) for k, v in per_fold.iterrows()})


if __name__ == "__main__":
    main()
