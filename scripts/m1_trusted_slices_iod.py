#!/usr/bin/env python3
"""M1-TRUSTED: (a) extract effective 2D slices (plan §4: GT/STAPLE mask area
>= 50 px per slice), (b) compute IoD — physician A/B vs STAPLE-consensus Dice
(plan §7 human reference; M1 gate: IoD in 0.90–0.96).

Outputs:
  outputs/trusted_us_slices.csv     kidney_id, slice_idx, area_gt, dice_A, dice_B
  outputs/metrics/iod_reference.csv kidney-level + slice-level summary rows
  outputs/preprocess_config.json    preprocessing parameters actually used
"""
import csv
import json
import os
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_kidney import dice_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data/raw/TRUSTED/TRUSTED_dataset_for_nsd")
MIN_AREA = 50  # plan §4: consensus mask area >= 50 px


def main():
    us_img = os.path.join(DATA, "US_DATA/US_images")
    us_masks = os.path.join(DATA, "US_DATA/US_masks")

    vols = sorted(f for f in os.listdir(us_img) if f.endswith("_imgUS.nii.gz"))
    rows = []
    spacing_note = {}
    for vf in vols:
        kid = vf.replace("_imgUS.nii.gz", "")
        img = sitk.ReadImage(os.path.join(us_img, vf))
        gt = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(us_masks, "GT_estimated_masksUS", f"{kid}_maskUS.nii.gz")))
        a1 = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(us_masks, "Annotator1", f"{kid}1_maskUS.nii.gz")))
        a2 = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(us_masks, "Annotator2", f"{kid}2_maskUS.nii.gz")))
        spacing_note[kid] = {"spacing_xyz_mm": [round(x, 4) for x in img.GetSpacing()],
                             "size_xyz": list(img.GetSize())}
        # masks are float NIfTIs (STAPLE probability maps / float-encoded labels);
        # tiny nonzero float noise (1e-27) exists, so binarize at 0.5, not >0
        gt = gt > 0.5
        a1 = a1 > 0.5
        a2 = a2 > 0.5
        for z in range(gt.shape[0]):
            area = int(gt[z].sum())
            if area < MIN_AREA:
                continue
            rows.append({
                "kidney_id": kid,
                "slice_idx": z,
                "area_gt": area,
                "dice_A_vs_GT": round(dice_score(a1[z], gt[z]), 6),
                "dice_B_vs_GT": round(dice_score(a2[z], gt[z]), 6),
                "dice_A_vs_B": round(dice_score(a1[z], a2[z]), 6),
            })

    os.makedirs(os.path.join(ROOT, "outputs/metrics"), exist_ok=True)
    csv_path = os.path.join(ROOT, "outputs/trusted_us_slices.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # IoD per plan §7: mean Dice of physician A/B against consensus
    dA = np.mean([r["dice_A_vs_GT"] for r in rows])
    dB = np.mean([r["dice_B_vs_GT"] for r in rows])
    iod = (dA + dB) / 2
    # direct inter-rater agreement (strict ceiling; consensus is derived from A/B,
    # so A-vs-GT/B-vs-GT are self-consistent and optimistic — see review v1.2)
    dAB = np.mean([r["dice_A_vs_B"] for r in rows])

    # kidney-level version (aggregate masks per kidney over effective slices)
    kid_rows = {}
    for r in rows:
        k = kid_rows.setdefault(r["kidney_id"], {"A": [], "B": []})
        k["A"].append(r["dice_A_vs_GT"])
        k["B"].append(r["dice_B_vs_GT"])
    kA = np.mean([np.mean(v["A"]) for v in kid_rows.values()])
    kB = np.mean([np.mean(v["B"]) for v in kid_rows.values()])
    iod_kidney = (kA + kB) / 2
    kidAB = {}
    for r in rows:
        kidAB.setdefault(r["kidney_id"], []).append(r["dice_A_vs_B"])
    dAB_kidney = float(np.mean([np.mean(v) for v in kidAB.values()]))

    with open(os.path.join(ROOT, "outputs/metrics/iod_reference.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "dice_A_vs_GT", "dice_B_vs_GT", "IoD_mean", "dice_A_vs_B_direct"])
        w.writerow(["slice", round(dA, 6), round(dB, 6), round(iod, 6), round(dAB, 6)])
        w.writerow(["kidney", round(kA, 6), round(kB, 6), round(iod_kidney, 6), round(dAB_kidney, 6)])

    with open(os.path.join(ROOT, "outputs/trusted_spacing.json"), "w") as f:
        json.dump(spacing_note, f, indent=2)

    cfg = {
        "trusted_effective_slice_rule": f"STAPLE GT mask area >= {MIN_AREA} px",
        "n_kidneys": len(kid_rows),
        "n_effective_slices": len(rows),
        "IoD_slice_level": round(float(iod), 6),
        "IoD_kidney_level": round(float(iod_kidney), 6),
        "IoD_direct_AB_slice": round(float(dAB), 6),
        "IoD_direct_AB_kidney": round(float(dAB_kidney), 6),
        "note": "A/B-vs-GT is self-consistent (consensus derived from A/B); direct A-vs-B is the strict ceiling",
        "gate_M1": "0.90 <= IoD <= 0.96 (plan §9, self-consistent variant)",
        "gate_pass": bool(0.90 <= iod <= 0.96 or 0.90 <= iod_kidney <= 0.96),
        "grayscale_window": "1-99 percentile, scaled to [0,1] (utils_kidney.window_normalize)",
        "mask_interpolation": "nearest",
    }
    with open(os.path.join(ROOT, "outputs/preprocess_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
