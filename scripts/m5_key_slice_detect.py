#!/usr/bin/env python3
"""M5 P3-7: detect hand-drawn (key) vs interpolated mask slices in TRUSTED.

TRUSTED masks were drawn roughly every 5 mm (~1 in 17 slices per the dataset
documentation) and interpolated in between. Key slices leave a signature in
the mask sequence: the contour jumps where an annotator re-drew, while
interpolated slices morph smoothly. Detector: for each kidney volume compute
the normalized boundary shift s(z) = XOR(m[z], m[z+1]) / min(area), flag local
maxima above a robust threshold (median + 3*MAD) as key slices.

Outputs:
  outputs/metrics/key_slices_trusted.csv  - per-slice flags (kidney ids ->
      GITIGNORED, internal use incl. rater-scoring stratification)
  outputs/summary/key_slice_aggregate.json - case-free aggregate (committable):
      keys per kidney, stratified nnU-Net / MedSAM / STAPLE-relevant Dice
      (from the primary e2 CSV) on key vs interpolated slices.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__import__("os").environ.get("KIDNEYUS_ROOT",
          str(Path(__file__).resolve().parent.parent)))
RAW = ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA/US_masks/GT_estimated_masksUS"


def key_slices_for(volume):
    m = sitk.GetArrayFromImage(sitk.ReadImage(str(volume))) > 0.5
    areas = m.reshape(m.shape[0], -1).sum(1).astype(np.float64)
    nz = np.nonzero(areas)[0]
    keys = set()
    for a, b in zip(nz[:-1], nz[1:]):
        if b - a != 1:
            continue  # shift only defined for adjacent annotated slices
        inter = (m[a] & m[b]).sum()
        union = (m[a] | m[b]).sum()
        s = 1.0 - (inter / union if union else 1.0)  # 1 - IoU of adjacent slices
        keys.add((a, s))
    if not keys:
        return m, [], areas
    zs = np.array([z for z, _ in keys])
    ss = np.array([s for _, s in keys])
    med = np.median(ss)
    mad = np.median(np.abs(ss - med)) + 1e-12
    thr = med + 3 * mad
    picked = [int(z) for z, s in keys if s >= thr and s > 0]
    # local-maximum filter within the flagged set
    smap = dict(keys)
    out = []
    for z in picked:
        left, right = smap.get(z - 1, -1), smap.get(z + 1, -1)
        if smap[z] >= left and smap[z] >= right:
            out.append(z)
    return m, out, areas


def main():
    kidneys = sorted({p.name.split("_maskUS")[0] for p in RAW.glob("*_maskUS.nii.gz")})
    # model dice per case from the primary CSV (nnU-Net best = primary caliber)
    model_dice = defaultdict(list)
    with open(ROOT / "outputs/metrics/e2_external.csv") as f:
        for r in csv.DictReader(f):
            if r["model"] == "nnunet" and r["prompt"] == "none" and r["seed"] == "42":
                model_dice[r["kidney_id"]].append((int(r["slice_idx"]), float(r["dice"])))

    with open(ROOT / "outputs/metrics/key_slices_trusted.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kidney_id", "slice_idx", "is_key"])
        agg = {}
        for kid in kidneys:
            _m, keys, _areas = key_slices_for(RAW / f"{kid}_maskUS.nii.gz")
            kset = set(keys)
            for z, d in model_dice.get(kid, []):
                w.writerow([kid, z, int(z in kset)])
            dk = [d for z, d in model_dice.get(kid, []) if z in kset]
            di = [d for z, d in model_dice.get(kid, []) if z not in kset]
            agg[kid] = {"n_ann_slices": len(model_dice.get(kid, [])),
                        "n_key": len(kset),
                        "dice_key_mean": round(float(np.mean(dk)), 4) if dk else None,
                        "dice_interp_mean": round(float(np.mean(di)), 4) if di else None}
        n_key = sum(v["n_key"] for v in agg.values())
        n_ann = sum(v["n_ann_slices"] for v in agg.values())
        dk = [v["dice_key_mean"] for v in agg.values() if v["dice_key_mean"] is not None]
        di = [v["dice_interp_mean"] for v in agg.values() if v["dice_interp_mean"] is not None]
        out = {
            "n_kidneys": len(kidneys), "n_ann_slices": n_ann, "n_key_slices": n_key,
            "key_fraction": round(n_key / max(n_ann, 1), 4),
            "expected_if_5mm_1in17": round(1 / 17, 4),
            "nnunet_seed42_dice_key_slices_mean_over_kidneys": round(float(np.mean(dk)), 4),
            "nnunet_seed42_dice_interp_slices_mean_over_kidneys": round(float(np.mean(di)), 4),
            "per_kidney_detail": "outputs/metrics/key_slices_trusted.csv (gitignored)",
        }
        (ROOT / "outputs/summary/key_slice_aggregate.json").write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v for k, v in out.items() if k != "per_kidney_detail"}, indent=1))


if __name__ == "__main__":
    main()
