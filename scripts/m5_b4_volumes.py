#!/usr/bin/env python3
"""B4 (part 1): US-vs-CT kidney volume Bland-Altman for the nnU-Net
zero-shot forward configuration.

Per kidney (59): US predicted volume (stacked E2 nnU-Net seed42 slices,
voxel count x spacing), US GT volume, CT reference volume (GT_estimated_masksCT
connected components, laterality assigned by centroid: patient-right kidney
appears at smaller x in the axial CT). Bland-Altman of (US pred - CT ref) and
(US GT - CT ref).

The TTA-R2 configuration is appended by m5_b4_tta_volumes.py once the GPU
mega-pass lands.
"""
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
RAW = ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd"
PRED = ROOT / "outputs/predictions/nnunet_seed42_best/trusted"


def vol_ml(n_vox, spacing_xyz):
    sx, sy, sz = spacing_xyz
    return n_vox * sx * sy * sz / 1000.0


def main():
    import pandas as pd
    sl = pd.read_csv(ROOT / "outputs/trusted_us_slices.csv")
    slices_by_kidney = {}
    for kid, z in zip(sl["kidney_id"], sl["slice_idx"]):
        slices_by_kidney.setdefault(kid, []).append(int(z))

    # --- US volumes (GT + nnU-Net pred) per kidney ---
    us_json = ROOT / "outputs/summary/us_volumes.json"
    if us_json.exists():
        us = json.loads(us_json.read_text())
        print(f"[skip] US volumes cached ({len(us)} kidneys)", flush=True)
    else:
        us = {}
    for kid, zs in sorted(slices_by_kidney.items()):
        if kid in us:
            continue
        gimg = sitk.ReadImage(str(RAW / f"US_DATA/US_masks/GT_estimated_masksUS/{kid}_maskUS.nii.gz"))
        gt = sitk.GetArrayFromImage(gimg) > 0.5
        sp = gimg.GetSpacing()[::-1]  # z,y,x mm
        pred_vol = 0
        for z in zs:
            f = PRED / f"{kid}_z{z}.nii.gz"
            if f.exists():
                pred_vol += int((sitk.GetArrayFromImage(sitk.ReadImage(str(f))) > 0).sum())
        us[kid] = {"gt_ml": round(vol_ml(int(gt.sum()), sp), 3),
                   "pred_ml": round(vol_ml(pred_vol, sp), 3),
                   "us_spacing_zyx_mm": [round(s, 3) for s in sp]}
    json.dump(us, open(ROOT / "outputs/summary/us_volumes.json", "w"), indent=1)
    print(f"US volumes: {len(us)} kidneys", flush=True)

    # --- CT reference volumes per kidney component ---
    ctm_dir = RAW / "CT_DATA/CT_masks/GT_estimated_masksCT"
    ct = {}
    for p in sorted(ctm_dir.glob("*_maskCT.nii.gz")):
        pid = p.name.replace(".nii.gz", "").replace("_maskCT", "")
        img = sitk.ReadImage(str(p))
        m = sitk.GetArrayFromImage(img) > 0.5
        sp = img.GetSpacing()  # x,y,z mm
        lab, n = ndimage.label(m)
        if n < 2:
            print(f"WARN {pid}: only {n} component(s)", flush=True)
            continue
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        order = np.argsort(sizes)[::-1][:2]
        comps = []
        for ci in order:
            ci += 1
            coords = np.argwhere(lab == ci)   # (z,y,x)
            comps.append({"vol_ml": round(vol_ml(int(sizes[ci - 1]), sp), 3),
                          "x_centroid": round(float(coords[:, 2].mean()), 1)})
        comps.sort(key=lambda c: c["x_centroid"])  # smaller x = patient right
        ct[pid] = {"ct_R_ml": comps[0]["vol_ml"], "ct_L_ml": comps[1]["vol_ml"]}
    json.dump(ct, open(ROOT / "outputs/summary/ct_reference_volumes.json", "w"), indent=1)
    print(f"CT reference: {len(ct)} patients", flush=True)

    # --- join + Bland-Altman ---
    def ba(pairs, label):
        d = np.array([p - c for p, c in pairs])
        m = np.array([(p + c) / 2 for p, c in pairs])
        bias = float(d.mean())
        loa = (float(bias - 1.96 * d.std(ddof=1)), float(bias + 1.96 * d.std(ddof=1)))
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.scatter(m, d, s=22, alpha=0.75, color="#34495e")
        ax.axhline(bias, color="#e67e22", lw=1.5, label=f"bias {bias:+.1f} mL")
        ax.axhline(loa[0], color="#c0392b", ls="--", lw=1.2, label="95% LoA")
        ax.axhline(loa[1], color="#c0392b", ls="--", lw=1.2)
        ax.set_xlabel("mean of US volume and CT reference (mL)")
        ax.set_ylabel("US volume − CT reference (mL)")
        ax.set_title(f"Bland–Altman: {label} (n={len(pairs)} kidneys)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        return {"n": len(pairs), "bias_ml": round(bias, 2),
                "loa95_ml": [round(loa[0], 2), round(loa[1], 2)],
                "mean_us_pred_ml": round(float(np.mean([p for p, _ in pairs])), 2),
                "mean_ct_ref_ml": round(float(np.mean([c for _, c in pairs])), 2)}, fig

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs_pred, pairs_gt = [], []
    for kid, u in us.items():
        pid, side = kid[:3], kid[3]
        if pid not in ct:
            continue
        ref = ct[pid]["ct_R_ml"] if side == "R" else ct[pid]["ct_L_ml"]
        pairs_pred.append((u["pred_ml"], ref))
        pairs_gt.append((u["gt_ml"], ref))
    s_pred, f_pred = ba(pairs_pred, "nnU-Net zero-shot US prediction vs CT")
    f_pred.savefig(ROOT / "outputs/figures/ba_us_pred_vs_ct_nnunet.png", dpi=150)
    s_gt, f_gt = ba(pairs_gt, "US GT (STAPLE) vs CT")
    f_gt.savefig(ROOT / "outputs/figures/ba_us_gt_vs_ct.png", dpi=150)
    out = {"nnunet_zeroshot_vs_ct": s_pred, "us_gt_vs_ct": s_gt}
    (ROOT / "outputs/summary/volume_bland_altman.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
