#!/usr/bin/env python3
"""M5 P1-3: nnU-Net external-validation per-kidney audit.

Aggregates e2_external.csv by kidney, renders the 59-kidney Dice histogram and
exports image+GT+prediction overlays for the two worst and two best kidneys
(three lowest-Dice slices each) for visual adjudication of failure mode:
consistent gain/depth/artifact pattern -> true heterogeneity; whole-slice
misalignment or all-zero predictions -> evaluation artifact.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

ROOT = Path(__import__("os").environ.get("KIDNEYUS_ROOT",
          str(Path(__file__).resolve().parent.parent)))
RAW = ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA"
PRED = ROOT / "outputs/predictions/nnunet_seed42_best/trusted"
SEED = 42  # median seed of the main-caliber (best-checkpoint) rows
OUT = ROOT / "outputs/figures"
N_WORST, N_BEST, N_SLICES = 2, 2, 3

rows = [r for r in csv.DictReader(open(ROOT / "outputs/metrics/e2_external.csv"))
        if r["model"] == "nnunet" and r["prompt"] == "none" and r["seed"] == str(SEED)]
by_kidney = defaultdict(list)
for r in rows:
    by_kidney[r["kidney_id"]].append((float(r["dice"]), int(r["slice_idx"])))

means = sorted((np.mean([d for d, _ in v]), k) for k, v in by_kidney.items())
assert len(means) == 59, f"expected 59 kidneys, got {len(means)}"

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist([m for m, _ in means], bins=np.arange(0, 1.01, 0.05), edgecolor="black")
ax.set_xlabel("per-kidney mean Dice (nnU-Net, TRUSTED external)")
ax.set_ylabel("# kidneys")
ax.set_title(f"nnU-Net seed{SEED}: 59 kidneys, slice-mean {np.mean([m for m, _ in means]):.3f}")
fig.tight_layout()
fig.savefig(OUT / "nnunet_e2_kidney_histogram.png", dpi=150)

picked = [k for _, k in means[:N_WORST]] + [k for _, k in means[-N_BEST:]]
report = {"seed": SEED, "kidney_mean_dice": {k: round(float(m), 4) for m, k in means},
          "overlays": []}
for kid in picked:
    img_v = sitk.GetArrayFromImage(sitk.ReadImage(str(RAW / f"US_images/{kid}_imgUS.nii.gz")))
    gt_v = sitk.GetArrayFromImage(sitk.ReadImage(str(RAW / f"US_masks/GT_estimated_masksUS/{kid}_maskUS.nii.gz"))) > 0.5
    ds = sorted(by_kidney[kid])
    zs = [z for _, z in ds[:N_SLICES]]
    fig, axes = plt.subplots(1, len(zs), figsize=(4 * len(zs), 4))
    for ax, z in zip(np.atleast_1d(axes), zs):
        img = img_v[z].astype(np.float32)
        img = (img - img.min()) / max(img.ptp(), 1e-6)
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(PRED / f"{kid}_z{z}.nii.gz"))) > 0
        pred = np.squeeze(pred)
        assert pred.ndim == 2, f"pred for {kid} z{z} has shape {pred.shape}"
        ax.imshow(img, cmap="gray")
        ax.contour(gt_v[z].astype(float), levels=[0.5], colors=["#2ecc71"], linewidths=1.5)
        ax.contour(pred.astype(float), levels=[0.5], colors=["#e74c3c"], linewidths=1.5)
        ax.set_title(f"{kid} z{z} dice={dict((zz, d) for d, zz in ds)[z]:.3f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"green=STAPLE GT  red=nnU-Net(pred)  | kidney mean {np.mean([d for d, _ in ds]):.3f}", fontsize=10)
    fig.tight_layout()
    p = OUT / f"bimodal/{kid}_overlay.png"
    fig.savefig(p, dpi=150)
    report["overlays"].append({"kidney": kid, "mean_dice": float(np.mean([d for d, _ in ds])),
                               "slices": zs, "png": str(p.relative_to(ROOT))})

with open(OUT / "bimodal/audit_report.json", "w") as f:
    json.dump(report, f, indent=1)
lo = [f"{k}={m:.3f}" for m, k in means[:5]]
hi = [f"{k}={m:.3f}" for m, k in means[-5:]]
print("worst5:", ", ".join(lo))
print("best5: ", ", ".join(hi))
print("overlays ->", OUT / "bimodal")
