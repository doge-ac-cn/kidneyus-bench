#!/usr/bin/env python3
"""Bland-Altman: per-kidney 3D Dice of the reverse-direction 2d vs 3d_fullres
configurations (same 59 kidneys, each held out once per configuration).
Bias, 95% limits of agreement, and the agreement figure (aggregate plot, no
patient identifiers) are written to outputs/figures/ and outputs/summary/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__import__("os").environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
df = pd.read_csv(ROOT / "outputs/summary/e4_reverse_cv.csv")
w = df.pivot(index="kidney_id", columns="config", values="dice3d")
w = w.dropna()
a, b = w["2d"].values, w["3d_fullres"].values
mean_ab = (a + b) / 2
diff = a - b
bias = float(diff.mean())
loa = (float(diff.mean() - 1.96 * diff.std(ddof=1)),
       float(diff.mean() + 1.96 * diff.std(ddof=1)))

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(mean_ab, diff, s=18, alpha=0.7, color="#34495e")
ax.axhline(bias, color="#e67e22", lw=1.5, label=f"bias {bias:+.3f}")
for v, lab in zip(loa, ("95% LoA upper", "95% LoA lower")):
    ax.axhline(v, color="#c0392b", ls="--", lw=1.2, label=lab)
ax.set_xlabel("mean of 2d and 3d Dice (per kidney)")
ax.set_ylabel("2d − 3d Dice difference")
ax.set_title(f"Bland–Altman: reverse CV, 2d vs 3d_fullres (n={len(a)} kidneys)")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(ROOT / "outputs/figures/bland_altman_reverse_2d_vs_3d.png", dpi=150)

out = {"n_kidneys": int(len(a)),
       "mean_dice_2d": round(float(a.mean()), 4),
       "mean_dice_3d": round(float(b.mean()), 4),
       "bias_2d_minus_3d": round(bias, 4),
       "limits_of_agreement_95": [round(loa[0], 4), round(loa[1], 4)],
       "pearson_r_means": round(float(np.corrcoef(a, b)[0, 1]), 4),
       "note": "per-kidney 3D Dice, reverse-direction 5-fold CV; positive bias = "
               "2d higher than 3d on the same held-out kidney"}
(ROOT / "outputs/summary/bland_altman_reverse.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
