#!/usr/bin/env python3
"""P5 seed-43 box source: largest-CC bbox of the nnU-Net seed-43 best-checkpoint
TRUSTED predictions, one per evaluation slice (registered box_from_mask)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent))
from utils_kidney import box_from_mask  # noqa: E402

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
PRED = ROOT / "outputs/predictions/nnunet_seed43_best/trusted"
OUT = ROOT / "outputs/prompts/p5_box_trusted_s43.npz"

import pandas as pd
sl = pd.read_csv(ROOT / "outputs/trusted_us_slices.csv")
boxes, ids = [], []
for kid, z in zip(sl["kidney_id"], sl["slice_idx"]):
    z = int(z)
    p = PRED / f"{kid}_z{z}.nii.gz"
    b = None
    if p.exists():
        m = sitk.GetArrayFromImage(sitk.ReadImage(str(p))) > 0
        b = box_from_mask(np.squeeze(m))
    ids.append(f"{kid}_z{z:03d}")
    boxes.append([np.nan]*4 if b is None else list(map(float, b)))
np.savez_compressed(OUT, case_ids=np.array(ids), boxes=np.array(boxes, dtype=np.float32))
n_none = sum(1 for b in boxes if np.isnan(b).any())
print(json.dumps({"n": len(ids), "empty": n_none, "out": str(OUT)}))
