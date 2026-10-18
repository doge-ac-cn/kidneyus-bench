#!/usr/bin/env python3
"""Shared utilities for the OKD/TRUSTED kidney benchmark (plan §5/§7).

All models must produce numbers through these same functions (plan §5:
"所有模型在同一预处理、同一评价脚本上出数").
"""
import json
import os
import random

import numpy as np
import torch
from scipy import ndimage as ndi


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- metrics
def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool)
    g = gt.astype(bool)
    denom = p.sum() + g.sum()
    if denom == 0:
        return 1.0  # both empty
    return float(2.0 * (p & g).sum() / denom)


def _surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    er = ndi.binary_erosion(mask)
    return mask & ~er


def hd95_mm(pred: np.ndarray, gt: np.ndarray, spacing_mm) -> float:
    """95th-percentile Hausdorff distance between symmetric surfaces (mm).

    medpy.metric.binary.hd95-equivalent: the 95th percentile is taken per
    direction (pred->gt, gt->pred) and the MAX of the two is returned
    (pooling the two directional distance sets before the percentile — as an
    earlier version did — is NOT equivalent when surface point counts differ).
    Self-contained re-implementation to avoid the unmaintained medpy
    dependency; verified equal on random 2D pairs.
    `spacing_mm` is per-axis voxel spacing in ARRAY axis order (z, y, x) —
    i.e. reversed relative to SimpleITK's GetSpacing() (x, y, z).
    """
    ps, gs = _surface(pred.astype(bool)), _surface(gt.astype(bool))
    if not ps.any() or not gs.any():
        return float("nan")
    dt_gt = ndi.distance_transform_edt(~gs, sampling=spacing_mm)
    dt_pred = ndi.distance_transform_edt(~ps, sampling=spacing_mm)
    d1 = dt_gt[ps]
    d2 = dt_pred[gs]
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


# ---------------------------------------------------------------- prompts
def box_from_mask(mask: np.ndarray, pad: int = 0, square: bool = False):
    """Axis-aligned bbox (x0,y0,x1,y1) of the largest connected component."""
    m = mask.astype(bool)
    if not m.any():
        return None
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    ys, xs = np.where(m)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    if pad:
        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
        x1 += pad; y1 += pad
    if square:  # MedSAM convention: box may need min side >= some px, keep as-is here
        pass
    return [x0, y0, x1, y1]


def centroid_from_mask(mask: np.ndarray):
    m = mask.astype(bool)
    if not m.any():
        return None
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    cy, cx = ndi.center_of_mass(m)
    return [int(round(cx)), int(round(cy))]


# ---------------------------------------------------------------- preprocessing
def window_normalize(img: np.ndarray, lo_p: float = 1.0, hi_p: float = 99.0) -> np.ndarray:
    """Plan §4: grayscale window at 1–99th percentile then scale to [0,1]."""
    a = img.astype(np.float32)
    lo, hi = np.percentile(a, [lo_p, hi_p])
    if hi <= lo:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


# ---------------------------------------------------------------- io
def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
