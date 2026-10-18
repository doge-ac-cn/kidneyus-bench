#!/usr/bin/env python3
"""Supplementary mega-pass (observer-stratified + boundary metrics): per-slice metrics for 9 configs x
18,322 slices with observer-stratified scoring.

Per slice, for one (model, box-source) config:
  dice      model vs STAPLE consensus   (registered scorer)
  dice_A    model vs Annotator1
  dice_B    model vs Annotator2
  hd95      model vs STAPLE (mm; NaN when empty pred or empty GT)
  nsd       normalized surface dice @ tau = 1.0 mm vs STAPLE
  pred_area_px
Empty box -> all zeros, hd95 NaN (registered v1.2 empty-prompt rule).

Configs (all with the model seed 42 = registered single-seed caliber):
  medsam    x {p3(oracle), p4reg, tta_r2s43}
  sam_lora  x {p3, p4reg, tta_r2s43}
  samus     x {p3, p4reg, tta_r2s43}
  samus43_tta_r2s43   (item ⑧ sensitivity: SAMUS seed 43 under TTA-R2 boxes)

Atomic .part -> rename with row-count validation; per-GPU worker CLI.
"""
import argparse
import csv
import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from m3_common import iter_trusted, load_sam_family, load_samus, \
    infer_sam_family, infer_samus_box, MODEL_RES  # noqa: E402
from utils_kidney import box_from_mask  # noqa: E402

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "outputs/metrics"
EXPECTED_LINES = 18323
MASKS = ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA/US_masks/GT_estimated_masksUS"
AMASKS = ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA/US_masks"

PASSES = {
    "medsam_p3": ("medsam", 42, "p3"), "medsam_p4reg": ("medsam", 42, "p4reg"),
    "medsam_tta_r2s43": ("medsam", 42, "tta_r2s43"),
    "samlora_p3": ("sam_lora", 42, "p3"), "samlora_p4reg": ("sam_lora", 42, "p4reg"),
    "samlora_tta_r2s43": ("sam_lora", 42, "tta_r2s43"),
    "samus_p3": ("samus", 42, "p3"), "samus_p4reg": ("samus", 42, "p4reg"),
    "samus_tta_r2s43": ("samus", 42, "tta_r2s43"),
    "samus43_tta_r2s43": ("samus", 43, "tta_r2s43"),
}

_boxes_cache = {}


def boxes_for(source):
    if source in _boxes_cache:
        return _boxes_cache[source]
    if source == "p3":
        z = np.load(ROOT / "outputs/prompts/p3_oracle_trusted.npz")
        table = {str(c): (None if np.isnan(np.asarray(b, dtype=float)).any()
                          else np.asarray(b, dtype=float).tolist())
                 for c, b in zip(z["case_ids"], z["boxes"])}
    elif source == "p4reg":
        z = np.load(ROOT / "outputs/prompts/p4_box_trusted.npz", allow_pickle=True)
        table = {}
        for k in z.files:
            arr = np.asarray(z[k], dtype=float)
            table[k] = None if (arr.size != 4 or np.isnan(arr).any()) else arr.tolist()
    elif source == "tta_r2s43":
        z = np.load(ROOT / "outputs/prompts/yolo_alldet_trusted_tta_r2_s43.npz")
        per = {}
        for i, b, c in zip(z["row_index"], z["boxes"], z["confs"]):
            per.setdefault(int(i), []).append((float(c), b))
        ids = [str(c) for c in z["case_ids"]]
        table = {}
        for i, cid in enumerate(ids):
            cands = [(c, b) for c, b in per.get(i, []) if c >= 0.25]
            table[cid] = max(cands, key=lambda x: x[0])[1].tolist() if cands else None
    else:
        raise KeyError(source)
    _boxes_cache[source] = table
    return table


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else 1.0


def surf_pts(mask, sp):
    er = ndimage.binary_erosion(mask, iterations=1, border_value=0)
    return np.argwhere(mask & ~er) * np.asarray(sp, dtype=np.float64)


def hd95(pred, gt, sp):
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    pp, gp = surf_pts(pred, sp), surf_pts(gt, sp)
    if len(pp) == 0 or len(gp) == 0:
        return float("nan")
    ta, tb = cKDTree(pp), cKDTree(gp)
    d = np.concatenate([ta.query(gp, k=1)[0], tb.query(pp, k=1)[0]])
    return float(np.percentile(d, 95))


def nsd(pred, gt, sp, tau=1.0):
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    pp, gp = surf_pts(pred, sp), surf_pts(gt, sp)
    if len(pp) == 0 or len(gt) == 0 or len(gp) == 0:
        return 0.0
    tb = cKDTree(gp)
    ta = cKDTree(pp)
    hit_p = int((tb.query(pp, k=1)[0] <= tau).sum())
    hit_g = int((ta.query(gp, k=1)[0] <= tau).sum())
    denom = len(pp) + len(gp)
    return float((hit_p + hit_g) / denom) if denom else 0.0


_ab_cache = {}
PACKED = ROOT / "outputs/masks_packed"


def _unpack_row(row, shape):
    n = int(np.prod(shape))
    return np.unpackbits(row, count=n).reshape(shape).astype(bool)


def ab_slices(kid, z):
    """(A_slice, B_slice) as bool arrays from the packed per-kidney cache
    (fallback: read the volumes)."""
    cache_file = PACKED / f"{kid}.npz"
    if cache_file.exists():
        if _ab_cache.get("kid") != kid:
            _ab_cache.clear()
            _ab_cache["kid"] = kid
            zt = np.load(cache_file)
            _ab_cache["zs"] = list(map(int, zt["zs"]))
            _ab_cache["shape"] = tuple(int(v) for v in zt["shape"])
            _ab_cache["A"] = zt["A"]
            _ab_cache["B"] = zt["B"]
        i = _ab_cache["zs"].index(int(z))
        shape = _ab_cache["shape"]
        return (_unpack_row(_ab_cache["A"][i], shape),
                _unpack_row(_ab_cache["B"][i], shape))
    if _ab_cache.get("kid") != kid:
        import SimpleITK as sitk
        _ab_cache.clear()
        _ab_cache["kid"] = kid
        _ab_cache["A"] = sitk.GetArrayFromImage(
            sitk.ReadImage(str(AMASKS / "Annotator1" / f"{kid}1_maskUS.nii.gz"))) > 0.5
        _ab_cache["B"] = sitk.GetArrayFromImage(
            sitk.ReadImage(str(AMASKS / "Annotator2" / f"{kid}2_maskUS.nii.gz"))) > 0.5
    return _ab_cache["A"][z], _ab_cache["B"][z]


def pass_complete(tag):
    p = OUT / f"rev_{tag}.csv"
    return p.exists() and sum(1 for _ in open(p, errors="ignore")) == EXPECTED_LINES


def run_pass(tag, gpu):
    if pass_complete(tag):
        print(f"[skip] {tag} (complete)", flush=True)
        return
    model_name, seed, source = PASSES[tag]
    part = OUT / f"rev_{tag}.csv.part"
    fh = open(part, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["case_id", "kidney_id", "patient", "dice", "dice_A", "dice_B",
                "hd95", "nsd", "pred_area_px"])

    model = load_samus(seed, "cuda") if model_name == "samus" else \
        load_sam_family(model_name, seed, "cuda")
    res = MODEL_RES[model_name]
    boxes = boxes_for(source)

    q = queue.Queue(maxsize=6)

    def producer():
        try:
            for item in iter_trusted():
                q.put(item)
        finally:
            q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    n = 0
    while True:
        item = q.get()
        if item is None:
            break
        cid, img, gt, spacing, meta = item
        box = boxes.get(cid)
        if box is None:
            row = [cid, meta["kidney_id"], meta["kidney_id"][:3], "0.000000",
                   "0.000000", "0.000000", "", "0.000000", 0]
        else:
            if model_name == "samus":
                pred = infer_samus_box(model, img, "cuda", box=box)
            else:
                pred = infer_sam_family(model, img, res, "cuda", box=box)
            pb = pred.astype(bool)
            sp = spacing[1:3]  # 2D (y,x) mm
            a, b = ab_slices(meta["kidney_id"], meta["slice_idx"])
            h = hd95(pb, gt, sp)
            row = [cid, meta["kidney_id"], meta["kidney_id"][:3],
                   f"{dice(pb, gt):.6f}", f"{dice(pb, a):.6f}", f"{dice(pb, b):.6f}",
                   "" if np.isnan(h) else f"{h:.4f}", f"{nsd(pb, gt, sp):.6f}",
                   int(pb.sum())]
        w.writerow(row)
        n += 1
        if n % 4000 == 0:
            print(f"  {tag}: {n} slices", flush=True)
    fh.close()
    final = OUT / f"rev_{tag}.csv"
    os.replace(part, final)
    if sum(1 for _ in open(final, errors="ignore")) != EXPECTED_LINES:
        raise RuntimeError(f"{tag}: row-count mismatch")
    print(f"[done] {tag} ({n} slices)", flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--passes", required=True)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    for tag in a.passes.split(","):
        run_pass(tag, a.gpu)
    print(f"WORKER DONE gpu={a.gpu} passes={a.passes}", flush=True)


if __name__ == "__main__":
    main()
