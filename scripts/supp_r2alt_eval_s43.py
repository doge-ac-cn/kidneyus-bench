#!/usr/bin/env python3
"""R2-alt downstream evaluation under the registered seed 43, using the packed
GT cache (no per-kidney GT volume re-read).

For each single-start R2 detector replicate (dumps r2alt_s42/43/44): top-1 box
at conf>=0.25 -> MedSAM seed-43 weights -> registered scorer; writes
rev43_r2alt_s{seed}_medsam.csv.
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from m3_common import load_sam_family, infer_sam_family, MODEL_RES, score  # noqa: E402

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
PACK = ROOT / "outputs/masks_packed"
US = ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA/US_images"
MET = ROOT / "outputs/metrics"


def unpack(row, shape):
    n = int(np.prod(shape))
    return np.unpackbits(row, count=n).reshape(shape).astype(bool)


def main():
    import SimpleITK as sitk
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sam_family("medsam", 43, dev)
    res = MODEL_RES["medsam"]
    seeds = tuple(int(s) for s in os.environ.get("R2ALT_SEEDS", "42,43,44").split(","))
    for seed in seeds:
        z = np.load(ROOT / f"outputs/prompts/yolo_alldet_trusted_r2alt_s{seed}.npz")
        per = {}
        for i, b, c in zip(z["row_index"], z["boxes"], z["confs"]):
            per.setdefault(int(i), []).append((float(c), b))
        ids = [str(c) for c in z["case_ids"]]
        box = {}
        for i, cid in enumerate(ids):
            cands = [(c, b) for c, b in per.get(i, []) if c >= 0.25]
            box[cid] = max(cands, key=lambda x: x[0])[1].tolist() if cands else None
        rows = []
        for kid in sorted({c.split("_z")[0] for c in ids}):
            pk = np.load(PACK / f"{kid}.npz")
            shape = tuple(int(v) for v in pk["shape"])
            zs = list(map(int, pk["zs"]))
            rd = sitk.ImageFileReader()
            rd.SetFileName(str(US / f"{kid}_imgUS.nii.gz"))
            rd.ReadImageInformation()
            sp = (rd.GetSpacing()[1], rd.GetSpacing()[0])  # (y,x)
            img_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(US / f"{kid}_imgUS.nii.gz")))
            for i, zv in enumerate(zs):
                cid = f"{kid}_z{zv:03d}"
                b = box.get(cid)
                gt = unpack(pk["G"][i], shape)
                img = img_vol[zv]
                if img.dtype != np.uint8:
                    img = ((img - img.min()) / max(img.max() - img.min(), 1e-6) * 255).astype(np.uint8)
                if b is None:
                    rows.append((cid, kid[:3], 0.0, 0))
                else:
                    pred = infer_sam_family(model, img, res, dev, box=b).astype(bool)
                    d, _ = score(pred, gt, sp)
                    rows.append((cid, kid[:3], float(d), int(pred.sum())))
        with open(MET / f"rev43_r2alt_s{seed}_medsam.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "patient", "dice", "pred_area_px"])
            for r in rows:
                w.writerow([r[0], r[1], f"{r[2]:.6f}", r[3]])
        print(f"done {seed} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
