#!/usr/bin/env python3
"""Score nnU-Net predictions (from m3_nnunet_predict.sh) against GT and append
rows to e1_internal.csv / e2_external.csv (prompt = "none")."""
import argparse
import os
import sys

import numpy as np
import SimpleITK as sitk

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_common as M  # noqa: E402
from utils_kidney import load_json  # noqa: E402

PRED_BASE = os.path.join(ROOT, "outputs/predictions")


def read_bool(path):
    if not os.path.exists(path):
        return None
    return sitk.GetArrayFromImage(sitk.ReadImage(path)) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--datasets", default="okd,trusted")
    ap.add_argument("--chk", default="checkpoint_final",
                    help="which checkpoint the predictions came from")
    ap.add_argument("--tag", default="final",
                    help="final -> sensitivity CSVs (e1/e2_internal_sens_final); "
                         "best -> primary CSVs (T1-2: best-val is the primary口径)")
    ap.add_argument("--pred-base", default=None,
                    help="override prediction base dir (e.g. .../nnunet_seedN_best)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    pred_base = args.pred_base or os.path.join(PRED_BASE)
    if args.tag == "final":
        csv1, csv2 = "e1_internal_sens_final.csv", "e2_external_sens_final.csv"
    else:
        csv1, csv2 = "e1_internal.csv", "e2_external.csv"

    if "okd" in args.datasets:
        for seed in seeds:
            pdir = args.pred_base.replace("{seed}", str(seed)) if args.pred_base else                 os.path.join(PRED_BASE, f"nnunet_seed{seed}", "okd")
            pdir = os.path.join(pdir, "okd") if not pdir.endswith("okd") else pdir
            n = 0
            for cid, _img, gt, _sp, _meta in M.iter_okd_test():
                if args.limit and n >= args.limit:
                    break
                # nnU-Net output drops the _0000 suffix AND our cid carries .png
                pred = read_bool(os.path.join(pdir, f"{cid[:-4]}.nii.gz"))
                if pred is None:
                    print(f"[warn] missing prediction for {cid}, skipped")
                    continue
                if pred.shape != gt.shape:
                    print(f"[warn] shape mismatch {cid}: {pred.shape} vs {gt.shape}, skipped")
                    continue
                d, h = M.score(pred, gt, (1.0, 1.0))
                M3_append(csv1,
                          ["nnunet", seed, "none", cid, f"{d:.6f}", f"{h:.6f}"])
                n += 1
            print(f"[okd] nnunet seed{seed}: {n} rows")

    if "trusted" in args.datasets:
        for seed in seeds:
            pdir = (args.pred_base.replace("{seed}", str(seed)) if args.pred_base else
                    os.path.join(PRED_BASE, f"nnunet_seed{seed}", "trusted"))
            pdir = os.path.join(pdir, "trusted") if not pdir.endswith("trusted") else pdir
            n = 0
            for kid, _vol, gt, sp, slices in iter_trusted_volumes():
                for z, cid in slices:
                    if args.limit and n >= args.limit:
                        break
                    pred = read_bool(os.path.join(pdir, f"{cid}.nii.gz"))
                    if pred is None or pred.shape != gt[z].shape:
                        print(f"[warn] missing/shape-mismatch prediction {cid}")
                        continue
                    d, h = M.score(pred, gt[z], sp)
                    import re as _re
                    M3_append(csv2,
                              ["nnunet", seed, "none", _re.sub(r"[LR]$", "", kid), kid, z, cid,
                               f"{d:.6f}", f"{h:.6f}"])
                    n += 1
                if args.limit and n >= args.limit:
                    break
            print(f"[trusted] nnunet seed{seed}: {n} rows")


HEADERS = {
    "e1_internal.csv": ["model", "seed", "prompt", "case_id", "dice", "hd95_px"],
    "e2_external.csv": ["model", "seed", "prompt", "patient", "kidney_id",
                        "slice_idx", "case_id", "dice", "hd95_mm"],
    "e1_internal_sens_final.csv": ["model", "seed", "prompt", "case_id", "dice", "hd95_px"],
    "e2_external_sens_final.csv": ["model", "seed", "prompt", "patient", "kidney_id",
                                   "slice_idx", "case_id", "dice", "hd95_mm"],
}


def M3_append(csv_name, row):
    os.makedirs(os.path.join(ROOT, "outputs/metrics"), exist_ok=True)
    path = os.path.join(ROOT, "outputs/metrics", csv_name)
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write(",".join(HEADERS[csv_name]) + "\n")
        f.write(",".join(str(x) for x in row) + "\n")


def iter_trusted_volumes():
    import csv
    import SimpleITK as sitk

    path = os.path.join(ROOT, "outputs/trusted_us_slices.csv")
    by_kid = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            by_kid.setdefault(r["kidney_id"], []).append(int(r["slice_idx"]))
    for kid in sorted(by_kid):
        gt = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(M.TRUSTED, "US_masks", "GT_estimated_masksUS",
                         f"{kid}_maskUS.nii.gz"))) > 0.5
        sp_im = sitk.ReadImage(os.path.join(M.TRUSTED, "US_images", f"{kid}_imgUS.nii.gz"))
        sp = (float(sp_im.GetSpacing()[1]), float(sp_im.GetSpacing()[0]))
        slices = [(z, f"{kid}_z{z:03d}") for z in sorted(by_kid[kid])]
        yield kid, None, gt, sp, slices


if __name__ == "__main__":
    main()
