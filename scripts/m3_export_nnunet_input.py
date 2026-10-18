#!/usr/bin/env python3
"""Export OKD test cases and TRUSTED effective slices as nifti inputs for
nnUNetv2_predict (spacing 1,1,1 — nnU-Net 2d config is spacing-agnostic for
single-slice data; the canonical 2d plans were built on this geometry)."""
import os
import sys

import numpy as np
import SimpleITK as sitk

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_common as M  # noqa: E402

OUT = os.path.join(ROOT, "scratch", "nnunet_eval_input")


def write_nifti(arr, path):
    im = sitk.GetImageFromArray(arr.astype(np.uint8))
    im.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(im, path, useCompression=True)


def main():
    for d in ("okd", "trusted"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)

    # nnUNetv2_predict requires the _0000 channel suffix in input filenames
    n = 0
    for cid, img, _gt, _sp, _meta in M.iter_okd_test():
        write_nifti(img, os.path.join(OUT, "okd", f"{cid[:-4]}_0000.nii.gz"))
        n += 1
    print("okd inputs:", n)

    n = 0
    for kid, vol, slices in iter_trusted_volumes():
        for z, cid in slices:
            write_nifti(vol[z], os.path.join(OUT, "trusted", f"{cid}_0000.nii.gz"))
            n += 1
    print("trusted inputs:", n)


def iter_trusted_volumes():
    import csv
    import SimpleITK as sitk

    path = os.path.join(ROOT, "outputs/trusted_us_slices.csv")
    by_kid = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            by_kid.setdefault(r["kidney_id"], []).append(int(r["slice_idx"]))
    for kid in sorted(by_kid):
        img = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(M.TRUSTED, "US_images", f"{kid}_imgUS.nii.gz")))
        slices = [(z, f"{kid}_z{z:03d}") for z in sorted(by_kid[kid])]
        yield kid, img, slices


if __name__ == "__main__":
    main()
