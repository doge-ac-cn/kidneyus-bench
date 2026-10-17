#!/usr/bin/env python3
"""M0-TRUSTED verification, v2 (real archive layout, plan §3 gates 2 & 4).

Actual layout (verified 2026-10-08):
  US_DATA/US_images/{kidney}_imgUS.nii.gz            59 volumes, 1 kidney each
  US_DATA/US_masks/{Annotator1,Annotator2,GT_estimated_masksUS}/...
  US_DATA/US_landmarks/{Annotator1,Annotator2,GT_estimated_ldksUS}/...txt (7 points)
  CT_DATA/...                                        48 CTs, 2 kidneys each
Kidney naming: {patient}{L|R}; GT_estimated_* = STAPLE consensus (paper §Data).
Official 5-fold splits live in README.txt (parsed here for later stages).
"""
import json
import os
import re
import sys

import numpy as np
import SimpleITK as sitk

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main(data_root):
    us_img = os.path.join(data_root, "US_DATA/US_images")
    us_masks = os.path.join(data_root, "US_DATA/US_masks")
    us_ldk = os.path.join(data_root, "US_DATA/US_landmarks")
    ct_img = os.path.join(data_root, "CT_DATA/CT_images")
    ct_masks = os.path.join(data_root, "CT_DATA/CT_masks")

    summary = {}

    # ---- US volumes ----
    vols = sorted(f for f in os.listdir(us_img) if f.endswith(".nii.gz"))
    kidneys = [f.replace("_imgUS.nii.gz", "") for f in vols]
    patients = sorted({re.sub(r"[LR]$", "", k) for k in kidneys})
    summary["us"] = {
        "n_volumes": len(vols),
        "n_kidneys": len(kidneys),
        "n_patients": len(patients),
        "kidneys": kidneys,
        "multi_kidney_patients": len(kidneys) - len(patients),
    }

    spacings, sizes = [], []
    for f in vols:
        img = sitk.ReadImage(os.path.join(us_img, f))
        spacings.append(img.GetSpacing())
        sizes.append(img.GetSize())
    sp, sz = np.array(spacings), np.array(sizes)
    summary["us"]["spacing_mean"] = sp.mean(axis=0).round(4).tolist()
    summary["us"]["spacing_range"] = [sp.min(axis=0).round(4).tolist(), sp.max(axis=0).round(4).tolist()]
    summary["us"]["size_range"] = [sz.min(axis=0).tolist(), sz.max(axis=0).tolist()]

    # ---- US masks: presence + binarity + STAPLE consensus ----
    for sub in ("Annotator1", "Annotator2", "GT_estimated_masksUS"):
        d = os.path.join(us_masks, sub)
        files = sorted(os.listdir(d))
        bin_checks = []
        for f in files[:8]:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(d, f)))
            bin_checks.append(sorted(np.unique(arr).tolist())[:4])
        summary["us"][f"masks_{sub}"] = {"n": len(files), "unique_values_sample": bin_checks[:4]}

    # GT/consensus image geometry must match US images
    gt = sitk.ReadImage(os.path.join(us_masks, "GT_estimated_masksUS", vols[0].replace("_imgUS", "_maskUS")))
    im = sitk.ReadImage(os.path.join(us_img, vols[0]))
    summary["us"]["gt_geometry_matches_image"] = (gt.GetSize() == im.GetSize()
                                                  and np.allclose(gt.GetSpacing(), im.GetSpacing()))

    # ---- landmarks: 7 points ----
    ldk_counts = {}
    for sub in ("Annotator1", "Annotator2", "GT_estimated_ldksUS"):
        d = os.path.join(us_ldk, sub)
        counts = []
        for f in sorted(os.listdir(d)):
            with open(os.path.join(d, f)) as fh:
                counts.append(sum(1 for line in fh if line.strip()))
        ldk_counts[sub] = {"n_files": len(counts),
                           "points_per_file_unique": sorted(set(counts))}
    summary["us"]["landmarks"] = ldk_counts

    # ---- CT ----
    cts = sorted(f for f in os.listdir(ct_img) if f.endswith(".nii.gz"))
    summary["ct"] = {"n_images": len(cts)}
    m1 = os.listdir(os.path.join(ct_masks, "Annotator1"))
    mgt = os.listdir(os.path.join(ct_masks, "GT_estimated_masksCT"))
    summary["ct"]["masks_annotator1"] = len(m1)
    summary["ct"]["masks_gt"] = len(mgt)
    # label values in a CT GT mask (expect two kidneys: labels 1 and 2)
    arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(ct_masks, "GT_estimated_masksCT", mgt[0])))
    summary["ct"]["gt_label_values_sample"] = sorted(np.unique(arr).tolist())[:6]

    # ---- official 5-fold splits from README.txt ----
    readme = open(os.path.join(data_root, "README.txt")).read()
    us_folds = dict(re.findall(r"US_cv(\d) = (\[[^\]]*\])", readme))
    folds = {k: [x.strip().strip("'") for x in v.strip("[]").split(",")] for k, v in us_folds.items()}
    summary["us"]["official_5fold_test_kidneys"] = {k: len(v) for k, v in folds.items()}
    total_fold = sum(len(v) for v in folds.values())
    summary["us"]["official_5fold_total"] = total_fold
    # save folds for M4 reverse-3D
    with open(os.path.join(ROOT, "outputs/trusted_official_5fold.json"), "w") as f:
        json.dump(folds, f, indent=2)

    # ---- data use agreement archived ----
    dua = os.path.join(data_root, "data_use_agreement.txt")
    if os.path.exists(dua):
        dst = os.path.join(ROOT, "outputs/trusted_data_use_agreement_archive.txt")
        with open(dua, "rb") as fi, open(dst, "wb") as fo:
            fo.write(fi.read())
        summary["data_use_agreement_archived"] = dst

    out = os.path.join(ROOT, "outputs/trusted_verify_summary.json")
    # T0-6 (v1.4): the repo-visible artifact must stay free of kidney IDs
    # (xxxL/R); keep counts only. The full list stays in the local run only.
    public = json.loads(json.dumps(summary))
    public["us"].pop("kidneys", None)
    public["us"].pop("multi_kidney_patients", None)
    with open(out, "w") as f:
        json.dump(public, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
