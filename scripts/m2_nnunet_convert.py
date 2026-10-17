#!/usr/bin/env python3
"""M2: convert the OKD cohort to nnU-Net v2 raw format for M2 (plan §5-M2).

- images: capsule task, rater-1 masks as labels (primary GT; documented)
- 2D nifti (HxWx1), spacing (1,1,1) mm — OKD's xlsx spacing is unit-ambiguous
  (documented in RUN_REPORT), so nnU-Net sees isotropic in-plane spacing
- our canonical patient-level split is imposed via splits_final.json (fold 0),
  NOT nnU-Net's internal random 5-fold
- custom seed trainers live in scripts/nnunet_trainers.py
"""
import json
import os
import sys

import numpy as np
import SimpleITK as sitk
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_kidney import load_json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(ROOT, "data/raw/OKD/kidneyUS_images_14_june_2022")
MASK_DIR = os.path.join(ROOT, "external_kidneyUS/labels/reviewed_masks_1/capsule")
SPLITS = load_json(os.path.join(ROOT, "outputs/splits.json"))
RAW = os.path.join(ROOT, "data/nnunet/nnUNet_raw/Dataset001_KidneyUS")


def main():
    os.makedirs(os.path.join(RAW, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(RAW, "imagesTs"), exist_ok=True)
    os.makedirs(os.path.join(RAW, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(RAW, "labelsTs"), exist_ok=True)

    def conv(fn, out_img, out_lbl):
        img = np.array(Image.open(os.path.join(IMG_DIR, fn)).convert("L"))
        msk = (np.array(Image.open(os.path.join(MASK_DIR, fn))) > 0).astype(np.uint8)
        for arr, path, is_label in ((img, out_img, False), (msk, out_lbl, True)):
            im = sitk.GetImageFromArray(arr[None].astype(np.uint8))
            im.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(im, path, useCompression=True)

    train_val = sorted(SPLITS["train"] + SPLITS["val"])
    test = sorted(SPLITS["test"])
    for fn in train_val:
        case = fn[:-4]
        conv(fn, os.path.join(RAW, "imagesTr", f"{case}_0000.nii.gz"),
             os.path.join(RAW, "labelsTr", f"{case}.nii.gz"))
    for fn in test:
        case = fn[:-4]
        conv(fn, os.path.join(RAW, "imagesTs", f"{case}_0000.nii.gz"),
             os.path.join(RAW, "labelsTs", f"{case}.nii.gz"))

    dataset_json = {
        "channel_names": {"0": "ultrasound"},
        "labels": {"background": 0, "kidney": 1},
        "numTraining": len(train_val),
        "file_ending": ".nii.gz",
        "name": "Dataset001_KidneyUS",
        "description": "OKD kidney-capsule segmentation, patient-level split (plan A v1.0)",
    }
    with open(os.path.join(RAW, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    # impose the canonical split as nnU-Net fold 0
    splits_final = [{
        "train": [f[:-4] for f in SPLITS["train"]],
        "val": [f[:-4] for f in SPLITS["val"]],
    }]
    pp_dir = os.path.join(ROOT, "data/nnunet/nnUNet_preprocessed/Dataset001_KidneyUS")
    os.makedirs(pp_dir, exist_ok=True)
    with open(os.path.join(pp_dir, "splits_final.json"), "w") as f:
        json.dump(splits_final, f, indent=2)

    print(f"converted: {len(train_val)} train+val, {len(test)} test cases -> {RAW}")


if __name__ == "__main__":
    main()
