#!/usr/bin/env python3
"""M2-prep: build the YOLO detection dataset for D1 from OKD capsule masks.

Plan §5-D1 / §6-E3: training labels are derived AUTOMATICALLY from the existing
segmentation masks (zero manual annotation). Boxes = bbox of the largest
connected component of the capsule mask, normalized to [0,1].
Splits come from the canonical outputs/splits.json (image filenames follow
their patient-level split).
"""
import csv
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_kidney import box_from_mask, load_json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(ROOT, "data/raw/OKD/kidneyUS_images_14_june_2022")
MASK_DIR = os.path.join(ROOT, "external_kidneyUS/labels/reviewed_masks_1/capsule")
SPLITS = load_json(os.path.join(ROOT, "outputs/splits.json"))
OUT = os.path.join(ROOT, "data/yolo")


def write_split(name, filenames):
    img_out = os.path.join(OUT, "images", name)
    lbl_out = os.path.join(OUT, "labels", name)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    n_box = 0
    for fn in filenames:
        img = Image.open(os.path.join(IMG_DIR, fn))
        w, h = img.size
        img.save(os.path.join(img_out, fn))
        mask = np.array(Image.open(os.path.join(MASK_DIR, fn)))
        box = box_from_mask(mask)
        lines = []
        if box is not None:
            x0, y0, x1, y1 = box
            cx = (x0 + x1) / 2.0 / w
            cy = (y0 + y1) / 2.0 / h
            bw = (x1 - x0 + 1) / w
            bh = (y1 - y0 + 1) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_box += 1
        with open(os.path.join(lbl_out, fn[:-4] + ".txt"), "w") as f:
            f.write("\n".join(lines))
    print(f"{name}: {len(filenames)} images, {n_box} boxes")
    return len(filenames), n_box


def main():
    totals = {}
    for split in ("train", "val", "test"):
        totals[split] = write_split(split, SPLITS[split])
    yaml_path = os.path.join(OUT, "okd.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {OUT}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n  0: kidney\n"
        )
    print("yaml:", yaml_path)


if __name__ == "__main__":
    main()
