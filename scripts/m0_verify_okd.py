#!/usr/bin/env python3
"""M0-OKD verification (plan §3 gate 1) -> canonical cohort manifest.

Reads the distributed image zip contents + kidneyUS repo labels, and writes
outputs/okd_cohort.csv with one row per usable case:
  filename, patient_key, view, quality, n_regions_capsule(L1/L2), width, height,
  spacing_x_mm (from TransducerInfo xlsx, unit-ambiguous -> see RUN_REPORT)

M0 gate facts established 2026-10-08 (see RUN_REPORT):
  - 534 images distributed = 514 unique images + 20 intra-rater duplicate
    captures (per the dataset's official duplicate/copy list); the paper/plan
    number 514 refers to the unique-image count, NOT to any xlsx row count
    (TransducerInfo xlsx row count is unrelated to 514/534 -- earlier
    explanation here was wrong, corrected 2026-10-11 per checklist P4-12)
  - 487 images carry >=1 Capsule polygon from rater 1; cohort below requires a
    non-empty capsule mask from BOTH raters
  - patient key = full IM-XXXX-YYYY filename component (468 unique, 9 multi-image;
    2026-10-09 dHash adjudication: same IM-XXXX prefix but different full key =
    visually distinct images, min distance 45/256 -> full key is the case ID)
  - no native/transplant field anywhere in the distribution -> plan §10 rule 3:
    E5 subgroup analysis is dropped
"""
import csv
import json
import os
import sys

import numpy as np
import openpyxl
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMG_DIR = os.path.join(ROOT, "data/raw/OKD/kidneyUS_images_14_june_2022")
LBL_DIR = os.path.join(ROOT, "external_kidneyUS/labels")
XLSX = os.path.join(ROOT, "external_kidneyUS/OpenKidneyUltrasoundDataSet_TransducerInfo.xlsx")
OUT_CSV = os.path.join(ROOT, "outputs/okd_cohort.csv")
OUT_JSON = os.path.join(ROOT, "outputs/okd_cohort_summary.json")


def load_polygons(path):
    per_file = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ra = json.loads(row["region_attributes"])
            fa = json.loads(row["file_attributes"])
            rec = per_file.setdefault(
                row["filename"],
                {"capsule": 0, "view": fa.get("View", ""), "quality": fa.get("Quality", "")},
            )
            if ra.get("Anatomy") == "Capsule":
                rec["capsule"] += 1
    return per_file


def main():
    summary = {}

    images = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith(".png"))
    summary["n_images_distributed"] = len(images)

    # transducer / spacing sheet (paper's 514-row table)
    wb = openpyxl.load_workbook(XLSX, read_only=True)
    ws = wb.worksheets[0]
    spacing = {}
    header = None
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i == 0:
            header = row
            continue
        key = str(row[0]).strip()
        dx, dy = row[4], row[5]
        spacing[key] = {"dx": float(dx) if dx is not None else None,
                        "dy": float(dy) if dy is not None else None}
    summary["n_rows_transducer_xlsx"] = len(spacing)

    lab1 = load_polygons(os.path.join(LBL_DIR, "reviewed_labels_1.csv"))
    lab2 = load_polygons(os.path.join(LBL_DIR, "reviewed_labels_2.csv"))
    m1dir = os.path.join(LBL_DIR, "reviewed_masks_1/capsule")
    m2dir = os.path.join(LBL_DIR, "reviewed_masks_2/capsule")
    masks1 = set(os.listdir(m1dir))
    masks2 = set(os.listdir(m2dir))

    cohort = []
    missing_mask, empty_mask = [], []
    for fn in images:
        if fn not in lab1 or fn not in lab2:
            continue
        if fn not in masks1 or fn not in masks2:
            missing_mask.append(fn)
            continue
        a1 = np.array(Image.open(os.path.join(m1dir, fn)))
        a2 = np.array(Image.open(os.path.join(m2dir, fn)))
        if a1.max() == 0 or a2.max() == 0:
            empty_mask.append(fn)
            continue
        stem = fn[:-4]
        im_key = stem.split("_")[1] if "_" in stem else ""
        img = Image.open(os.path.join(IMG_DIR, fn))
        sp = spacing.get(stem, {})
        cohort.append({
            "filename": fn,
            "patient_key": im_key,
            "view": lab1[fn]["view"],
            "quality": lab1[fn]["quality"],
            "n_capsule_l1": lab1[fn]["capsule"],
            "n_capsule_l2": lab2[fn]["capsule"],
            "width": img.size[0],
            "height": img.size[1],
            "spacing_dx_xlsx": sp.get("dx", ""),
            "spacing_dy_xlsx": sp.get("dy", ""),
        })

    uniq_patients = sorted({c["patient_key"] for c in cohort})
    multi = {}
    for c in cohort:
        multi.setdefault(c["patient_key"], []).append(c["filename"])
    n_multi = sum(1 for v in multi.values() if len(v) > 1)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cohort[0].keys()))
        w.writeheader()
        w.writerows(cohort)

    summary.update({
        "n_cohort_both_raters_nonempty": len(cohort),
        "missing_mask_files": len(missing_mask),
        "empty_mask_files": len(empty_mask),
        "n_unique_patient_keys": len(uniq_patients),
        "n_patient_keys_with_multiple_images": n_multi,
        "split_granularity": "patient_key (IM-XXXX)",
        "native_transplant_field": "absent in distribution -> E5 subgroup dropped (plan §10 rule 3)",
        "mask_values": "binary {0,1}",
        "view_distribution": {v: sum(1 for c in cohort if c["view"] == v)
                              for v in {c["view"] for c in cohort}},
    })
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
