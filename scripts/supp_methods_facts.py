#!/usr/bin/env python3
"""Item ⑦ fact collection: numbers and parameters for the methods materials.

Computes:
  - YOLO detector training facts (from logs + registered scripts)
  - P4-TTA recipe facts (from m5_tta_round.py / m5_tta_aug.py + run logs)
  - self-training slice-selection bias: mean GT area of included vs excluded
    slices for R1 and each R2 replicate
  - compute cost per round from the run logs
Writes outputs/summary/supp_methods_facts.json.
"""
import json
import os
import re
import statistics
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "outputs/summary"
LOGS = ROOT / "outputs/logs"


def wall_minutes(log_path, tag):
    mins = []
    with open(log_path, errors="ignore") as f:
        for line in f:
            m = re.match(rf"\[(.+?)\] {tag} (?:train )?start", line)
            e = re.match(rf"\[(.+?)\] {tag} .*exit=", line)
            if m:
                mins.append(("s", line))
            if e:
                mins.append(("e", line))
    return mins


def round_cost():
    """Approximate per-replicate wall time from the wrapper log timestamps."""
    from datetime import datetime
    out = {}
    try:
        p = LOGS / "m5_tta_r1.log"
        stamps = []
        for line in open(p, errors="ignore"):
            m = re.search(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", line)
            if m:
                stamps.append((m.group(1), line.strip()[:60]))
        # report first & last for the training window
        out["r1_first_stamp"] = stamps[0] if stamps else None
        out["r1_last_stamp"] = stamps[-1] if stamps else None
    except FileNotFoundError:
        pass
    return out


def selection_bias():
    sl = pd.read_csv(ROOT / "outputs/trusted_us_slices.csv")
    area = {(r.kidney_id, int(r.slice_idx)): float(r.area_gt) for r in sl.itertuples()}
    res = {}
    for tag, d in (("r1_s42", ROOT / "outputs/p4tta/r1_s42/dataset/images"),
                   ("r2_s42", ROOT / "outputs/p4tta/r2_s42/dataset/images"),
                   ("r2_s43", ROOT / "outputs/p4tta/r2_s43/dataset/images"),
                   ("r2_s44", ROOT / "outputs/p4tta/r2_s44/dataset/images")):
        if not d.exists():
            continue
        kept = set()
        for f in d.glob("*.png"):
            m = re.match(r"p4tta_(.+)_z(\d+)\.png", f.name)
            if m:
                kept.add((m.group(1), int(m.group(2))))
        all_k = set(area)
        inc = [area[k] for k in all_k if k in kept]
        exc = [area[k] for k in all_k if k not in kept]
        res[tag] = {"n_kept": len(kept), "n_total": len(all_k),
                    "kept_fraction": round(len(kept) / len(all_k), 4),
                    "mean_area_kept": round(statistics.mean(inc), 1) if inc else None,
                    "mean_area_excluded": round(statistics.mean(exc), 1) if exc else None,
                    "median_area_kept": round(statistics.median(inc), 1) if inc else None,
                    "median_area_excluded": round(statistics.median(exc), 1) if exc else None}
    return res


def main():
    facts = {
        "yolo_detector": {
            "model": "YOLOv8s (ultralytics 8.3.108)",
            "training": "50 epochs, imgsz 640 (default), ultralytics default "
                        "geometric/color augmentation, seeds 42/43/44, "
                        "patient-level split 293/94/99 (train/val/test)",
            "operating_point": "conf >= 0.25 (ultralytics default, fixed at "
                               "registration); threshold-sensitivity analysis "
                               "(tau 0.10/0.15/0.25) shows the failure is "
                               "detection absence, not threshold placement",
            "source": "scripts/m2_fix_yolo_nnunet.sh, m1_split_okd.py, "
                      "m5_p4_conf_sensitivity.py, figures/yolo_detection/"},
        "p4_tta": {
            "optimizer": "Ultralytics 8.3.108 defaults (SGD momentum 0.937, "
                         "weight decay 5e-4) with lr0 = 1e-3 (10x below default)",
            "epochs": 15,
            "loss": "Ultralytics detection loss (box/cls/dfl) unchanged",
            "frozen_layers": "none - full fine-tune from the current weights",
            "augmentation": "registered defaults + per-image p=0.8 draw of 1+ "
                            "of six domain ops: console text, fan vignette, "
                            "gamma/contrast, vertical shadow, speckle, cutout "
                            "(parameters in scripts/m5_tta_aug.py)",
            "pseudo_labels": "top-1 box per slice at conf >= 0.50; slices "
                             "without a detection excluded from training "
                             "(neither positive nor negative)",
            "compute_cost": round_cost(),
            "source": "scripts/m5_tta_round.py, m5_tta_aug.py"},
        "self_training_selection": selection_bias(),
    }
    (OUT / "supp_methods_facts.json").write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
