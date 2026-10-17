#!/usr/bin/env python3
"""Aggregate the TTA round-level detection metrics from the raw npz dumps.

Produces outputs/summary/tta_r{round}_detection.json (the numbers quoted in
RUN_REPORT) from yolo_alldet_trusted_tta_r{round}_s{seed}.npz - the commits
had quoted these numbers before this producer was committed (audit m3).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m5_yolo_detect_metrics as M  # noqa: E402

ROOT = Path(__import__("os").environ.get("KIDNEYUS_ROOT",
          str(Path(__file__).resolve().parent.parent)))


def analyse_dump(npz_path):
    import SimpleITK as sitk
    z = np.load(npz_path)
    gt = np.load(ROOT / "outputs/prompts/p3_oracle_trusted.npz")
    gt_boxes, case_ids = gt["boxes"], list(gt["case_ids"])
    assert list(z["case_ids"]) == case_ids
    pole = set()
    ps = json.loads((ROOT / "outputs/figures/bimodal/pole_slice_stats.json").read_text())
    RAWM = ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA/US_masks/GT_estimated_masksUS"
    for kid, st in ps.items():
        v = sitk.GetArrayFromImage(sitk.ReadImage(str(RAWM / f"{kid}_maskUS.nii.gz"))) > 0.5
        areas = v.reshape(v.shape[0], -1).sum(1)
        nz = np.nonzero(areas)[0]
        med = np.median(areas[nz])
        for zz in nz[areas[nz] < 0.15 * med]:
            pole.add(f"{kid}_z{zz:03d}")
    by = {}
    for i, b, c in zip(z["row_index"], z["boxes"], z["confs"]):
        by.setdefault(int(i), []).append((float(c), b))
    n_eval = n_det = 0
    all_preds, ious25 = [], []
    for i, cid in enumerate(case_ids):
        if np.isnan(gt_boxes[i]).any():
            continue
        n_eval += 1
        cands = by.get(i, [])
        cs = np.array([c for c, _ in cands])
        bs = np.array([b for _, b in cands]) if cands else np.zeros((0, 4))
        ious = M.iou_vs(bs, gt_boxes[i]) if cs.size else np.zeros(0)
        for c, io in zip(cs, ious):
            all_preds.append((float(c), i, float(io)))
        if (cs >= 0.25).any():
            n_det += 1
            ious25.append(float(ious[int(np.argmax(cs))]))
    return {"no_box_rate": round(1 - n_det / n_eval, 4),
            "det_rate@0.25": round(n_det / n_eval, 4),
            "iou_median@0.25": round(float(np.median(ious25)), 4) if ious25 else None,
            "ap@0.5": round(M.ap_from_preds(all_preds, n_eval, 0.5), 4)}


def main():
    for rnd in (1, 2):
        out = {}
        for s in (42, 43, 44):
            p = ROOT / f"outputs/prompts/yolo_alldet_trusted_tta_r{rnd}_s{s}.npz"
            if not p.exists():
                continue
            out[f"s{s}"] = analyse_dump(p)
            print(f"r{rnd} s{s}: {out[f's{s}']}", flush=True)
        if out:
            nb = [v["no_box_rate"] for v in out.values()]
            out["round2_mean_no_box"] = round(float(np.mean(nb)), 4)
            (ROOT / f"outputs/summary/tta_r{rnd}_detection.json").write_text(
                json.dumps(out, indent=1))
            print(f"saved tta_r{rnd}_detection.json", flush=True)


if __name__ == "__main__":
    main()
