#!/usr/bin/env python3
"""M5 P1-1/P1-2 offline analysis: YOLO detection quality on TRUSTED.

Consumes the npz dumps written by m5_yolo_alldet_trusted.py (all candidate
boxes+confs per slice, conf floor 0.01) plus the cached oracle GT boxes
(STAPLE-derived). Per seed:
  - detection rate at conf in {0.10, 0.15, 0.25}, overall and per kidney;
  - IoU(top-1 box @0.25, GT) distribution;
  - AP@[0.5] and AP@[0.50:0.95], single class, all candidates ranked,
    first detection with IoU>=thr claims the slice's GT, standard
    all-point interpolation;
  - P4 conf-sensitivity: how many slices change their top-1 box (or gain one)
    when the threshold moves 0.25 -> 0.15 / 0.10 = the exact re-segmentation
    workload for re-running the P4 arm at a lower threshold.
Writes outputs/metrics/yolo_detection_trusted.csv (case-level) and
outputs/figures/yolo_detection/summary.json (aggregate, safe to commit).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__import__("os").environ.get("KIDNEYUS_ROOT",
          str(Path(__file__).resolve().parent.parent)))
PROMPTS = ROOT / "outputs/prompts"
FIGS = ROOT / "outputs/figures/yolo_detection"
TAUS = (0.10, 0.15, 0.25)


def iou_vs(a, gb):
    """a: (N,4), gb: (4,) xyxy -> (N,)."""
    if len(a) == 0:
        return np.zeros(0)
    iw = np.clip(np.minimum(a[:, 2], gb[2]) - np.maximum(a[:, 0], gb[0]), 0, None)
    ih = np.clip(np.minimum(a[:, 3], gb[3]) - np.maximum(a[:, 1], gb[1]), 0, None)
    inter = iw * ih
    union = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
             + (gb[2] - gb[0]) * (gb[3] - gb[1]) - inter)
    return np.where(union > 0, inter / union, 0.0)


def ap_from_preds(preds, n_gt, thr):
    """preds: list of (conf, slice_idx, iou); first candidate with IoU>=thr
    claims that slice's GT (one GT per slice). Standard all-point AP."""
    if n_gt == 0:
        return float("nan")
    preds = sorted(preds, key=lambda x: -x[0])
    claimed = set()
    hits = []
    for conf, sl, iou in preds:
        if iou >= thr and sl not in claimed:
            hits.append(True)
            claimed.add(sl)
        else:
            hits.append(False)
    tp = np.cumsum(hits)
    fp = np.cumsum([not h for h in hits])
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / n_gt
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def analyse(seed):
    z = np.load(PROMPTS / f"yolo_alldet_trusted_seed{seed}.npz")
    gt = np.load(PROMPTS / "p3_oracle_trusted.npz")
    case_ids = list(z["case_ids"])
    gt_boxes = gt["boxes"]
    assert list(gt["case_ids"]) == case_ids, "GT/case order mismatch"
    per_slice = defaultdict(list)
    for i, b, c in zip(z["row_index"], z["boxes"], z["confs"]):
        per_slice[int(i)].append((float(c), b.tolist()))

    # pole-slice flags (annotation-range edges, P1-3 finding)
    pole = set()
    ps = ROOT / "outputs/figures/bimodal/pole_slice_stats.json"
    if ps.exists():
        import SimpleITK as sitk
        RAWM = ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA/US_masks/GT_estimated_masksUS"
        for kid, st in json.loads(ps.read_text()).items():
            v = sitk.GetArrayFromImage(sitk.ReadImage(str(RAWM / f"{kid}_maskUS.nii.gz"))) > 0.5
            areas = v.reshape(v.shape[0], -1).sum(1)
            nz = np.nonzero(areas)[0]
            med = np.median(areas[nz])
            for z in nz[areas[nz] < 0.15 * med]:
                pole.add(f"{kid}_z{z:03d}")
    kid_slices = defaultdict(int)     # kidney -> n slices with GT
    kid_detected = defaultdict(int)   # kidney -> detected @0.25
    all_preds = []                    # (conf, iou) for AP
    iou_at_25 = []
    box_change = {0.10: 0, 0.15: 0}
    n_eval = 0
    n_eval_pole = 0
    det_pole = defaultdict(int)   # tau -> detections among pole slices
    for i, cid in enumerate(case_ids):
        gb = gt_boxes[i]
        if np.isnan(gb).any():
            continue
        n_eval += 1
        is_pole = cid in pole
        n_eval_pole += is_pole
        kid = cid.split("_z")[0]
        kid_slices[kid] += 1
        cands = per_slice.get(i, [])
        cs = np.array([c for c, _ in cands])
        bs = np.array([b for _, b in cands]) if cands else np.zeros((0, 4))
        ious = iou_vs(bs, gb)
        for c, io in zip(cs, ious):
            all_preds.append((float(c), i, float(io)))
        for t in TAUS:
            if is_pole and (cs >= t).any():
                det_pole[t] += 1
        has25 = (cs >= 0.25).any()
        if has25:
            kid_detected[kid] += 1
            iou_at_25.append(float(ious[int(np.argmax(cs))]))
        top25 = int(np.argmax(cs)) if has25 else -1
        for t in (0.10, 0.15):
            has = (cs >= t).any()
            topt = int(np.argmax(cs)) if has else -1
            if topt != top25:
                box_change[t] += 1

    res = {"seed": seed, "n_slices_with_gt": n_eval}
    res["pole_slices"] = n_eval_pole
    res["det_rate_pole"] = {f"{t:.2f}": round(det_pole[t] / max(n_eval_pole, 1), 4) for t in TAUS}
    for t in TAUS:
        n_det = sum(1 for cs, _ in [([c for c, _ in per_slice.get(i, [])], None)
                                    for i in range(len(case_ids))
                                    if not np.isnan(gt_boxes[i]).any()]
                    if np.asarray(cs).size and max(cs) >= t)
        res[f"det_rate@{t:.2f}"] = round(n_det / n_eval, 4)
    v = np.asarray(iou_at_25)
    res["iou_top1@0.25_mean"] = round(float(v.mean()), 4)
    res["iou_top1@0.25_median"] = round(float(np.median(v)), 4)
    res["iou_top1@0.25_lt0.5_frac"] = round(float((v < 0.5).mean()), 4)
    res["ap@0.5"] = round(ap_from_preds(all_preds, n_eval, 0.5), 4)
    res["ap@0.50:0.95"] = round(float(np.mean(
        [ap_from_preds(all_preds, n_eval, t) for t in np.arange(0.5, 0.96, 0.05)])), 4)
    res["p4_reseg_workload"] = {f"tau={t:.2f}": {"slices_with_changed_or_new_box": box_change[t],
                                                 "frac_of_eval": round(box_change[t] / n_eval, 4)}
                                for t in (0.10, 0.15)}
    rates = {k: kid_detected[k] / kid_slices[k] for k in kid_slices}
    res["per_kidney_det_rate@0.25"] = {
        "kidneys_full": sum(1 for x in rates.values() if x >= 0.999),
        "kidneys_partial": sum(1 for x in rates.values() if 0 < x < 0.999),
        "kidneys_zero": sum(1 for x in rates.values() if x == 0),
        "worst5": sorted((round(x, 3), k) for k, x in rates.items())[:5],
    }
    return res


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    out = []
    for seed in (42, 43, 44):
        try:
            res = analyse(seed)
        except FileNotFoundError:
            print(f"seed{seed}: npz missing, skipped")
            continue
        out.append(res)
        print(json.dumps(res, indent=1))
    if out:
        (FIGS / "summary.json").write_text(json.dumps(out, indent=1))
        print(f"saved -> {FIGS}/summary.json")


if __name__ == "__main__":
    main()
