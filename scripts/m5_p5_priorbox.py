#!/usr/bin/env python3
"""Amendment 1 - Arm P5: training-free prior-box prompts (DIAGNOSTIC).

LOCKED until the OSF amendment timestamp is recorded in RUN_REPORT.
Do not execute before then (pre-registration gate, RUN_REPORT 2026-10-11).

Spec (docs/tta_amendment_v1.md section 3):
- box source: binarized predictions of the registered nnU-Net v2 (2d) on
  TRUSTED, seed 42 (median internal-val Dice 0.9486 of {0.9486, 0.9484,
  0.9519}; primary caliber = best checkpoint);
- box derivation: per slice, LCC tight bbox via the registered
  utils_kidney.box_from_mask (pad=0), native grid; empty prediction ->
  no box -> strict Dice = 0;
- evaluation: MedSAM seed 42 (median internal-val 0.9627 of {0.9627,
  0.9639, 0.9625}), single pass, all 18,322 slices, registered scorer;
- report: coverage, box IoU vs oracle, strict Dice;
- reporting criterion: "viable alternative prompt source" iff coverage
  >= 0.75 AND median box-IoU-vs-oracle >= 0.70 AND strict Dice >= 0.55.

Compliance: reads TRUSTED IMAGES ONLY; ground truth enters via
m3_common.iter_trusted's registered scorer path only.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
from m3_common import iter_trusted, load_sam_family, score, \
    infer_sam_family, MODEL_RES  # noqa: E402
from utils_kidney import box_from_mask  # noqa: E402

ROOT = Path(__import__("os").environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
PRED = ROOT / "outputs/predictions/nnunet_seed42_best/trusted"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CRIT = {"coverage": 0.75, "iou_median": 0.70, "dice": 0.55}


def main():
    # ---- pass 1: derive P5 boxes from nnU-Net seed42 predictions (images only)
    boxes, ids = {}, []
    for cid, _img, _gt, _sp, meta in iter_trusted():
        pred = np.squeeze(sitk_read(PRED / f"{cid}.nii.gz")) > 0
        ids.append(cid)
        boxes[cid] = box_from_mask(pred)  # None when prediction empty
    np.savez_compressed(ROOT / "outputs/prompts/p5_box_trusted.npz",
                        case_ids=np.array(ids),
                        boxes=np.array([np.asarray(boxes[c], dtype=np.float32)
                                        if boxes[c] is not None else np.full(4, np.nan)
                                        for c in ids]))
    coverage = sum(1 for c in ids if boxes[c] is not None) / len(ids)

    # ---- box IoU vs oracle (oracle boxes cached by the registered audit)
    gt = np.load(ROOT / "outputs/prompts/p3_oracle_trusted.npz")
    gt_of = {c: gt["boxes"][i] for i, c in enumerate(gt["case_ids"])}
    ious = []
    for c in ids:
        if boxes[c] is None or np.isnan(gt_of[c]).any():
            continue
        a, b = np.asarray(boxes[c], dtype=float), gt_of[c]
        iw = max(min(a[2], b[2]) - max(a[0], b[0]), 0)
        ih = max(min(a[3], b[3]) - max(a[1], b[1]), 0)
        inter = iw * ih
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        ious.append(inter / union if union > 0 else 0.0)
    iou_med = float(np.median(ious)) if ious else float("nan")

    # ---- pass 2: MedSAM seed42, single pass, registered scorer
    model = load_sam_family("medsam", 42, DEVICE)
    res = MODEL_RES["medsam"]
    dices = []
    for cid, img, gtm, spacing, _meta in iter_trusted():
        if boxes[cid] is None:
            dices.append(0.0)  # registered empty-prompt rule
            continue
        pred = infer_sam_family(model, img, res, DEVICE, box=boxes[cid])
        d, _h = score(pred.astype(bool), gtm, spacing)
        dices.append(float(d))

    import statistics as st
    out = {
        "n": len(ids), "coverage": round(coverage, 4),
        "box_iou_vs_oracle_median": round(iou_med, 4) if ious else None,
        "strict_dice_mean": round(st.mean(dices), 4),
        "strict_dice_std": round(st.stdev(dices), 4),
        "criterion": {"coverage>=0.75": coverage >= CRIT["coverage"],
                      "iou_median>=0.70": bool(iou_med >= CRIT["iou_median"]),
                      "dice>=0.55": st.mean(dices) >= CRIT["dice"]},
        "verdict": "viable alternative prompt source" if (
            coverage >= CRIT["coverage"] and iou_med >= CRIT["iou_median"]
            and st.mean(dices) >= CRIT["dice"]) else "diagnostic/negative",
        "seeds": {"nnunet_box_source": 42, "medsam_eval": 42},
    }
    (ROOT / "outputs/summary/p5_priorbox.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def sitk_read(path):
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


if __name__ == "__main__":
    main()
