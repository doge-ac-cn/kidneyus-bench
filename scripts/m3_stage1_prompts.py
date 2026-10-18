#!/usr/bin/env python3
"""M3 stage 1: derive the automatic prompts for E3 (plan §6, v1.2 rules).

  P1/P2: from the M1 U-Net (median-val seed, pre-specified = seed43) predicted
         mask on each case -> largest-CC box / centroid point
  P4:    from the D1 YOLO detector (median-val-mAP seed, pre-specified = seed43)
         highest-confidence box per image
  D1 detection metrics: per-case IoU against the GT-derived (oracle) box for
         both datasets -> outputs/metrics/e3_detection.csv

Outputs: outputs/prompts/{p1_box,p2_point,p4_box}_{okd,trusted}.npz
Prompt missing (empty prediction / no detection) is stored as masked None.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_kidney import box_from_mask, load_json  # noqa: E402

import m3_common as M  # noqa: E402

M1_PRIOR_SEED = 43   # median val Dice (0.7092 / 0.6975 / 0.5655) — plan §14 v1.2
D1_SEED = 43         # median val mAP50 (0.993 / 0.994 / 0.995) — same rule


def save_prompts(dataset, p1, p2, p4):
    os.makedirs(M.PROMPT_DIR, exist_ok=True)
    for kind, d in (("p1_box", p1), ("p2_point", p2), ("p4_box", p4)):
        M.save_prompt_npz(os.path.join(M.PROMPT_DIR, f"{kind}_{dataset}.npz"), d)
    print(f"[{dataset}] prompts saved: {len(p1)} cases")


@torch.no_grad()
def m1_predict_all(model, cases, device):
    """cases: iterable of (case_id, img, gt, spacing, meta). Returns
    {case_id: pred_bool}."""
    out = {}
    for cid, img, _gt, _sp, _meta in cases:
        out[cid] = M.infer_unet(model, img, device)
    return out


def derive_p1p2(pred):
    from utils_kidney import centroid_from_mask
    box = box_from_mask(pred)
    if box is None:
        return None, None
    return box, centroid_from_mask(pred)


def yolo_predict(model, img, device="cpu"):
    img_bgr = np.stack([img] * 3, axis=-1)  # ultralytics expects HWC (gray -> 3ch)
    r = model.predict(img_bgr, verbose=False, conf=0.25, device=device)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None, 0.0
    confs = r.boxes.conf.cpu().numpy()
    i = int(np.argmax(confs))
    return r.boxes.xyxy.cpu().numpy()[i], float(confs[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="okd,trusted")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N cases only")
    args = ap.parse_args()
    with open(os.path.join(ROOT, "outputs/test_access_log.txt"), "a") as f:
        f.write(f"{__import__('datetime').datetime.now().isoformat()} | stage1_prompts | "
                f"datasets={args.datasets} (derives prompts from OKD test + TRUSTED images; "
                f"GT used only for detection-IoU reference)\n")
    device = args.device

    from m2_train_unet import UNet
    unet = UNet().to(device)
    ck = torch.load(os.path.join(ROOT, "outputs/checkpoints", f"unet_seed{M1_PRIOR_SEED}.pt"),
                    map_location=device, weights_only=False)
    unet.load_state_dict(ck["model"])
    unet.eval()

    from ultralytics import YOLO
    import glob
    yolo_ckpt = sorted(glob.glob(os.path.join(
        ROOT, "outputs/yolo_runs", f"d1_seed{D1_SEED}", "weights", "best.pt")))
    assert yolo_ckpt, "YOLO best.pt not found"
    yolo = YOLO(yolo_ckpt[0])
    yolo.to(device)

    if "okd" in args.datasets:
        cases = list(M.iter_okd_test())
        if args.limit:
            cases = cases[: args.limit]
        preds = m1_predict_all(unet, cases, device)
        p1, p2, p4, det = {}, {}, {}, {}
        for cid, img, gt, _sp, _meta in cases:
            b, pt = derive_p1p2(preds[cid])
            p1[cid], p2[cid] = b, pt
            pb, conf = yolo_predict(yolo, img, device)
            p4[cid] = pb
            det[cid] = (M.iou_box(pb, M.oracle_box(gt)), conf)
        save_prompts("okd", p1, p2, p4)
        with open(os.path.join(ROOT, "outputs/metrics/e3_detection_okd.csv"), "w") as f:
            f.write("case_id,iou_p4_vs_oracle,conf\n")
            for cid, (iou, cf) in det.items():
                f.write(f"{cid},{iou:.6f},{cf:.4f}\n")
        print("[okd] detection rows:", len(det))

    if "trusted" in args.datasets:
        p1, p2, p4, det = {}, {}, {}, {}
        n = 0
        limit_t = args.limit
        for cid, img, gt, _sp, meta in M.iter_trusted():
            pred = M.infer_unet(unet, img, device)
            b, pt = derive_p1p2(pred)
            p1[cid], p2[cid] = b, pt
            pb, conf = yolo_predict(yolo, img, device)
            p4[cid] = pb
            det[cid] = (M.iou_box(pb, M.oracle_box(gt)), conf)
            n += 1
            if limit_t and n >= limit_t:
                break
            if n % 2000 == 0:
                print(f"[trusted] {n} slices...", flush=True)
        save_prompts("trusted", p1, p2, p4)
        with open(os.path.join(ROOT, "outputs/metrics/e3_detection_trusted.csv"), "w") as f:
            f.write("case_id,iou_p4_vs_oracle,conf\n")
            for cid, (iou, cf) in det.items():
                f.write(f"{cid},{iou:.6f},{cf:.4f}\n")
        print("[trusted] detection rows:", len(det))


if __name__ == "__main__":
    main()
