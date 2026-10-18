#!/usr/bin/env python3
"""M5 P1-1/P1-2: dump ALL YOLO detections (boxes+confs, conf floor 0.01) for
every TRUSTED evaluation slice, one npz per seed, plus the oracle GT boxes.

The pre-registered P4 prompt used only the top-1 box at conf=0.25
(m3_stage1_prompts.yolo_predict). Re-running inference with a 0.01 floor and
keeping every candidate lets us compute, offline and exactly:
  - detection-rate / box-quality curves over any threshold (0.10/0.15/0.25),
  - AP@[0.5] against the STAPLE-derived oracle boxes,
without touching the GPU again. Inference settings mirror m3_stage1_prompts
(grayscale replicated to 3 channels, model default imgsz) so the top-1 box at
conf=0.25 reproduces the stored p4 prompts.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

# stream slices; never materialize the full set (18,322 slices ~ 30 GB RAM)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from m3_common import iter_trusted  # noqa: E402
from utils_kidney import box_from_mask  # noqa: E402

ROOT = Path(__import__("os").environ.get("KIDNEYUS_ROOT",
          str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "outputs/prompts"
SEEDS = [42, 43, 44]
CONF_FLOOR = 0.01

# oracle GT boxes are model-independent: dump once (streaming pass)
if (OUT / "p3_oracle_trusted.npz").exists():
    ids = list(np.load(OUT / "p3_oracle_trusted.npz")["case_ids"])
    print(f"slices: {len(ids)}; oracle npz exists, skip re-dump", flush=True)
    ids_only = True
else:
    ids_only = False
t0 = time.time()
gt_boxes, miss = [], 0
for cid, _img, gt, _sp, _meta in ([] if ids_only else iter_trusted()):
    ids.append(cid)
    b = box_from_mask(gt)
    if b is None:
        miss += 1
    gt_boxes.append([np.nan] * 4 if b is None else list(map(float, b)))
if not ids_only:
    np.savez_compressed(OUT / "p3_oracle_trusted.npz",
                        case_ids=np.array(ids), boxes=np.array(gt_boxes, dtype=np.float32))
    print(f"slices: {len(ids)}; oracle boxes saved ({miss} empty-GT) in {time.time()-t0:.0f}s", flush=True)

import os
for seed in SEEDS:
    if (OUT / f"yolo_alldet_trusted_seed{seed}.npz").exists():
        print(f"seed{seed}: npz exists, skip", flush=True)
        continue
    from ultralytics import YOLO
    w = ROOT / f"outputs/yolo_runs/d1_seed{seed}/weights/best.pt"
    model = YOLO(str(w))
    all_boxes, all_confs, all_idx = [], [], []
    t0 = time.time()
    for i, (cid, img, _gt, _sp, _meta) in enumerate(iter_trusted()):
        img_bgr = np.stack([img] * 3, axis=-1)
        r = model.predict(img_bgr, verbose=False, conf=CONF_FLOOR,
                          device="cuda" if torch.cuda.is_available() else "cpu")[0]
        if r.boxes is not None and len(r.boxes) > 0:
            bx = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            cf = r.boxes.conf.cpu().numpy().astype(np.float32)
            all_boxes.append(bx)
            all_confs.append(cf)
            all_idx.append(np.full(len(cf), i, dtype=np.int32))
        if (i + 1) % 4000 == 0:
            print(f"  seed{seed}: {i+1}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)
    boxes = np.concatenate(all_boxes) if all_boxes else np.zeros((0, 4), np.float32)
    confs = np.concatenate(all_confs) if all_confs else np.zeros((0,), np.float32)
    idx = np.concatenate(all_idx) if all_idx else np.zeros((0,), np.int32)
    np.savez_compressed(OUT / f"yolo_alldet_trusted_seed{seed}.npz",
                        case_ids=np.array(ids), row_index=idx, boxes=boxes, confs=confs)
    n_with = len({int(j) for j in idx})
    print(f"seed{seed}: {len(confs)} detections on {n_with}/{len(ids)} slices "
          f"-> yolo_alldet_trusted_seed{seed}.npz ({time.time()-t0:.0f}s)", flush=True)
print("DONE", flush=True)
