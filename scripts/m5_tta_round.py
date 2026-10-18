#!/usr/bin/env python3
"""Amendment 1 - Arm P4-TTA round runner (LOCKED until OSF timestamp).

Spec (docs/tta_amendment_v1.md sections 4.1-4.5):
- init: registered d1_seed43 YOLOv8s weights (round 1) / round-1 output (round 2)
- pseudo-labels: detections of the CURRENT detector on all 18,322 external
  slices; top-1 box per slice if conf >= 0.50; slices without detection are
  EXCLUDED from training (neither positive nor negative)
- training: 15 epochs, lr0=1e-3, all other ultralytics 8.3.108 defaults,
  plus domain ops (m5_tta_aug.domain_augment, p=0.8) applied to the raw
  image before the standard pipeline; data-order seed per replicate
- checkpoints archived to outputs/yolo_runs/p4tta_r{round}_s{seed}/
- eval: registered detection metrics + downstream P4 pathway re-scoring
  against the registered scorer (TRUSTED GT touches evaluation only)

Usage (AFTER OSF lock):
  python scripts/m5_tta_round.py --round 1 --seed 42
  python scripts/m5_tta_round.py --round 1 --seed 42 --dryrun  # dataset build only

OPEN INTERPRETATION (flagged to the writing team pre-submission, see
RUN_REPORT): §4.3 "Data-order seeds fixed (42/43/44)" is implemented as
three replicates per round, each fine-tuned from the same registered
weights with a different data-order seed; §4.4 (ii)/(iii) are evaluated
for each of the three prompted models separately.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# GPU selection must happen before torch import; the shell-inherited
# CUDA_VISIBLE_DEVICES has proven unreliable in this environment
# (R2 launch 2026-10-12 00:12 arrived with CVD=0 despite exported =1).
if os.environ.get("KIDNEYUS_GPU"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["KIDNEYUS_GPU"]

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__import__("os").environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
CONF_PSEUDO = 0.50
EPOCHS = 15
LR0 = 1e-3


def build_pseudo_dataset(weights: Path, out_dir: Path, seed: int):
    """Detect with the current detector on all external slices (IMAGES ONLY),
    keep top-1 box per slice at conf>=0.50, write a YOLO-format dataset of
    detected slices only."""
    import cv2  # ultralytics dependency, available in kidneyus2
    from m3_common import iter_trusted
    from ultralytics import YOLO

    model = YOLO(str(weights))
    img_dir, lbl_dir = out_dir / "images", out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_det, n_total = 0, 0
    for cid, img, _gt, _sp, _meta in iter_trusted():
        n_total += 1
        r = model.predict(np.stack([img] * 3, -1), verbose=False, conf=0.01,
                          device="cuda")[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        confs = r.boxes.conf.cpu().numpy()
        j = int(np.argmax(confs))
        if float(confs[j]) < CONF_PSEUDO:
            continue
        h, w = img.shape
        x0, y0, x1, y1 = r.boxes.xyxy.cpu().numpy()[j]
        xc, yc, bw, bh = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h
        stem = f"p4tta_{cid}"
        cv2.imwrite(str(img_dir / f"{stem}.png"), img)
        (lbl_dir / f"{stem}.txt").write_text(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        n_det += 1
        if n_det % 2000 == 0:
            print(f"  pseudo-labels: {n_det} slices", flush=True)
    print(f"pseudo dataset: {n_det}/{n_total} slices kept (conf>={CONF_PSEUDO})")
    return n_det


def wait_for_gpu(min_gb=8.0, tries=20, gap=60):
    """nnU-Net co-tenant validation spikes (~7 GB for ~1 min every 50 epochs)
    transiently shrink free memory; wait for a calm window instead of dying."""
    import torch, time
    for k in range(tries):
        free, total = torch.cuda.mem_get_info(0)
        free_gb = free / 2**30
        print(f"device guard[{k}]: cuda:0 free={free_gb:.1f}GiB "
              f"(CVD={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')!r})", flush=True)
        if free_gb >= min_gb:
            return
        time.sleep(gap)
    raise RuntimeError(f"device guard: free memory stayed below {min_gb} GiB "
                       f"for {tries} checks - giving up")


def train_round(data_yaml: Path, weights: Path, out_dir: Path, seed: int):
    wait_for_gpu()

    from ultralytics import YOLO
    import m5_tta_aug

    model = YOLO(str(weights))
    # domain ops are applied to the loaded image before the standard pipeline
    # by patching the dataset loader used by ultralytics' trainer
    _orig_load = __import__("ultralytics.data.dataset", fromlist=["YOLODataset"]) \
        .YOLODataset.load_image

    rng = np.random.default_rng(seed)

    def patched_load_image(self, i, rect_mode=True):
        im, f, fn = _orig_load(self, i, rect_mode)  # 8.3.108 returns a triple
        if im is not None and getattr(self, "augment", False):
            arr = im[..., 0] if im.ndim == 3 else im
            arr = m5_tta_aug.domain_augment(arr, rng, p=0.8)
            im = np.stack([arr] * 3, -1)
        return im, f, fn

    import ultralytics.data.dataset as _ds
    _ds.YOLODataset.load_image = patched_load_image
    try:
        model.train(data=str(data_yaml), epochs=EPOCHS, lr0=LR0, seed=seed,
                    project=str(out_dir.parent), name=out_dir.name,
                    exist_ok=True, cache=False)
    finally:
        _ds.YOLODataset.load_image = _orig_load


def evaluate(weights: Path, round_tag: str, seed: int):
    """Detection metrics on TRUSTED (no-box rate etc.) + OKD in-domain check;
    downstream P4 pathway re-scoring is run by m5_p4_conf_sensitivity-style
    scripts and aggregated in RUN_REPORT."""
    import subprocess
    out = ROOT / f"outputs/prompts/yolo_alldet_trusted_tta_r{round_tag}_s{seed}.npz"
    # reuse the registered all-detection dump machinery against new weights
    code = f"""
import sys; sys.path.insert(0, "{ROOT / 'scripts'}")
from m5_yolo_alldet_trusted import *  # noqa
from ultralytics import YOLO
import numpy as np
from m3_common import iter_trusted
model = YOLO("{weights}")
ids, all_b, all_c, all_i = [], [], [], []
for i, (cid, img, _g, _s, _m) in enumerate(iter_trusted()):
    ids.append(cid)
    r = model.predict(np.stack([img]*3, -1), verbose=False, conf=0.01, device="cuda")[0]
    if r.boxes is not None and len(r.boxes):
        all_b.append(r.boxes.xyxy.cpu().numpy().astype(np.float32))
        all_c.append(r.boxes.conf.cpu().numpy().astype(np.float32))
        all_i.append(np.full(len(r.boxes), i, dtype=np.int32))
np.savez_compressed("{out}", case_ids=np.array(ids),
                    row_index=np.concatenate(all_i) if all_i else np.zeros(0, np.int32),
                    boxes=np.concatenate(all_b) if all_b else np.zeros((0,4), np.float32),
                    confs=np.concatenate(all_c) if all_c else np.zeros(0, np.float32))
print("dumped", "{out}")
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True, choices=[1, 2])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dryrun", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse the existing pseudo dataset (identical within a round)")
    ap.add_argument("--data-dir", default=None,
                    help="shared pseudo-dataset dir for all replicates of a round")
    a = ap.parse_args()
    tag = f"r{a.round}_s{a.seed}"
    work = ROOT / "outputs/p4tta" / tag
    work.mkdir(parents=True, exist_ok=True)
    if a.round == 1:
        weights = ROOT / "outputs/yolo_runs/d1_seed43/weights/best.pt"
    else:
        # paired design: replicate s continues from its OWN round-1 checkpoint
        weights = ROOT / f"outputs/p4tta/r1_s{a.seed}/train/weights/best.pt"
        assert weights.exists(), f"round-1 checkpoint missing for seed {a.seed}"

    ds_dir = Path(a.data_dir) if getattr(a, "data_dir", None) else work / "dataset"
    if a.reuse and (ds_dir / "data.yaml").exists():
        n_det = len(list((ds_dir / "images").glob("*.png")))
        print(f"reusing pseudo dataset: {n_det} slices")
    else:
        n_det = build_pseudo_dataset(weights, ds_dir, a.seed)
        (ds_dir / "data.yaml").write_text(
            f"path: {ds_dir}\ntrain: images\nval: images\nnames:\n  0: kidney\n")
    data_yaml = ds_dir / "data.yaml"
    if a.dryrun:
        print(f"dryrun: dataset built ({n_det} slices), training NOT started")
        return
    train_round(data_yaml, weights, work / "train", a.seed)
    best = work / "train" / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"training produced no best.pt (see {work / 'train'})")
    shutil.copy(best, work / "weights_best_archived.pt")
    evaluate(best, a.round, a.seed)
    print(f"round {a.round} seed {a.seed} complete; checkpoint archived")


if __name__ == "__main__":
    main()
