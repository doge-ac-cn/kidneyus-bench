#!/usr/bin/env python3
"""B2/B3b/B4b GPU mega-pass v2 (parallel, crash-safe).

v2 changes (after the 4-agent audit):
- oracle boxes come from the precomputed p3_oracle_trusted.npz (fixes the
  GT-path typo AND the 32-h per-slice GT re-read);
- per-pass CSV written atomically (.part + os.replace) with a 18,323-line
  completion check, so interrupted passes are re-run, never silently skipped;
- prefetch thread hides CPU-side volume loading behind GPU inference;
- pass-level CLI: one worker process per GPU, passes distributed statically.

Pass inventory (7): medsam42 x {p1, p5, p4reg, tta_r2s43, oracle},
sam_lora42 x tta_r2s43, samus42 x tta_r2s43.

Usage (3 GPUs): launch three workers --
  python m5_bpass_gpu_v2.py --gpu 0 --passes p1,p4reg,oracle
  python m5_bpass_gpu_v2.py --gpu 1 --passes p5,tta_r2s43
  python m5_bpass_gpu_v2.py --gpu 2 --passes samlora_tta,samus_tta
"""
import argparse
import json
import os
import queue
import sys
import threading
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from m3_common import iter_trusted, load_sam_family, load_samus, score, \
    infer_sam_family, infer_samus_box, MODEL_RES, oracle_box  # noqa: E402

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "outputs/metrics"
N_ROWS = 18323  # header + 18,322 slices

PASSES = {
    "medsam_p1": ("medsam", 42, "p1"),
    "medsam_p5": ("medsam", 42, "p5"),
    "medsam_p4reg": ("medsam", 42, "p4reg"),
    "medsam_tta_r2s43": ("medsam", 42, "tta_r2s43"),
    "medsam_oracle": ("medsam", 42, "oracle"),
    "samlora_tta_r2s43": ("sam_lora", 42, "tta_r2s43"),
    "samus_tta_r2s43": ("samus", 42, "tta_r2s43"),
}

_boxes_cache = {}


def boxes_for(source):
    if source in _boxes_cache:
        return _boxes_cache[source]
    if source == "oracle":
        z = np.load(ROOT / "outputs/prompts/p3_oracle_trusted.npz")
        table = {str(c): (None if np.isnan(np.asarray(b, dtype=float)).any()
                          else np.asarray(b, dtype=float).tolist())
                 for c, b in zip(z["case_ids"], z["boxes"])}
    elif source in ("p1", "p4reg", "p5"):
        fname = {"p1": "p1_box_trusted.npz", "p4reg": "p4_box_trusted.npz",
                 "p5": "p5_box_trusted.npz"}[source]
        z = np.load(ROOT / f"outputs/prompts/{fname}", allow_pickle=True)
        if "case_ids" in z.files:   # flat layout (p5)
            table = {str(c): (None if np.isnan(np.asarray(b, dtype=float)).any()
                              else np.asarray(b, dtype=float).tolist())
                     for c, b in zip(z["case_ids"], z["boxes"])}
        else:                        # per-case keys layout (p1, p4reg)
            table = {}
            for k in z.files:
                arr = np.asarray(z[k], dtype=float)
                table[k] = None if (arr.size != 4 or np.isnan(arr).any()) else arr.tolist()
    elif source == "tta_r2s43":
        z = np.load(ROOT / "outputs/prompts/yolo_alldet_trusted_tta_r2_s43.npz")
        per = {}
        for i, b, c in zip(z["row_index"], z["boxes"], z["confs"]):
            per.setdefault(int(i), []).append((float(c), b))
        ids = [str(c) for c in z["case_ids"]]
        table = {}
        for i, cid in enumerate(ids):
            cands = [(c, b) for c, b in per.get(i, []) if c >= 0.25]
            table[cid] = max(cands, key=lambda x: x[0])[1].tolist() if cands else None
    else:
        raise KeyError(source)
    _boxes_cache[source] = table
    return table


def pass_complete(tag):
    p = OUT / f"bpass_{tag}.csv"
    return p.exists() and sum(1 for _ in open(p, errors="ignore")) == N_ROWS + 1


def run_pass(tag, model_name, seed, box_source, gpu):
    if pass_complete(tag):
        print(f"[skip] {tag} (complete)", flush=True)
        return
    part = OUT / f"bpass_{tag}.csv.part"
    import csv
    fh = open(part, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["case_id", "kidney_id", "patient", "dice", "pred_area_px"])

    model_name_key = model_name
    if model_name_key == "samus":
        model = load_samus(s, "cuda")
    else:
        model = load_sam_family(model_name_key, seed, "cuda")
    res = MODEL_RES[model_name_key]
    boxes = boxes_for(box_source)

    # prefetch: the iter_trusted generator's volume loads run in a background
    # thread while the main thread infers on the GPU
    q = queue.Queue(maxsize=6)

    def producer():
        for item in iter_trusted():
            q.put(item)
        q.put(None)

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    n = 0
    while True:
        item = q.get()
        if item is None:
            break
        cid, img, gt, spacing, meta = item
        box = boxes.get(cid)
        if box is None:
            d, area = 0.0, 0
        else:
            if model_name_key == "samus":
                pred = infer_samus_box(model, img, "cuda", box=box)
            else:
                pred = infer_sam_family(model, img, res, "cuda", box=box)
            pb = pred.astype(bool)
            d, _h = score(pb, gt, spacing)
            area = int(pb.sum())
        w.writerow([cid, meta["kidney_id"], meta["kidney_id"][:3], f"{d:.6f}", area])
        n += 1
        if n % 4000 == 0:
            print(f"  {tag}: {n} slices", flush=True)
    fh.close()
    final = OUT / f"bpass_{tag}.csv"
    os.replace(part, final)
    if sum(1 for _ in open(final, errors="ignore")) != N_ROWS + 1:
        raise RuntimeError(f"{tag}: row-count mismatch after pass")
    print(f"[done] {tag} ({n} slices)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--passes", required=True, help="comma-separated pass keys")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    for tag in a.passes.split(","):
        model_name, seed, source = PASSES[tag]
        run_pass(tag, model_name, seed, source, a.gpu)
    print(f"WORKER DONE gpu={a.gpu} passes={a.passes}", flush=True)


if __name__ == "__main__":
    main()
