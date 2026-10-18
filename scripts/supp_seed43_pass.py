#!/usr/bin/env python3
"""Seed-43 primary recomputation (registered seed): GPU passes.

Registers additional PASSES entries (model seed 43) inside supp_megapass and
runs the requested ones, writing rev43_{tag}.csv in the same schema
(dice vs STAPLE / A / B, HD95, NSD@1mm, pred_area).

P5 with the seed-43 nnU-Net box source is supported via a precomputed box npz
(outputs/prompts/p5_box_trusted_s43.npz), injected into the box cache.

Usage:
  python supp_seed43_pass.py --gpu 0 --passes medsam43_p3,medsam43_p4reg
  python supp_seed43_pass.py --gpu 1 --passes medsam43_p1,medsam43_tta,medsam43_p5s43
  python supp_seed43_pass.py --gpu 2 --passes samlora43_tta,samus43_tta
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location("mp", Path(__file__).resolve().parent / "supp_megapass.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

# tag -> (model, seed, box source)
S43 = {
    "medsam43_p3": ("medsam", 43, "p3"),
    "medsam43_p4reg": ("medsam", 43, "p4reg"),
    "medsam43_p1": ("medsam", 43, "p1"),
    "medsam43_tta": ("medsam", 43, "tta_r2s43"),
    "medsam43_p5s43": ("medsam", 43, "p5s43"),
    "samlora43_tta": ("sam_lora", 43, "tta_r2s43"),
    "samus43b_tta": ("samus", 43, "tta_r2s43"),  # distinct from the earlier samus43_tta_r2s43 tag
    # other detection replicates (for the round-level means)
    "medsam43_tta42": ("medsam", 43, "tta_r2s42"),
    "medsam43_tta44": ("medsam", 43, "tta_r2s44"),
    "samlora43_tta42": ("sam_lora", 43, "tta_r2s42"),
    "samlora43_tta44": ("sam_lora", 43, "tta_r2s44"),
    "samus43b_tta42": ("samus", 43, "tta_r2s42"),
    "samus43b_tta44": ("samus", 43, "tta_r2s44"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--passes", required=True)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    # P1 boxes (U-Net prior) injected: megapass.boxes_for has no p1 branch
    p1 = ROOT / "outputs/prompts/p1_box_trusted.npz"
    if p1.exists():
        z = np.load(p1, allow_pickle=True)
        table = {}
        for k in z.files:
            arr = np.asarray(z[k], dtype=float)
            table[k] = None if (arr.size != 4 or np.isnan(arr).any()) else arr.tolist()
        M._boxes_cache["p1"] = table
    # TTA detection dumps for the other replicates (r2_s42 / r2_s44)
    def tta_table(path):
        z = np.load(path)
        per = {}
        for i, b, c in zip(z["row_index"], z["boxes"], z["confs"]):
            per.setdefault(int(i), []).append((float(c), b))
        ids = [str(c) for c in z["case_ids"]]
        out = {}
        for i, cid in enumerate(ids):
            cands = [(c, b) for c, b in per.get(i, []) if c >= 0.25]
            out[cid] = max(cands, key=lambda x: x[0])[1].tolist() if cands else None
        return out
    for rep in (42, 44):
        f = ROOT / f"outputs/prompts/yolo_alldet_trusted_tta_r2_s{rep}.npz"
        if f.exists():
            M._boxes_cache[f"tta_r2s{rep}"] = tta_table(f)
    # P5 seed-43 boxes (precomputed) injected into the cache
    p5 = ROOT / "outputs/prompts/p5_box_trusted_s43.npz"
    if p5.exists():
        z = np.load(p5)
        M._boxes_cache["p5s43"] = {
            str(c): (None if np.isnan(np.asarray(b, dtype=float)).any()
                     else np.asarray(b, dtype=float).tolist())
            for c, b in zip(z["case_ids"], z["boxes"])}
    for tag in a.passes.split(","):
        model, seed, source = S43[tag]
        M.PASSES[tag] = (model, seed, source)
        M.run_pass(tag, a.gpu)
        # rename output to the seed-43 namespace
        src = ROOT / "outputs/metrics" / f"rev_{tag}.csv"
        dst = ROOT / "outputs/metrics" / f"rev43_{tag}.csv"
        if src.exists():
            os.replace(src, dst)
            print(f"[seed43] {tag} -> {dst.name}", flush=True)
    print(f"SEED43 WORKER DONE gpu={a.gpu}", flush=True)


if __name__ == "__main__":
    main()
