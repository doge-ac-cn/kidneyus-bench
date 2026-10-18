#!/usr/bin/env python3
"""M5 P1-2: P4 threshold sensitivity at the METRIC level.

Detection-level analysis (m5_yolo_detect_metrics) shows the conf distribution
has a cliff - lowering the operating threshold only changes the top-1 box on a
small slice set. This script makes the sensitivity claim exact at the metric
level: for tau in {0.10, 0.15} it re-runs inference ONLY on slices whose top-1
YOLO box (the pre-registered seed-43 detector behind the P4 prompts) differs
from the operating point at 0.25, and patches those rows into a copy of the
primary e2 results. Unchanged slices reuse their existing rows verbatim, so
the resulting table is exactly the full re-evaluation at no extra GPU cost.

Writes outputs/summary/p4_conf_sensitivity.json (case-free, committable).
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
from m3_common import iter_trusted, load_sam_family, load_samus, score, \
    infer_sam_family, infer_samus_box, MODEL_RES  # noqa: E402

ROOT = Path(__import__("os").environ.get("KIDNEYUS_ROOT",
          str(Path(__file__).resolve().parent.parent)))
TAUS = (0.10, 0.15)
MODELS = ("medsam", "sam_lora", "samus")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# the pre-registered P4 prompt source is the seed-43 detector for every model seed
DET = np.load(ROOT / "outputs/prompts/yolo_alldet_trusted_seed43.npz")
_by_slice = defaultdict(list)
for i, b, c in zip(DET["row_index"], DET["boxes"], DET["confs"]):
    _by_slice[int(i)].append((float(c), b.tolist()))


def top1_box(cs, bs, tau):
    sel = [j for j in range(len(cs)) if cs[j] >= tau]
    if not sel:
        return None
    j = max(sel, key=lambda k: cs[k])
    return bs[j]


def main():
    rows = defaultdict(lambda: defaultdict(dict))  # model -> seed -> case_id -> (dice, hd95)
    with open(ROOT / "outputs/metrics/e2_external.csv") as f:
        for r in csv.DictReader(f):
            if r["model"] in MODELS and r["prompt"] == "p4":
                h = r["hd95_mm"]
                rows[r["model"]][int(r["seed"])][r["case_id"]] = (
                    float(r["dice"]), float("nan") if h in ("", "nan") else float(h))
    case_ids = list(DET["case_ids"])
    idx_of = {c: i for i, c in enumerate(case_ids)}

    changed, newbox = {}, {}
    for tau in (0.25,) + TAUS:
        ch, nb = set(), {}
        for i in range(len(case_ids)):
            cands = _by_slice.get(i, [])
            b = top1_box([c for c, _ in cands], [x for _, x in cands], tau)
            nb[i] = b
        if tau == 0.25:
            base = nb
        else:
            changed[tau] = {case_ids[i] for i in nb if nb[i] != base[i]}
            newbox[tau] = nb
    for tau in TAUS:
        print(f"tau={tau}: {len(changed[tau])} slices change their top-1 box", flush=True)

    results = {}
    for model in MODELS:
        res = MODEL_RES[model]
        model_obj = load_samus(42, DEVICE) if model == "samus" else load_sam_family(model, 42, DEVICE)
        for tau in TAUS:
            todo = changed[tau]
            print(f"{model} tau={tau}: re-segmenting {len(todo)} slices", flush=True)
            newvals = {}
            for cid, img, gt, spacing, _meta in iter_trusted():
                if cid not in todo:
                    continue
                box = newbox[tau][idx_of[cid]]
                if box is None:
                    newvals[cid] = (0.0, float("nan"))
                    continue
                if model == "samus":
                    pred = infer_samus_box(model_obj, img, DEVICE, box=box)
                else:
                    pred = infer_sam_family(model_obj, img, res, DEVICE, box=box)
                d, h = score(pred.astype(bool), gt, spacing)
                newvals[cid] = (float(d), float(h))
            for seed in (42, 43, 44):
                merged = dict(rows[model][seed])
                merged.update(newvals)
                dices = np.array([merged[c][0] for c in merged])
                hds = np.array([merged[c][1] for c in merged if not np.isnan(merged[c][1])])
                results[f"{model}/tau{tau:.2f}/seed{seed}"] = {
                    "n": len(dices),
                    "dice_mean": round(float(dices.mean()), 4),
                    "hd95_mean_valid": round(float(hds.mean()), 2) if len(hds) else None,
                    "hd95_nan_n": int(sum(1 for c in merged if np.isnan(merged[c][1]))),
                }
        del model_obj
        torch.cuda.empty_cache()

    for model in MODELS:
        for seed in (42, 43, 44):
            dices = np.array([v[0] for v in rows[model][seed].values()])
            results[f"{model}/tau0.25/seed{seed}"] = {
                "n": len(dices), "dice_mean": round(float(dices.mean()), 4),
                "note": "unpatched primary rows (operating point)"}
    (ROOT / "outputs/summary/p4_conf_sensitivity.json").write_text(json.dumps(results, indent=1))
    print("saved -> outputs/summary/p4_conf_sensitivity.json", flush=True)


if __name__ == "__main__":
    main()
