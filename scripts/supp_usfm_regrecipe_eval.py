#!/usr/bin/env python3
"""Supplementary item 9 (task 1): evaluate the USFM registered-recipe
checkpoint (lr=1e-5, 50 epochs, Dice+BCE, seed 42) on E1/E2, prompt-free.

Mirrors the registered evaluation pathway (m3_common.infer_usfm ->
m6_usfm.preprocess_image/infer_logits); checkpoint built with
`m6_usfm.py --tag _regrecipe`. Outputs supp_usfm_regrecipe.json.
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import m6_usfm  # noqa: E402
from m3_common import iter_okd_test, iter_trusted, infer_usfm  # noqa: E402
from utils_kidney import dice_score  # noqa: E402

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
SEED = int(os.environ.get("USFM_SEED", "42"))
CKPT = ROOT / f"outputs/checkpoints/usfm_regrecipe_seed{SEED}.pt"
LOG = ROOT / f"outputs/logs/usfm_regrecipe_seed{SEED}.log"
OUT = ROOT / f"outputs/summary/supp_usfm_regrecipe_seed{SEED}.json"


def best_val():
    best = None
    if LOG.exists():
        for line in open(LOG, errors="ignore"):
            m = re.search(r"val_dice=([0-9.]+)", line)
            if m:
                v = float(m.group(1))
                best = v if best is None else max(best, v)
    return best


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = m6_usfm.build_usfm(device)
    ck = torch.load(str(CKPT), map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    e1 = []
    for _cid, img, gt, _sp, _m in iter_okd_test():
        pred = infer_usfm(model, img, device)
        e1.append(dice_score(pred, gt))
    print(f"E1 done ({len(e1)})", flush=True)
    e2 = []
    for i, (_cid, img, gt, _sp, _m) in enumerate(iter_trusted()):
        pred = infer_usfm(model, img, device)
        e2.append(dice_score(pred, gt))
        if (i + 1) % 4000 == 0:
            print(f"E2 {i+1}", flush=True)
    # official-recipe reference for the SAME seed, read from the registered CSVs
    import pandas as pd
    e1df = pd.read_csv(ROOT / "outputs/metrics/e1_internal.csv")
    e2df = pd.read_csv(ROOT / "outputs/metrics/e2_external.csv")
    import re as _re
    _m = _re.search(r"seed(\d+)", str(CKPT))
    _seed = int(_m.group(1)) if _m else 42
    official = {
        f"e1_seed{_seed}": round(float(e1df[(e1df.model == "usfm") & (e1df.prompt == "none")
                                            & (e1df.seed == _seed)]["dice"].mean()), 4),
        f"e2_seed{_seed}": round(float(e2df[(e2df.model == "usfm") & (e2df.prompt == "none")
                                            & (e2df.seed == _seed)]["dice"].mean()), 4)}
    res = {f"arm": f"USFM registered recipe (lr=1e-5, 50 epochs, Dice+BCE, seed {SEED})",
           "recipe_source": "plan_A v1.4 §A1 (frozen registration basis): "
                            "'微调协议对齐 M3 (lr=1e-5、50ep)；验收门 val Dice ≥0.85'",
           "training_details": {"epochs": 50, "base_lr": 1e-5, "layer_decay": 1.0,
                                "warmup_epochs": 0, "weight_decay": 0.01, "bs": 8,
                                "loss": "Dice+BCE (probability domain, same as official run)"},
           "internal_val_best": best_val(),
           "acceptance_gate_0.85": (best_val() or 0) >= 0.85,
           "e1_test_dice": round(float(np.mean(e1)), 4),
           "e2_slice_pooled_dice": round(float(np.mean(e2)), 4),
           "official_seed42": official,
           "delta_vs_official": {
               "e1": round(float(np.mean(e1)) - official[f"e1_seed{_seed}"], 4),
               "e2": round(float(np.mean(e2)) - official[f"e2_seed{_seed}"], 4)},
           "n_e1": len(e1), "n_e2": len(e2)}
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
