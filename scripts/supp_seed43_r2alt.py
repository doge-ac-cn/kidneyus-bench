#!/usr/bin/env python3
"""R2-alt sensitivity summary under the registered seed 43 (MedSAM seed-43
downstream weights on the three single-start R2 detector replicates)."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
MET = ROOT / "outputs/metrics"
OUT = ROOT / "outputs/summary/supp_r2alt.json"

out = {"arm": "R2-alt: single-start from the REGISTERED detector (three replicates, "
              "data-order seeds 42/43/44); downstream = MedSAM seed-43 weights",
       "caliber": "registered seed 43", "per_replicate": {}}
for seed in (42, 43, 44):
    df = pd.read_csv(MET / f"rev43_r2alt_s{seed}_medsam.csv")
    assert len(df) == 18322, (seed, len(df))
    det = df[df["pred_area_px"] > 0]
    out["per_replicate"][f"s{seed}"] = {
        "no_box_rate": round(float((df["pred_area_px"] == 0).mean()), 4),
        "strict": round(float(df["dice"].mean()), 4),
        "conditional": round(float(det["dice"].mean()), 4),
        "coverage": round(float(len(det) / len(df)), 4)}
keys = ("no_box_rate", "strict", "conditional", "coverage")
out["round_mean"] = {k: round(float(np.mean([out["per_replicate"][f"s{s}"][k]
                                             for s in (42, 43, 44)])), 4) for k in keys}
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
