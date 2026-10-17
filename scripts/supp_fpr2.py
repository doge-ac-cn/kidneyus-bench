#!/usr/bin/env python3
"""Item ⑦ FPR (parallel): discontinuity-detector false-positive probes.

Per kidney (multiprocessing across 8 workers; each rater volume read ONCE):
  (a) flags at 3x MAD on the real order  -> STAPLE / A / B sets
  (b) flags at 2x and 4x MAD             -> threshold sensitivity
  (c) flags on within-kidney shuffled order (20 shuffles) -> null flag rate
"""
import json
import os
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
MASKS = ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA/US_masks"


def flags_from_series(shifts, mult):
    if len(shifts) < 3:
        return set()
    ss = np.array([s for _, s in shifts])
    med = np.median(ss)
    mad = np.median(np.abs(ss - med)) + 1e-12
    thr = med + mult * mad
    sm = dict(shifts)
    out = set()
    for z, s in shifts:
        if s >= thr and s > 0:
            l = sm.get(z - 1, -1)
            r = sm.get(z + 1, -1)
            if s >= l and s >= r:
                out.add(int(z))
    return out


def work(args):
    kid, zs = args
    out = {"kid": kid}
    rng = np.random.default_rng(abs(hash(kid)) % 2**32)
    for rater_dir, suffix, key in (("GT_estimated_masksUS", "_maskUS", "staple"),
                                   ("Annotator1", "1_maskUS", "A"),
                                   ("Annotator2", "2_maskUS", "B")):
        m = sitk.GetArrayFromImage(sitk.ReadImage(
            str(MASKS / rater_dir / f"{kid}{suffix}.nii.gz"))) > 0.5
        shifts = []
        for a, b in zip(zs[:-1], zs[1:]):
            if b - a != 1:
                continue
            u = (m[a] | m[b]).sum()
            shifts.append((a, 1.0 - ((m[a] & m[b]).sum() / u if u else 1.0)))
        out[f"{key}_f3"] = sorted(flags_from_series(shifts, 3))
        if key == "staple":
            out["f2"] = len(flags_from_series(shifts, 2))
            out["f4"] = len(flags_from_series(shifts, 4))
            out["pairs"] = len(shifts)
            null = 0
            seq = [z for z, _ in shifts]
            for _ in range(20):
                perm = list(rng.permutation(seq))
                sh = []
                for a, b in zip(perm[:-1], perm[1:]):
                    u = (m[a] | m[b]).sum()
                    sh.append((a, 1.0 - ((m[a] & m[b]).sum() / u if u else 1.0)))
                null += len(flags_from_series(sh, 3))
            out["null_flags_20x"] = null
        del m
    return out


def main():
    sl = pd.read_csv(ROOT / "outputs/trusted_us_slices.csv")
    jobs = [(kid, sorted(int(z) for z in g["slice_idx"]))
            for kid, g in sl.groupby("kidney_id", sort=True)]
    with Pool(8) as pool:
        res = pool.map(work, jobs, chunksize=2)
    staple = {(r["kid"], z) for r in res for z in r["staple_f3"]}
    A = {(r["kid"], z) for r in res for z in r["A_f3"]}
    B = {(r["kid"], z) for r in res for z in r["B_f3"]}
    pairs = sum(r["pairs"] for r in res)
    null_flags = sum(r["null_flags_20x"] for r in res)
    out = {"definition": "adjacent-slice 1-IoU series; flags = local maxima >= median + k*MAD",
           "n_staple_flags_k3": len(staple), "n_adjacent_pairs": pairs,
           "flag_rate_k3": round(len(staple) / pairs, 4),
           "threshold_sensitivity_flags": {"k2": sum(r["f2"] for r in res),
                                           "k3": len(staple),
                                           "k4": sum(r["f4"] for r in res)},
           "null_shuffle_flag_rate": round(null_flags / (20 * pairs), 4),
           "rater_agreement_jaccard": {
               "staple_vs_A": round(len(staple & A) / max(len(staple | A), 1), 3),
               "staple_vs_B": round(len(staple & B) / max(len(staple | B), 1), 3),
               "A_vs_B": round(len(A & B) / max(len(A | B), 1), 3)},
           "note": "shuffled-order flag rate is the chance-level noise floor of the criterion"}
    (ROOT / "outputs/summary/supp_detector_fpr.json").write_text(json.dumps(out, indent=1))
    # also save the flagged sets for tolerant-overlap analysis
    (ROOT / "outputs/summary/supp_detector_fpr_sets.json").write_text(json.dumps(
        {"staple": sorted(staple), "A": sorted(A), "B": sorted(B)}, indent=0))
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
