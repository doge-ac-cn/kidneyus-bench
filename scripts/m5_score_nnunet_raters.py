#!/usr/bin/env python3
"""M5 P3-8: score nnU-Net external predictions against Annotator1/Annotator2.

The E2 primary protocol scores against the STAPLE consensus (>0.5). Reviewers
may object that a consensus target is optimistically biased; this re-scores
the SAME prediction set (nnU-Net best-checkpoint, all 3 seeds) against each
individual rater, binarized at the same >0.5 threshold as the pipeline's GT
(rater maps are float with interpolation noise, same as STAPLE).

Per-case rows -> outputs/metrics/e2_external_rater{A,B}.csv (gitignored, carry
dataset-internal kidney ids). Aggregate (case-free) ->
outputs/summary/nnunet_rater_aggregate.json alongside the STAPLE numbers for
the identical slice set.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).parent))
from m3_common import iter_trusted, score  # noqa: E402

ROOT = Path(__import__("os").environ.get("KIDNEYUS_ROOT",
          str(Path(__file__).resolve().parent.parent)))
RAW = ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA/US_masks"
SEEDS = [42, 43, 44]
PRED = "outputs/predictions/nnunet_seed{seed}_best/trusted"

# cache volumes across their slices: iter_trusted groups by volume already,
# but simplest correct approach: load lazily per case and memoize last volume.
_vol_cache = {}


def rater_slice(kid, z, rater_dir):
    key = (kid, rater_dir)
    if key not in _vol_cache:
        # keep at most 2 volumes in memory
        if len(_vol_cache) > 2:
            _vol_cache.pop(next(iter(_vol_cache)))
        path = RAW / rater_dir / f"{kid}{'1' if rater_dir == 'Annotator1' else '2'}_maskUS.nii.gz"
        _vol_cache[key] = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return _vol_cache[key][z] > 0.5


def main():
    files = {r: open(ROOT / f"outputs/metrics/e2_external_rater{r}.csv", "w", newline="")
             for r in ("A", "B")}
    writers = {r: csv.writer(f) for r, f in files.items()}
    for w in writers.values():
        w.writerow(["model", "seed", "prompt", "case_id", "dice", "hd95", "is_key"])
    # optional key-slice flags from the P3-7 detector (stratified reporting)
    key_map = {}
    ks = ROOT / "outputs/metrics/key_slices_trusted.csv"
    if ks.exists():
        for r in csv.DictReader(open(ks)):
            if r["is_key"] == "1":
                key_map[f"{r['kidney_id']}_z{int(r['slice_idx']):03d}"] = 1
    agg = defaultdict(lambda: defaultdict(list))
    n = 0
    for seed in SEEDS:
        pred_dir = ROOT / PRED.format(seed=seed)
        for cid, _img, _gt, spacing, meta in iter_trusted():
            kid, z = meta["kidney_id"], meta["slice_idx"]
            p = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_dir / f"{cid}.nii.gz"))) > 0
            p = np.squeeze(p)
            ik = key_map.get(cid, 0)
            for rater, tag in (("Annotator1", "A"), ("Annotator2", "B")):
                gt = rater_slice(kid, z, rater)
                d, h = score(p, gt, spacing)
                writers[tag].writerow(["nnunet", seed, "none", cid,
                                       f"{d:.6f}", "" if np.isnan(h) else f"{h:.4f}", ik])
                agg[(seed, tag)]["dice"].append(d)
                agg[(seed, tag)]["hd"].append(h)
                agg[(seed, tag + ("_key" if ik else "_interp"))]["dice"].append(d)
            n += 1
            if n % 5000 == 0:
                print(f"  {n} slices scored", flush=True)
    for f in files.values():
        f.close()

    # STAPLE reference for the same slice set, from the primary CSV
    staple = defaultdict(list)
    with open(ROOT / "outputs/metrics/e2_external.csv") as f:
        for r in csv.DictReader(f):
            if r["model"] == "nnunet" and r["prompt"] == "none":
                staple[int(r["seed"])].append(float(r["dice"]))
    out = {}
    import statistics as st
    for (seed, tag), v in sorted(agg.items()):
        d = v["dice"]
        hd = [x for x in v["hd"] if not np.isnan(x)]
        e = {"n": len(d), "dice_mean": round(st.mean(d), 4),
             "dice_std": round(st.stdev(d), 4), "dice_median": round(st.median(d), 4),
             "hd95_mean_valid": round(float(np.mean(hd)), 2) if hd else None,
             "hd95_nan_n": int(np.isnan(v["hd"]).sum())}
        if tag.endswith(("_key", "_interp")):  # stratified: dice only
            e = {"n": len(d), "dice_mean": round(st.mean(d), 4),
                 "dice_std": round(st.stdev(d), 4)}
        out[f"seed{seed}_rater{tag}"] = e
    for seed, d in sorted(staple.items()):
        out[f"seed{seed}_staple"] = {"n": len(d), "dice_mean": round(st.mean(d), 4),
                                     "dice_std": round(st.stdev(d), 4)}
    (ROOT / "outputs/summary/nnunet_rater_aggregate.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
