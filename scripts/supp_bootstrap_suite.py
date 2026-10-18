#!/usr/bin/env python3
"""Item ③: paired bootstrap CI suite (patient-39 clusters, 2000 iters,
duplicates preserved; kidney-59 variant where noted).

A. IoD CIs: A-vs-B direct Dice, slice caliber 0.9063 / kidney caliber 0.9047,
   patient-39 and kidney-59 cluster bootstrap.
B. R0 (registered MedSAM P4, seed42) vs R2 (TTA-R2 repair arm) paired strict
   and conditional differences:
     strict      = all common slices, means per patient, paired diff
     conditional = restricted to the FIXED subset = R0 detection set
                   (the registered P4-detected slices; both arms evaluated on
                   the same slices; R2 misses counted as the registered
                   empty-prompt rule, coverage reported).
C. Reverse-CV ceiling minus forward, paired by patient:
     2D slice: rev2 (nnU-Net trained on TRUSTED, 5-fold) vs e2 nnU-Net
     3D volume: rev3 (per-kidney 3-fold mean of reverse CV) vs forward
                stacked-volumes 3-seed per-kidney mean.
D. Domain-shift magnitudes: |degradation| with CI; CT volume-vs-US-dice
   correlation r (patient-cluster bootstrap) + quartile table lookup.

Outputs outputs/summary/supp_bootstrap_cis.json.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "outputs/summary"
N_BOOT = 2000


def cluster_boot_stat(clusters, fn, n=N_BOOT, seed=42):
    """Generic cluster bootstrap: fn(pick) -> scalar."""
    rng = np.random.default_rng(seed)
    cl = np.asarray(clusters)
    stats = np.empty(n)
    for it in range(n):
        stats[it] = fn(rng.choice(cl, len(cl), replace=True))
    return {"point": None, "ci95": [round(float(np.percentile(stats, 2.5)), 4),
                                    round(float(np.percentile(stats, 97.5)), 4)]}


def main():
    iod = pd.read_csv(ROOT / "outputs/metrics/iod_ab_per_slice.csv")
    out = {"n_boot": N_BOOT, "bootstrap": "cluster resample with replacement, duplicates preserved"}

    # ---- A. IoD CIs ----
    iod["kidney_id"] = iod["kidney_id"].astype(str)
    slice_mean = float(iod["dice_ab"].mean())
    kid_mean = float(iod.groupby("kidney_id")["dice_ab"].mean().mean())
    pat = iod.groupby("patient")["dice_ab"].mean()
    cp = cluster_boot_stat(pat.index, lambda pick: pat.loc[pick].mean())
    cp["point"] = round(slice_mean, 4)
    cp["patient_mean"] = round(float(pat.mean()), 4)
    cp["kidney_caliber_point"] = round(kid_mean, 4)
    kp = iod.groupby("kidney_id")["dice_ab"].mean()
    ck = cluster_boot_stat(kp.index, lambda pick: kp.loc[pick].mean())
    ck["point"] = round(kid_mean, 4)
    out["iod_direct"] = {"slice_level_point": round(slice_mean, 4),
                         "patient39": cp, "kidney59": ck}

    # ---- C. reverse minus forward, paired ----
    rev2 = pd.read_csv(ROOT / "outputs/metrics/e4_reverse_slices_2d.csv")
    e2 = pd.read_csv(ROOT / "outputs/metrics/e2_external.csv")
    fwd = e2[(e2["model"] == "nnunet") & (e2["prompt"] == "none")]
    rp = rev2.groupby("patient")["dice"].mean()
    fp = fwd.groupby("patient")["dice"].mean()
    both = rp.index.intersection(fp.index)
    diff = (fp.loc[both] - rp.loc[both])
    cd = cluster_boot_stat(both, lambda pick: diff.loc[pick].mean())
    cd["point"] = round(float(diff.mean()), 4)
    out["domain_2d_slice_paired"] = cd

    rev3 = pd.read_csv(ROOT / "outputs/summary/e4_reverse_cv.csv")
    r3 = rev3[rev3["config"] == "3d_fullres"].groupby("patient")["dice3d"].mean()
    fwd3 = json.loads((OUT / "e4_forward_volume3d.json").read_text())
    ks = {}
    for s in ("42", "43", "44"):
        for k, v in fwd3[s]["per_kidney"].items():
            ks.setdefault(k, []).append(v)
    f3 = pd.DataFrame([{"kidney_id": k, "dice": float(np.mean(v))} for k, v in ks.items()])
    f3["patient"] = f3["kidney_id"].str[:3].astype(int)
    f3p = f3.groupby("patient")["dice"].mean()  # 3-seed per-kidney mean, patient-level
    both3 = r3.index.intersection(f3p.index)
    d3 = (f3p.loc[both3] - r3.loc[both3])
    cd3 = cluster_boot_stat(both3, lambda pick: d3.loc[pick].mean())
    cd3["point"] = round(float(d3.mean()), 4)
    cd3["forward_caliber"] = "per-kidney 3-seed mean of stacked E2 nnU-Net volumes"
    out["domain_3d_volume_paired"] = cd3

    # ---- B. R0 vs R2 (needs the megapass CSV; skip gracefully if absent) ----
    r2p = ROOT / "outputs/metrics/rev_medsam_tta_r2s43.csv"
    if r2p.exists():
        r2 = pd.read_csv(r2p)
        r0 = e2[(e2["model"] == "medsam") & (e2["prompt"] == "p4") &
                (e2["seed"] == 42)]
        common = set(r0["case_id"]) & set(r2["case_id"])
        a = r0[r0["case_id"].isin(common)].set_index("case_id")
        b = r2[r2["case_id"].isin(common)].set_index("case_id")
        dmap = (b["dice"] - a["dice"])
        pat_of = a["kidney_id"].str[:3].astype(str)
        dpat = dmap.groupby(pat_of).mean()
        cs = cluster_boot_stat(dpat.index, lambda pick: dpat.loc[pick].mean())
        cs["point"] = round(float(dpat.mean()), 4)
        cs["n_common_slices"] = len(common)
        cs["note"] = "R2 reported dice already includes empty-rule zeros"
        out["r0_vs_r2_strict_paired"] = cs

        p4npz = np.load(ROOT / "outputs/prompts/p4_box_trusted.npz", allow_pickle=True)
        detected = set()
        for k in p4npz.files:
            arr = np.asarray(p4npz[k], dtype=float)
            if arr.size == 4 and not np.isnan(arr).any():
                detected.add(k)
        fixed = sorted(detected & common)
        af = a.loc[fixed]; bf = b.loc[fixed]
        dfix = bf["dice"] - af["dice"]
        pf = af["kidney_id"].str[:3].astype(str)
        dp = dfix.groupby(pf).mean()
        cf = cluster_boot_stat(dp.index, lambda pick: dp.loc[pick].mean())
        cf["point"] = round(float(dp.mean()), 4)
        cf["n_fixed_subset"] = len(fixed)
        cf["r2_coverage_on_fixed"] = round(float((bf["dice"] > 0).mean()), 4)
        cf["note"] = ("fixed subset = registered P4 detection set; both arms "
                      "evaluated on identical slices; R2 misses -> dice 0")
        out["r0_vs_r2_conditional_fixedsubset_paired"] = cf
    else:
        out["r0_vs_r2_strict_paired"] = {"status": "pending megapass CSV"}

    # ---- D. CT correlation r CI ----
    ctd = json.loads((OUT / "ct_diagnostic_table.json").read_text())
    rec = []
    for pid, v in ctd["per_patient"].items():
        if v.get("us_fwd_dice") is not None:
            vol = v["ct_total_kidney_voxels"] * v["voxel_mm3"]
            rec.append((pid, vol, v["us_fwd_dice"]))
    df = pd.DataFrame(rec, columns=["patient", "vol", "dice"])
    r_point = float(np.corrcoef(df["vol"], df["dice"])[0, 1])
    cr = cluster_boot_stat(df["patient"], lambda pick: float(np.corrcoef(
        df.set_index("patient").loc[pick]["vol"],
        df.set_index("patient").loc[pick]["dice"])[0, 1]))
    cr["point"] = round(r_point, 4)
    cr["n_patients"] = int(df["patient"].nunique())
    out["ct_volume_vs_us_dice_r"] = cr

    (OUT / "supp_bootstrap_cis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
