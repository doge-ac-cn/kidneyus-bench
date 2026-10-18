#!/usr/bin/env python3
"""Seed-43 primary recomputation (registered seed): analysis driver.

Consumes the seed-43 pass CSVs and rewrites the supplementary outputs with
seed 43 as the primary caliber:

  supp_prompt_ladder_common_subset.json   (5 sources, MedSAM seed 43,
                                           common subset = P4-detected slices)
  supp_tta_criteria_bootstrap.json        (3 models x seed-43 weights under
                                           the TTA-R2 boxes, patient-cluster CI)
  supp_stratified_tables.{json,md}        (pole / discontinuity / rest;
                                           observer + model-vs-A/B rows)
  supp_boundary_metrics.{json,md}         (HD95 / NSD@1mm)
  supp_bootstrap_cis.json                 (R0 vs R2 paired with seed-43 rows;
                                           other legs unchanged 3-seed pooled)
  volume_bland_altman.json                (seed-43 US volume primary entry)

Run after all rev43_*.csv are complete.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "outputs/summary"
MET = ROOT / "outputs/metrics"
N_BOOT = 2000


def rd(name):
    p = MET / name
    df = pd.read_csv(p)
    if len(df) != 18322:
        raise RuntimeError(f"{name}: {len(df)} rows")
    return df


def cluster_ci(df, value="dice", seed=42):
    rng = np.random.default_rng(seed)
    pat = df.groupby("patient")[value].mean()
    pats = pat.index.values
    vals = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.choice(pats, len(pats), replace=True)
        vals[i] = pat.loc[pick].mean()
    return round(float(pat.mean()), 4), [round(float(np.percentile(vals, 2.5)), 4),
                                         round(float(np.percentile(vals, 97.5)), 4)]


def main():
    # ---------- 1) prompt ladder on the common subset (P4-detected) ----------
    p4npz = np.load(ROOT / "outputs/prompts/p4_box_trusted.npz", allow_pickle=True)
    detected = {k for k in p4npz.files
                if np.asarray(p4npz[k], dtype=float).size == 4
                and not np.isnan(np.asarray(p4npz[k], dtype=float)).any()}
    sources = {"p1": "rev43_medsam43_p1.csv",
               "p5_nnunet_s43": "rev43_medsam43_p5s43.csv",
               "p4_tta_r2": "rev43_medsam43_tta.csv",
               "p4_registered": "rev43_medsam43_p4reg.csv",
               "oracle": "rev43_medsam43_p3.csv"}
    ladder = {"caliber": "registered seed 43; MedSAM seed-43 weights; common subset "
                         "= slices detected by the registered P4 detector",
              "common_subset_n": len(detected)}
    for name, fn in sources.items():
        df = rd(fn)
        sub = df[df["case_id"].isin(detected)]
        ladder[name] = {"conditional_dice_on_common_subset": round(float(sub["dice"].mean()), 4),
                        "coverage_on_common_subset": round(float((sub["pred_area_px"] > 0).mean()), 4),
                        "n": int(len(sub))}
    (OUT / "supp_prompt_ladder_common_subset.json").write_text(json.dumps(ladder, indent=1))
    print("ladder:", json.dumps({k: v.get("conditional_dice_on_common_subset")
                                 for k, v in ladder.items() if isinstance(v, dict)}), flush=True)

    # ---------- 2) TTA criteria bootstrap (seed 43) ----------
    models = {"medsam": "rev43_medsam43_tta.csv",
              "sam_lora": "rev43_samlora43_tta.csv",
              "samus": "rev_samus43_tta_r2s43.csv"}  # model seed 43 (existing pass)
    out = {"caliber": "registered seed 43 weights under the TTA-R2 boxes",
           "n_boot": N_BOOT,
           "bootstrap": "patient clusters resampled with replacement, duplicates preserved"}
    for m, fn in models.items():
        df = rd(fn)
        strict_pt, strict_ci = cluster_ci(df)
        det = df[df["pred_area_px"] > 0]
        cond_pt, cond_ci = cluster_ci(det)
        out[m] = {"strict": {"point": strict_pt, "ci95": strict_ci},
                  "conditional": {"point": cond_pt, "ci95": cond_ci},
                  "coverage": round(float((df["pred_area_px"] > 0).mean()), 4)}
    (OUT / "supp_tta_criteria_bootstrap.json").write_text(json.dumps(out, indent=1))
    print("criteria:", json.dumps({m: out[m]["conditional"]["point"] for m in models}), flush=True)

    # ---------- 3) stratified tables (seed 43) ----------
    canon = json.loads((OUT / "canonical_pole_slices.json").read_text())
    pole = {f"{k}_z{z:03d}" for k, zs in canon["canonical_pole_slices"].items() for z in zs}
    ks = pd.read_csv(MET / "key_slices_trusted.csv")
    disc = {f"{r.kidney_id}_z{int(r.slice_idx):03d}" for r in ks.itertuples() if r.is_key == 1}
    iod = pd.read_csv(MET / "iod_ab_per_slice.csv")
    iod["cid"] = iod.apply(lambda r: f"{r.kidney_id}_z{int(r.slice_idx):03d}", axis=1)
    iod["layer"] = ["pole" if c in pole else ("discontinuity" if c in disc else "rest")
                    for c in iod["cid"]]
    strat = {"caliber": "registered seed 43 (nnU-Net seed-43 predictions; MedSAM seed-43 weights)",
             "definition": {"pole": "canonical pole slices, n=999",
                            "discontinuity": "boundary-discontinuity flags (pole takes "
                                             "precedence: 3,456 - 420 = 3,036)",
                            "rest": "all other scored slices"},
             "A_vs_B": {}, "models": {}}
    for layer in ("pole", "discontinuity", "rest"):
        d = iod[iod["layer"] == layer]["dice_ab"].values
        strat["A_vs_B"][layer] = {"n": int(len(d)),
                                  "dice_mean": round(float(d.mean()), 4),
                                  "iou_mean": round(float((d / (2 - d)).mean()), 4)}
    cfgs = {"nnunet": "rev43_nnunet_best.csv",
            "medsam_p3": "rev43_medsam43_p3.csv",
            "medsam_p4reg": "rev43_medsam43_p4reg.csv",
            "medsam_tta_r2": "rev43_medsam43_tta.csv",
            "samlora_p3": None,  # not re-run (only tta needed); omitted
            "samus_p3": None}
    for c, fn in cfgs.items():
        if fn is None:
            continue
        df = rd(fn).copy()
        df["layer"] = ["pole" if cid in pole else ("discontinuity" if cid in disc else "rest")
                       for cid in df["case_id"]]
        strat["models"][c] = {L: {"n": int(len(df[df["layer"] == L])),
                                  "dice_consensus": round(float(df[df["layer"] == L]["dice"].mean()), 4),
                                  "dice_vs_A": round(float(df[df["layer"] == L]["dice_A"].mean()), 4),
                                  "dice_vs_B": round(float(df[df["layer"] == L]["dice_B"].mean()), 4)}
                              for L in ("pole", "discontinuity", "rest")}
    (OUT / "supp_stratified_tables.json").write_text(json.dumps(strat, indent=1))
    md = ["# Observer-stratified tables (supplementary, registered seed 43)\n",
          "## Sonographer A vs B (direct, by layer)", "",
          "| layer | n | Dice | IoU |", "|---|---|---|---|"]
    for L in ("pole", "discontinuity", "rest"):
        v = strat["A_vs_B"][L]
        md.append(f"| {L} | {v['n']} | {v['dice_mean']} | {v['iou_mean']} |")
    md += ["", "_Layer note: pole slices take precedence (3,456 flags - 420 "
           "overlap = 3,036 discontinuity-only)._", "",
           "## Model vs consensus / A / B (by layer)", "",
           "| config | layer | n | vs consensus | vs A | vs B |", "|---|---|---|---|---|---|"]
    for c, rec in strat["models"].items():
        for L in ("pole", "discontinuity", "rest"):
            v = rec[L]
            md.append(f"| {c} | {L} | {v['n']} | {v['dice_consensus']} | {v['dice_vs_A']} | {v['dice_vs_B']} |")
    (OUT / "supp_stratified_tables.md").write_text("\n".join(md) + "\n")

    # ---------- 4) boundary metrics (seed 43) ----------
    bnd = {"caliber": "registered seed 43"}
    for c, fn in (("nnunet", "rev43_nnunet_best.csv"),
                  ("medsam_p3", "rev43_medsam43_p3.csv"),
                  ("medsam_p4reg", "rev43_medsam43_p4reg.csv"),
                  ("medsam_tta_r2", "rev43_medsam43_tta.csv"),
                  ("samlora_tta_r2", "rev43_samlora43_tta.csv"),
                  ("samus_tta_r2", "rev_samus43_tta_r2s43.csv")):
        df = rd(fn)
        h = df["hd95"].astype(float)
        bnd[c] = {"n": len(df),
                  "hd95_mean_valid": round(float(h.mean(skipna=True)), 2),
                  "hd95_valid_n": int(h.notna().sum()),
                  "hd95_nan_n": int(h.isna().sum()),
                  "nsd_mean": round(float(df["nsd"].mean()), 4),
                  "nsd_zero_n": int((df["nsd"] == 0).sum())}
    (OUT / "supp_boundary_metrics.json").write_text(json.dumps(bnd, indent=1))
    md4 = ["# Boundary metrics on TRUSTED (supplementary, registered seed 43)\n",
           "HD95 = conditional (valid slices). NSD@1 mm vs STAPLE; empty prediction -> 0.\n",
           "| config | n | HD95 mean (mm) | HD95 valid n | HD95 NaN n | NSD@1mm | NSD=0 n |",
           "|---|---|---|---|---|---|---|"]
    for c, v in bnd.items():
        if not isinstance(v, dict):
            continue
        md4.append(f"| {c} | {v['n']} | {v['hd95_mean_valid']} | {v['hd95_valid_n']} | "
                   f"{v['hd95_nan_n']} | {v['nsd_mean']} | {v['nsd_zero_n']} |")
    (OUT / "supp_boundary_metrics.md").write_text("\n".join(md4) + "\n")

    # ---------- 5) R0 vs R2 paired (seed 43) ----------
    cis = json.loads((OUT / "supp_bootstrap_cis.json").read_text())
    e2 = pd.read_csv(MET / "e2_external.csv")
    r0 = e2[(e2.model == "medsam") & (e2.prompt == "p4") & (e2.seed == 43)]
    r2 = rd("rev43_medsam43_tta.csv")
    common = set(r0["case_id"]) & set(r2["case_id"])
    a = r0[r0["case_id"].isin(common)].set_index("case_id")
    b = r2[r2["case_id"].isin(common)].set_index("case_id")
    dmap = b["dice"] - a["dice"]
    pat_of = a["kidney_id"].str[:3].astype(str)
    dpat = dmap.groupby(pat_of).mean()
    rng = np.random.default_rng(42)
    pats = dpat.index.values
    vals = np.array([dpat.loc[rng.choice(pats, len(pats), replace=True)].mean()
                     for _ in range(N_BOOT)])
    cis["r0_vs_r2_strict_paired"] = {
        "caliber": "registered seed 43",
        "point": round(float(dpat.mean()), 4),
        "ci95": [round(float(np.percentile(vals, 2.5)), 4),
                 round(float(np.percentile(vals, 97.5)), 4)],
        "n_common_slices": len(common)}
    detected_set = sorted(detected & common)
    af = a.loc[detected_set]; bf = b.loc[detected_set]
    dfix = bf["dice"] - af["dice"]
    pf = af["kidney_id"].str[:3].astype(str)
    dp = dfix.groupby(pf).mean()
    pats2 = dp.index.values
    vals2 = np.array([dp.loc[rng.choice(pats2, len(pats2), replace=True)].mean()
                      for _ in range(N_BOOT)])
    cis["r0_vs_r2_conditional_fixedsubset_paired"] = {
        "caliber": "registered seed 43; fixed subset = P4 detection set",
        "point": round(float(dp.mean()), 4),
        "ci95": [round(float(np.percentile(vals2, 2.5)), 4),
                 round(float(np.percentile(vals2, 97.5)), 4)],
        "n_fixed_subset": len(detected_set),
        "r2_coverage_on_fixed": round(float((bf["dice"] > 0).mean()), 4)}
    (OUT / "supp_bootstrap_cis.json").write_text(json.dumps(cis, indent=1))

    # ---------- 6) volume (seed 43) ----------
    ctv = json.loads((OUT / "ct_reference_volumes.json").read_text())
    usv = json.loads((OUT / "us_volumes.json").read_text())
    pr = ROOT / "outputs/predictions/nnunet_seed43_best/trusted"
    import SimpleITK as sitk
    sk = {}
    for row in pd.read_csv(ROOT / "outputs/trusted_us_slices.csv").itertuples():
        sk.setdefault(row.kidney_id, []).append(int(row.slice_idx))
    sp = sitk.ReadImage(str(ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA/US_masks/"
                                 "GT_estimated_masksUS/200R_maskUS.nii.gz")).GetSpacing()
    mm3 = sp[0] * sp[1] * sp[2]
    rows = []
    for kid, zs in sk.items():
        v = 0
        for z in zs:
            p = pr / f"{kid}_z{z}.nii.gz"
            v += int((sitk.GetArrayFromImage(sitk.ReadImage(str(p))) > 0).sum())
        pid = kid[:3]
        if pid not in ctv:
            continue
        ref = ctv[pid]["ct_R_ml"] if kid[3] == "R" else ctv[pid]["ct_L_ml"]
        rows.append((v * mm3 / 1000.0, ref, pid))
    df = pd.DataFrame(rows, columns=["us_ml", "ct_ml", "patient"])
    d = df["us_ml"] - df["ct_ml"]
    bias = float(d.mean())
    rng = np.random.default_rng(42)
    pats = df["patient"].unique()
    boot = np.array([(df[df["patient"].isin(rng.choice(pats, len(pats), replace=True))]
                      .pipe(lambda x: x["us_ml"] - x["ct_ml"]).mean()) for _ in range(N_BOOT)])
    vba = json.loads((OUT / "volume_bland_altman.json").read_text())
    vba["nnunet_zeroshot_vs_ct_seed43_primary"] = {
        "caliber": "registered seed 43",
        "n_kidney_pairs": int(len(df)), "n_patients": int(df["patient"].nunique()),
        "bias_ml": round(bias, 2),
        "bias_ci95_ml_patient39": [round(float(np.percentile(boot, 2.5)), 2),
                                   round(float(np.percentile(boot, 97.5)), 2)],
        "loa95_ml": [round(bias - 1.96 * float(d.std(ddof=1)), 2),
                     round(bias + 1.96 * float(d.std(ddof=1)), 2)],
        "mean_us_pred_ml": round(float(df["us_ml"].mean()), 2),
        "level": "per-kidney pair; patient-clustered CI"}
    # TTA-R2 volume with seed-43 medsam
    t = rd("rev43_medsam43_tta.csv")
    per = t.groupby(["kidney_id", "patient"])["pred_area_px"].sum().reset_index()
    per["patient"] = per["patient"].astype(str)
    rows2 = []
    for r in per.itertuples():
        pid = str(r.patient)
        if pid not in ctv or r.kidney_id not in usv:
            continue
        ref = ctv[pid]["ct_R_ml"] if r.kidney_id[3] == "R" else ctv[pid]["ct_L_ml"]
        rows2.append((r.pred_area_px * mm3 / 1000.0, ref, pid))
    df2 = pd.DataFrame(rows2, columns=["us_ml", "ct_ml", "patient"])
    d2 = df2["us_ml"] - df2["ct_ml"]
    bias2 = float(d2.mean())
    pats2 = df2["patient"].unique()
    boot2 = np.array([(df2[df2["patient"].isin(rng.choice(pats2, len(pats2), replace=True))]
                       .pipe(lambda x: x["us_ml"] - x["ct_ml"]).mean()) for _ in range(N_BOOT)])
    vba["tta_r2_vs_ct_seed43"] = {
        "caliber": "registered seed 43 (MedSAM seed-43 weights)",
        "n_kidney_pairs": int(len(df2)), "bias_ml": round(bias2, 2),
        "bias_ci95_ml_patient39": [round(float(np.percentile(boot2, 2.5)), 2),
                                   round(float(np.percentile(boot2, 97.5)), 2)],
        "loa95_ml": [round(bias2 - 1.96 * float(d2.std(ddof=1)), 2),
                     round(bias2 + 1.96 * float(d2.std(ddof=1)), 2)],
        "mean_us_pred_ml": round(float(df2["us_ml"].mean()), 2)}
    (OUT / "volume_bland_altman.json").write_text(json.dumps(vba, indent=1))
    print(json.dumps({k: vba[k] for k in ("nnunet_zeroshot_vs_ct_seed43_primary", "tta_r2_vs_ct_seed43")},
                     indent=1), flush=True)
    print("SEED43 ANALYSIS DONE", flush=True)


if __name__ == "__main__":
    main()
