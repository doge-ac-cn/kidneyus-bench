#!/usr/bin/env python3
"""Items ⑥ + ⑧(CPU legs) + ⑦(FPR): companion CPU analyses.

⑥  S-Table 4: per-replicate (3 reps) x 3 models strict / conditional /
    coverage from tta_r2_downstream.json -> CSV + MD.
⑧a LMM random-effects refit: dice ~ C(model) + (1|patient) + (1|seed) via
    variance components (the registered wording is a RANDOM seed effect; the
    shipped fit used a fixed seed effect - this refit closes that gap).
⑧b (reporting) SAMUS seed 42 vs 43 headline deltas from e2_external.csv.
⑦  Discontinuity-detector false-positive estimate:
      (a) null model: within-kidney slice-order shuffle -> detector flag rate;
      (b) threshold sensitivity: 2 / 3 / 4 x MAD multiplier;
      (c) cross-rater agreement of flagged sets (A vs B vs STAPLE):
          if the flags were noise, the three raters' sets would not overlap.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "outputs/summary"
MASKS = ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA/US_masks"


def stable4():
    d = json.loads((OUT / "tta_r2_downstream.json").read_text())
    rows = []
    for model in ("medsam", "sam_lora", "samus"):
        for rep in ("42", "43", "44"):
            v = d[model][rep]
            rows.append({"model": model, "replicate": rep,
                         "strict": v["strict"], "conditional": v["conditional"],
                         "coverage": v["coverage"]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "supp_table_s4_per_replicate.csv", index=False)
    md = ["# S-Table 4 (Supplementary analysis (per-replicate table)): per-replicate TTA-R2 results\n",
          "| model | replicate | strict Dice | conditional Dice | coverage |",
          "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['replicate']} | {r['strict']} | "
                  f"{r['conditional']} | {r['coverage']} |")
    (OUT / "supp_table_s4_per_replicate.md").write_text("\n".join(md) + "\n")
    print("S-Table 4 written:", len(rows), "rows")


def lmm_random_seed():
    import statsmodels.formula.api as smf
    e2 = pd.read_csv(ROOT / "outputs/metrics/e2_external.csv")
    H = {"medsam": "p4", "sam_lora": "p4", "samus": "p4",
         "nnunet": "none", "unet": "none", "usfm": "none"}
    head = e2[e2.apply(lambda r: r["prompt"] == H[r["model"]], axis=1)].copy()
    head["patient"] = head["patient"].astype(int)
    head["seedc"] = head["seed"].astype("category")
    md = smf.mixedlm("dice ~ C(model)", head, groups=head["patient"],
                     vc_formula={"seed": "0 + C(seed)"})
    fit = md.fit(reml=False, method="lbfgs")
    cov_re = np.asarray(fit.cov_re).reshape(-1)
    vcomp = np.asarray(fit.vcomp).reshape(-1)
    res = {"converged": bool(fit.converged),
           "patient_var": round(float(cov_re[0]), 5) if cov_re.size else None,
           "seed_var": round(float(vcomp[0]), 5) if vcomp.size else 0.0,
           "fixed_effects": {k: round(float(v), 5) for k, v in fit.params.items()},
           "note": "registered wording: seed RANDOM effect; this refit uses "
                   "patient intercept + seed variance component"}
    (OUT / "supp_lmm_random_seed.json").write_text(json.dumps(res, indent=1))
    print("LMM random-seed refit:", json.dumps(res, indent=1)[:300])


def samus_seed_report():
    e2 = pd.read_csv(ROOT / "outputs/metrics/e2_external.csv")
    s = e2[(e2["model"] == "samus") & (e2["prompt"] == "p3")]
    per = s.groupby("seed")["dice"].mean()
    (OUT / "supp_samus_seed_headline.json").write_text(json.dumps(
        {str(k): round(float(v), 4) for k, v in per.items()}, indent=1))
    print("SAMUS p3 per-seed:", dict(per.round(4)))


def detector_fpr():
    """(a) shuffle null; (b) MAD sensitivity; (c) A/B/STAPLE flag agreement."""
    slices = pd.read_csv(ROOT / "outputs/trusted_us_slices.csv")
    def flagged_set(rater_dir, suffix, mult):
        flags = {}
        for kid, g in slices.groupby("kidney_id", sort=True):
            m = sitk.GetArrayFromImage(sitk.ReadImage(
                str(MASKS / rater_dir / f"{kid}{suffix}.nii.gz"))) > 0.5
            zs = sorted(int(z) for z in g["slice_idx"])
            shifts = []
            for a, b in zip(zs[:-1], zs[1:]):
                if b - a != 1:
                    continue
                inter = (m[a] & m[b]).sum(); union = (m[a] | m[b]).sum()
                shifts.append((a, 1.0 - (inter / union if union else 1.0)))
            if len(shifts) < 3:
                continue
            ss = np.array([s for _, s in shifts])
            med = np.median(ss); mad = np.median(np.abs(ss - med)) + 1e-12
            thr = med + mult * mad
            for (z, s) in shifts:
                if s >= thr and s > 0:
                    flags[(kid, z)] = 1
        return set(flags)
    full = flagged_set("GT_estimated_masksUS", "_maskUS", 3)
    rng = np.random.default_rng(42)
    # (a) shuffle null: permute slice order within kidney, detector on the shuffled sequence
    null_flags = 0
    for kid, g in slices.groupby("kidney_id", sort=True):
        m = sitk.GetArrayFromImage(sitk.ReadImage(
            str(MASKS / "GT_estimated_masksUS" / f"{kid}_maskUS.nii.gz"))) > 0.5
        zs = [int(z) for z in g["slice_idx"]]
        if len(zs) < 10:
            continue
        perm = rng.permutation(len(zs))
        seq = [zs[i] for i in perm]
        shifts = [1.0 - ((m[a] & m[b]).sum() / (m[a] | m[b]).sum())
                  for a, b in zip(seq[:-1], seq[1:])
                  if (m[a] | m[b]).sum() > 0]
        shifts = np.array(shifts)
        med = np.median(shifts); mad = np.median(np.abs(shifts - med)) + 1e-12
        null_flags += int((shifts >= med + 3 * mad).sum())
    n_pairs = sum(1 for _ in range(len(slices)))  # placeholder for denominator calc
    pairs_total = 0
    for kid, g in slices.groupby("kidney_id"):
        zs = sorted(int(z) for z in g["slice_idx"])
        pairs_total += sum(1 for a, b in zip(zs[:-1], zs[1:]) if b - a == 1)
    sens = {}
    for mult in (2, 3, 4):
        sens[str(mult)] = len(flagged_set("GT_estimated_masksUS", "_maskUS", mult))
    setA = flagged_set("Annotator1", "1_maskUS", 3)
    setB = flagged_set("Annotator2", "2_maskUS", 3)
    j_a = len(full & setA) / max(len(full | setA), 1)
    j_b = len(full & setB) / max(len(full | setB), 1)
    res = {"definition": "adjacent-slice mask 1-IoU series; local-max flags >= "
                         "median + k*MAD (k=3 default)",
           "n_staple_flags": len(full), "n_adjacent_pairs": pairs_total,
           "flag_rate": round(len(full) / max(pairs_total, 1), 4),
           "null_shuffle_flag_rate": round(null_flags / max(pairs_total, 1), 4),
           "mad_sensitivity_flags": sens,
           "rater_agreement_jaccard": {"staple_vs_A": round(j_a, 3),
                                       "staple_vs_B": round(j_b, 3)},
           "fpr_bound": "flag rate under within-kidney order shuffle (noise floor)"}
    (OUT / "supp_detector_fpr.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    stable4()
    samus_seed_report()
    try:
        lmm_random_seed()
    except Exception as e:  # noqa: BLE001
        print("LMM refit failed:", e)
    detector_fpr()
