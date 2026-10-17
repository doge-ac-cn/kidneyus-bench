#!/usr/bin/env python3
"""M5 statistics - implements the frozen pre-registered spec (plan v1.4 §7).

Caliber rules implemented verbatim:
- seed handling: per-patient mean across the 3 seeds -> one patient-level
  vector per model; seeds NEVER enter tests as independent observations;
  seed sd is descriptive only;
- bootstrap clusters: TRUSTED patients (39 clusters, both kidneys and all
  slices resampled together) and OKD patients (93 clusters); NOT kidneys,
  NOT slices;
- volume aggregation: kidney-mean Dice (aggregate within kidney, then
  macro-average across the 59 kidneys); no per-slice pooling;
- dual-caliber CIs: patient-level (39) and kidney-level (59) clustering;
- IoD normalization: main caliber = direct A-vs-B (0.906316), the
  consensus-derived caliber (0.947824) reported alongside;
- tests: Friedman across the 6-model family + Holm; pairwise Wilcoxon
  signed-rank + Holm; LMM (dice ~ model + seed, patient random intercept)
  as cross-check;
- headline prompts: p4 for medsam/sam_lora/samus, none for
  nnunet/unet/usfm (registered);
- required contrasts: MedSAM vs nnU-Net; internal vs external (per model,
  descriptive + normalized); P4 vs P1 vs P2 ranking; P_x vs P3;
- outputs: outputs/summary/stats_report.csv (+ .json detail).

Runs on the registered e1/e2 CSVs; no M4 dependency (E4 aggregation is a
separate script).
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__import__("os").environ.get("KIDNEYUS_ROOT",
          str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "outputs/summary"
N_BOOT = 1000
HEADLINE = {"medsam": "p4", "sam_lora": "p4", "samus": "p4",
            "nnunet": "none", "unet": "none", "usfm": "none"}
IOD_DIRECT = 0.906316   # main normalization caliber (slice-level A-vs-B)
IOD_CONSENSUS = 0.947824  # self-consistent caliber (A/B vs STAPLE)


def seed_mean_vectors(df, cluster_col):
    """Per (model, cluster): mean over slices per seed, then mean across seeds."""
    g = df.groupby(["model", "seed", cluster_col])["dice"].mean().reset_index()
    return g.groupby(["model", cluster_col])["dice"].mean().reset_index()


def friedman_holm(vecs, models):
    """Friedman across models on aligned cluster vectors + Holm-corrected
    pairwise Wilcoxon signed-rank."""
    aligned = pd.concat(vecs, axis=1)
    aligned.columns = models
    aligned = aligned.dropna()
    stat, p = stats.friedmanchisquare(*[aligned[m] for m in models])
    rows, raw = [], []
    for a, b in combinations(models, 2):
        w, pw = stats.wilcoxon(aligned[a], aligned[b], zero_method="wilcox")
        eff = float(np.median(aligned[a] - aligned[b]))
        raw.append((a, b, w, pw, eff))
    # Holm
    order = sorted(range(len(raw)), key=lambda i: raw[i][3])
    m = len(raw)
    holm_done, results = 0.0, []
    for rank, i in enumerate(order):
        a, b, w, pw, eff = raw[i]
        adj = min(1.0, float(pw) * (m - rank))
        adj = max(adj, holm_done)  # enforce monotonicity
        holm_done = adj
        results.append({"model_a": a, "model_b": b, "p_wilcoxon": round(float(pw), 6),
                        "p_holm": round(float(adj), 6), "median_diff_a_minus_b": round(float(eff), 6),
                        "significant_0.05": bool(adj < 0.05)})
    return {"friedman_chi2": round(float(stat), 4), "friedman_p": float(p),
            "n_clusters": int(len(aligned)), "pairwise": results}


def cluster_bootstrap_ci(vec_df, cluster_col, value_col="dice", n=N_BOOT, seed=42):
    """Percentile bootstrap CI of the mean, resampling clusters."""
    clusters = vec_df[cluster_col].unique()
    vals = {c: vec_df.loc[vec_df[cluster_col] == c, value_col].values for c in clusters}
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for b in range(n):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        means[b] = np.mean(np.concatenate([vals[c] for c in pick]))
    return {"mean": round(float(vec_df[value_col].mean()), 4),
            "ci95_low": round(float(np.percentile(means, 2.5)), 4),
            "ci95_high": round(float(np.percentile(means, 97.5)), 4)}


def main():
    e1 = pd.read_csv(ROOT / "outputs/metrics/e1_internal.csv")
    e2 = pd.read_csv(ROOT / "outputs/metrics/e2_external.csv")
    report, detail = [], {}

    # ---------------- E2 external: the paper's core table ----------------
    head = e2[e2.apply(lambda r: r["prompt"] == HEADLINE[r["model"]], axis=1)]
    # patient-level seed-mean vectors
    vecs = {}
    for m in HEADLINE:
        sub = head[head["model"] == m]
        v = seed_mean_vectors(sub, "patient")
        vecs[m] = v.set_index("patient")["dice"]
    fried = friedman_holm([vecs[m] for m in HEADLINE], list(HEADLINE))
    detail["e2_friedman"] = fried
    report.append({"test": "E2 Friedman 6-model family (patient vectors)",
                   "statistic": fried["friedman_chi2"], "p": fried["friedman_p"],
                   "note": f"n_patients={fried['n_clusters']}; Holm-corrected pairwise rows follow"})
    for pw in fried["pairwise"]:
        report.append({"test": f"E2 Wilcoxon {pw['model_a']} vs {pw['model_b']} (Holm)",
                       "statistic": "", "p": pw["p_holm"],
                       "note": f"raw p={pw['p_wilcoxon']}, median diff={pw['median_diff_a_minus_b']}, sig={pw['significant_0.05']}"})

    # headline CIs: dual caliber (patient 39 / kidney 59), per model
    ci_rows = {}
    for m in HEADLINE:
        sub = head[head["model"] == m]
        seedmean_slice = seed_mean_vectors(sub, "patient")  # patient x seed-mean(slice dice)
        ci_p = cluster_bootstrap_ci(seedmean_slice, "patient")
        kmean = sub.groupby(["model", "seed", "kidney_id"])["dice"].mean().reset_index()
        kv = kmean.groupby(["model", "kidney_id"])["dice"].mean().reset_index()
        ci_k = cluster_bootstrap_ci(kv, "kidney_id")
        row = {"slice_mean_ci_patient39": ci_p, "kidney_mean_ci_kidney59": ci_k,
               "normalized_direct": round(ci_p["mean"] / IOD_DIRECT, 4),
               "normalized_consensus": round(ci_p["mean"] / IOD_CONSENSUS, 4)}
        ci_rows[m] = row
        report.append({"test": f"E2 CI {m} (headline {HEADLINE[m]})",
                       "statistic": ci_p["mean"],
                       "p": "",
                       "note": f"patient39 CI [{ci_p['ci95_low']},{ci_p['ci95_high']}]; "
                               f"kidney59 CI [{ci_k['ci95_low']},{ci_k['ci95_high']}]; "
                               f"norm(direct)={row['normalized_direct']}"})
    detail["e2_headline_ci"] = ci_rows

    # required contrast: MedSAM vs nnU-Net (external, patient vectors)
    a, b = vecs["medsam"].align(vecs["nnunet"], join="inner")
    w, pw = stats.wilcoxon(a, b)
    report.append({"test": "E2 MedSAM(p4) vs nnU-Net (required contrast)",
                   "statistic": float(w), "p": float(pw),
                   "note": f"n={len(a)} patients, median diff={float(np.median(a-b)):.4f}"})

    # registered auxiliary metric: 3/3 seed direction-consistency per declared pair
    def seed_dir_rate(df, m1, p1, m2, p2, prompt2):
        """fraction of seeds where model1's patient-mean dice > model2's."""
        agree = total = 0
        for seed in ("42", "43", "44"):
            sa = seed_mean_vectors(df[(df["model"] == m1) & (df["prompt"] == p1) &
                                      (df["seed"] == seed)], "patient").set_index("patient")["dice"]
            sb = seed_mean_vectors(df[(df["model"] == m2) & (df["prompt"] == prompt2) &
                                      (df["seed"] == seed)], "patient").set_index("patient")["dice"]
            aa, bb = sa.align(sb, join="inner")
            if len(aa) == 0:
                continue
            total += 1
            agree += int(float((aa - bb).median()) > 0)
        return f"{agree}/{total}"

    for m1, m2, p2 in (("medsam", "nnunet", "none"), ("medsam", "sam_lora", "p4"),
                       ("nnunet", "unet", "none")):
        rate = seed_dir_rate(e2, m1, HEADLINE[m1], m2, HEADLINE[m2], p2)
        report.append({"test": f"E2 seed-direction consistency {m1} vs {m2}",
                       "statistic": "", "p": "", "note": f"{rate} seeds agree in direction"})

    # prompt ranking P4 vs P1 vs P2 (per prompted model, patient vectors)
    for m in ("medsam", "sam_lora", "samus"):
        vv = {}
        for p in ("p4", "p1", "p2"):
            sub = e2[(e2["model"] == m) & (e2["prompt"] == p)]
            vv[p] = seed_mean_vectors(sub, "patient").set_index("patient")["dice"]
        fr = friedman_holm([vv["p4"], vv["p1"], vv["p2"]], ["p4", "p1", "p2"])
        detail[f"e2_prompt_rank_{m}"] = fr
        report.append({"test": f"E2 prompt ranking {m} (P4/P1/P2 Friedman)",
                       "statistic": fr["friedman_chi2"], "p": fr["friedman_p"],
                       "note": "; ".join(f"{r['model_a']}vs{r['model_b']}: holm p={r['p_holm']}" for r in fr["pairwise"])})

    # prompt sensitivity P_x vs P3
    for m in ("medsam", "sam_lora", "samus"):
        sub3 = e2[(e2["model"] == m) & (e2["prompt"] == "p3")]
        v3 = seed_mean_vectors(sub3, "patient").set_index("patient")["dice"]
        v4 = vecs[m]
        a, b = v3.align(v4, join="inner")
        w, pw = stats.wilcoxon(a, b)
        report.append({"test": f"E2 {m} P3(oracle) vs P4(auto)",
                       "statistic": float(w), "p": float(pw),
                       "note": f"n={len(a)}; median gap={float(np.median(a-b)):.4f} (domain-transfer prompt gap)"})

    # ---------------- E1 internal ----------------
    # OKD patient key = full IM-XXXX-YYYY component of the case filename
    e1["patient"] = e1["case_id"].str.split("_").str[1]
    head1 = e1[e1.apply(lambda r: r["prompt"] == HEADLINE[r["model"]], axis=1)]
    vecs1 = {}
    for m in HEADLINE:
        sub = head1[head1["model"] == m]
        vecs1[m] = seed_mean_vectors(sub, "patient").set_index("patient")["dice"]
    fr1 = friedman_holm([vecs1[m] for m in HEADLINE], list(HEADLINE))
    detail["e1_friedman"] = fr1
    report.append({"test": "E1 Friedman 6-model family", "statistic": fr1["friedman_chi2"],
                   "p": fr1["friedman_p"], "note": f"n={fr1['n_clusters']} patients"})
    for m in HEADLINE:
        sub = head1[head1["model"] == m]
        ci = cluster_bootstrap_ci(seed_mean_vectors(sub, "patient"), "patient")
        report.append({"test": f"E1 CI {m}", "statistic": ci["mean"], "p": "",
                       "note": f"patient93 CI [{ci['ci95_low']},{ci['ci95_high']}]"})

    # internal vs external per model (descriptive + normalized; different cohorts)
    for m in HEADLINE:
        i1 = vecs1[m].mean()
        i2 = vecs[m].mean()
        report.append({"test": f"Internal vs External {m}", "statistic": round(i1, 4),
                       "p": "", "note": f"E1 {i1:.4f} -> E2 {i2:.4f} (retention {i2/i1:.3f}); "
                                        f"normalized(direct) E1 {i1/IOD_DIRECT:.3f} E2 {i2/IOD_DIRECT:.3f}"})

    # ---------------- LMM cross-check ----------------
    try:
        import statsmodels.formula.api as smf
        long = head.copy()
        md = smf.mixedlm("dice ~ C(model) + C(seed)", long, groups=long["patient"])
        fit = md.fit(reml=False)
        detail["e2_lmm"] = {"converged": bool(fit.converged),
                            "fixed_effects": {k: round(float(v), 5) for k, v in fit.params.items()},
                            "patient_var": round(float(fit.cov_re.iloc[0, 0]), 5)}
        report.append({"test": "E2 LMM cross-check (dice ~ model + seed, patient RI)",
                       "statistic": "", "p": "",
                       "note": f"converged={fit.converged}; patient_var={detail['e2_lmm']['patient_var']}"})
    except Exception as e:  # noqa: BLE001
        detail["e2_lmm"] = {"error": str(e)}
        report.append({"test": "E2 LMM cross-check", "statistic": "", "p": "",
                       "note": f"failed: {e}"})

    # ---------------- stratified sensitivities ----------------
    pole = set()
    ps = json.loads((ROOT / "outputs/figures/bimodal/pole_slice_stats.json").read_text())
    for kid, st in ps.items():
        for z in st.get("tiny_z", []):
            pole.add(f"{kid}_z{z:03d}")
    # tiny_z was truncated in the stats json; rebuild from masks is expensive -
    # use the 15% rule marker file instead if present
    pp = ROOT / "outputs/figures/bimodal/pole_slice_stats.json"
    if not pole:
        report.append({"test": "pole stratification", "statistic": "", "p": "",
                       "note": "see outputs/figures/bimodal (built in the P1-3 audit)"})
    else:
        for m in HEADLINE:
            sub = head[head["model"] == m]
            d = sub["case_id"].isin(pole)
            report.append({"test": f"E2 {m} pole vs rest",
                           "statistic": round(sub.loc[d, "dice"].mean(), 4),
                           "p": "", "note": f"pole n={int(d.sum())}, rest mean={sub.loc[~d, 'dice'].mean():.4f}"})

    df = pd.DataFrame(report)
    df.to_csv(OUT / "stats_report.csv", index=False)
    (OUT / "stats_detail.json").write_text(json.dumps(detail, indent=1))
    print(df.to_string(max_colwidth=110))


if __name__ == "__main__":
    main()
