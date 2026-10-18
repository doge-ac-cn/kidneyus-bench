#!/usr/bin/env python3
"""B-workshop assembly (runs AFTER m5_bpass_gpu_v2.py prints WORKER DONE on
all three GPUs; this script validates completeness before computing).

B2  prompt-ladder conditional Dice on the COMMON subset = slices detected by
    the registered P4 detector (seed43, conf>=0.25); sources P1/P5/P4-TTA-R2/
    P4-registered/oracle, all segmented by MedSAM seed42.
B3b joint patient-cluster bootstrap CIs (39 clusters x 2000, duplicates
    PRESERVED - same semantics as m5_joint_bootstrap.py) for strict and
    conditional Dice of MedSAM/SAM+LoRA/SAMUS under TTA-R2 boxes.
B4b TTA-R2 US predicted volume per kidney -> Bland-Altman vs CT reference.

Outputs:
  outputs/summary/prompt_ladder_common_subset.json
  outputs/summary/tta_criteria_bootstrap.json
  outputs/summary/volume_bland_altman.json  (tta_r2 entry appended)
  outputs/figures/ba_us_pred_vs_ct_tta_r2.png
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "outputs/summary"


def rows(tag):
    p = OUT.parent / "metrics" / f"bpass_{tag}.csv"
    df = pd.read_csv(p)
    if len(df) != 18322:
        raise RuntimeError(f"{p.name}: {len(df)} rows (expected 18,322) - pass incomplete, aborting")
    if df["patient"].nunique() != 39:
        raise RuntimeError(f"{p.name}: {df['patient'].nunique()} patients (expected 39)")
    return df


def main():
    # ---- common subset: P4-registered detected slices ----
    p4npz = np.load(ROOT / "outputs/prompts/p4_box_trusted.npz", allow_pickle=True)
    detected = set()
    for k in p4npz.files:
        arr = np.asarray(p4npz[k], dtype=float)
        if arr.size == 4 and not np.isnan(arr).any():
            detected.add(k)
    print(f"P4-registered detected slices: {len(detected)}", flush=True)

    tags = {"p1": "medsam_p1", "p5": "medsam_p5", "p4_tta_r2": "medsam_tta_r2s43",
            "p4_registered": "medsam_p4reg", "oracle": "medsam_oracle"}
    ladder = {"common_subset_n": len(detected)}
    for name, tag in tags.items():
        r = rows(tag)
        sub = r[r["case_id"].isin(detected)]
        ladder[name] = {
            "conditional_dice_on_common_subset": round(float(sub["dice"].mean()), 4),
            "coverage_on_common_subset": round(float((sub["pred_area_px"] > 0).mean()), 4),
            "n_conditional": int(len(sub))}
        print(f"{name}: cond={ladder[name]['conditional_dice_on_common_subset']} "
              f"cov={ladder[name]['coverage_on_common_subset']}", flush=True)
    (OUT / "prompt_ladder_common_subset.json").write_text(json.dumps(ladder, indent=1))

    # ---- B3b: TTA-R2 criteria bootstrap (3 models; duplicates preserved) ----
    out = {"n_boot": 2000, "clusters": 39,
           "bootstrap_semantics": "patient clusters resampled WITH replacement; "
                                  "duplicates preserved (same as m5_joint_bootstrap.py)"}
    for model, tag in (("medsam", "medsam_tta_r2s43"), ("sam_lora", "samlora_tta_r2s43"),
                       ("samus", "samus_tta_r2s43")):
        r = rows(tag)
        rng = np.random.default_rng(42)
        patients = np.array(sorted(r["patient"].unique()))
        assert len(patients) == 39
        strict_all = float(r["dice"].mean())
        cond_all = float(r.loc[r["pred_area_px"] > 0, "dice"].mean())
        strict_s = r.groupby("patient")["dice"].mean()
        cond_s = r[r["pred_area_px"] > 0].groupby("patient")["dice"].mean()
        cond_full = cond_s.reindex(patients)   # NaN = 该患者此轮无检出切片
        sp, cp = np.empty(2000), np.empty(2000)
        for it in range(2000):
            pick = rng.choice(patients, len(patients), replace=True)
            sp[it] = strict_s.loc[pick].mean()
            sel = cond_full.loc[pick]
            cp[it] = sel.mean(skipna=True) if not sel.isna().all() else np.nan
        cpv = cp[~np.isnan(cp)]
        out.setdefault(model, {})["strict"] = {
            "point": round(float(strict_all), 4),
            "ci95": [round(float(np.percentile(sp, 2.5)), 4),
                     round(float(np.percentile(sp, 97.5)), 4)]}
        out[model]["conditional"] = {
            "point": round(float(cond_all), 4),
            "ci95": [round(float(np.percentile(cpv, 2.5)), 4),
                     round(float(np.percentile(cpv, 97.5)), 4)],
            "note": "point = pooled detected-slice mean (matches the published "
                    "downstream caliber); CI = patient-cluster bootstrap of "
                    "per-patient conditional means (duplicates preserved)"}
    (OUT / "tta_criteria_bootstrap.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({m: out[m] for m in ("medsam", "sam_lora", "samus")}, indent=1), flush=True)

    # ---- B4b: TTA-R2 US predicted volume per kidney -> BA vs CT ----
    tta_rows = rows("medsam_tta_r2s43")
    usv = json.loads((OUT / "us_volumes.json").read_text())
    ctv = json.loads((OUT / "ct_reference_volumes.json").read_text())
    import SimpleITK as sitk
    _sp = sitk.ReadImage(str(ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA/"
                                 "US_masks/GT_estimated_masksUS/200R_maskUS.nii.gz")).GetSpacing()
    px_mm2 = _sp[0] * _sp[1]
    z_mm = _sp[2]
    assert abs(px_mm2 - 0.09) < 1e-6 and abs(z_mm - 0.3) < 1e-6, "US spacing changed"
    per_kid = tta_rows.groupby(["kidney_id", "patient"])["pred_area_px"].sum().reset_index()
    per_kid["patient"] = per_kid["patient"].astype(str)
    pairs = []
    for _, rr in per_kid.iterrows():
        kid, pid = rr["kidney_id"], str(rr["patient"])
        if pid not in ctv or kid not in usv:
            continue
        side = kid[3]
        ref = ctv[pid]["ct_R_ml"] if side == "R" else ctv[pid]["ct_L_ml"]
        vol = rr["pred_area_px"] * px_mm2 * z_mm / 1000.0
        pairs.append((vol, ref, kid))
    assert len(pairs) > 0, "B4b: no US-CT pairs - join broken"
    d = np.array([p - c for p, c, _ in pairs])
    m = np.array([(p + c) / 2 for p, c, _ in pairs])
    bias = float(d.mean())
    loa = (float(bias - 1.96 * d.std(ddof=1)), float(bias + 1.96 * d.std(ddof=1)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(m, d, s=22, alpha=0.75, color="#34495e")
    ax.axhline(bias, color="#e67e22", lw=1.5, label=f"bias {bias:+.1f} mL")
    ax.axhline(loa[0], color="#c0392b", ls="--", lw=1.2, label="95% LoA")
    ax.axhline(loa[1], color="#c0392b", ls="--", lw=1.2)
    ax.set_xlabel("mean of TTA-R2 US volume and CT reference (mL)")
    ax.set_ylabel("TTA-R2 US volume − CT reference (mL)")
    ax.set_title(f"Bland–Altman: TTA-R2 repaired vs CT (n={len(pairs)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "outputs/figures/ba_us_pred_vs_ct_tta_r2.png", dpi=150)
    no_box = float((tta_rows["pred_area_px"] == 0).mean())
    ba = {"config": "TTA-R2 (medsam42, r2_s43 detector)",
          "n": len(pairs), "bias_ml": round(bias, 2),
          "loa95_ml": [round(loa[0], 2), round(loa[1], 2)],
          "mean_us_pred_ml": round(float(np.mean([p for p, _, _ in pairs])), 2),
          "note": f"undetected slices contribute 0 volume; TTA-R2 no-box slice rate "
                  f"{no_box:.1%} (clustered near poles) biases US volumes downward - "
                  f"interpret the negative bias with this in mind"}
    vba = json.loads((OUT / "volume_bland_altman.json").read_text())
    vba["tta_r2_vs_ct"] = ba
    (OUT / "volume_bland_altman.json").write_text(json.dumps(vba, indent=1))
    print(json.dumps(ba, indent=1), flush=True)


if __name__ == "__main__":
    main()
