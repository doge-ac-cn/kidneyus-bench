#!/usr/bin/env python3
"""B3a: joint patient-cluster bootstrap CIs (39/93 clusters, 2000 iters).

Joint = one resample re-computes BOTH legs (or numerator and IoD denominator)
of each quantity. TTA-R2 legs are deferred to B3b (needs per-slice rows from
the GPU mega-pass).

Legs implemented here:
  normalized_external : E2 nnU-Net slice-mean / IoD-direct slice-mean (joint)
  normalized_p4       : MedSAM p4   / IoD (joint)
  normalized_oracle   : MedSAM p3   / IoD (joint)
  normalized_internal : E1 nnU-Net  / 0.906316 fixed (93 OKD clusters; the
                        IoD is TRUSTED-only, so the denominator is a constant
                        here - documented deviation)
  domain_2d_slice     : fwd(2d slice) - rev(2d slice)      (joint)
  domain_3d_volume    : fwd(3d volume) - rev(3d volume)    (joint)
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
N_BOOT = 2000
IOD = 0.906316


def patient_vec(df, value_col):
    return df.groupby("patient")[value_col].mean()


def joint_ratio_ci(num_df, den_df, clusters, n=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    num = patient_vec(num_df, "dice")
    den = patient_vec(den_df, "dice_ab" if "dice_ab" in den_df.columns else "dice")
    point = float(num.mean() / den.mean())
    stats = np.empty(n)
    for b in range(n):
        pick = rng.choice(clusters, len(clusters), replace=True)
        stats[b] = num.loc[pick].mean() / den.loc[pick].mean()
    return {"point": round(point, 4),
            "ci95": [round(float(np.percentile(stats, 2.5)), 4),
                     round(float(np.percentile(stats, 97.5)), 4)]}


def joint_diff_ci(a_df, b_df, clusters, n=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    a = patient_vec(a_df, "dice")
    b = patient_vec(b_df, "dice")
    point = float(a.mean() - b.mean())
    stats = np.empty(n)
    for it in range(n):
        pick = rng.choice(clusters, len(clusters), replace=True)
        stats[it] = a.loc[pick].mean() - b.loc[pick].mean()
    return {"point": round(point, 4),
            "ci95": [round(float(np.percentile(stats, 2.5)), 4),
                     round(float(np.percentile(stats, 97.5)), 4)]}


def main():
    e2 = pd.read_csv(ROOT / "outputs/metrics/e2_external.csv")
    iod = pd.read_csv(ROOT / "outputs/metrics/iod_ab_per_slice.csv")
    rev2 = pd.read_csv(ROOT / "outputs/metrics/e4_reverse_slices_2d.csv")
    rev3 = pd.read_csv(ROOT / "outputs/metrics/e4_reverse_slices_3d_fullres.csv")
    e4kv = pd.read_csv(ROOT / "outputs/summary/e4_reverse_cv.csv")
    fwd3 = json.loads((ROOT / "outputs/summary/e4_forward_volume3d.json").read_text())
    e1 = pd.read_csv(ROOT / "outputs/metrics/e1_internal.csv")
    e1["patient"] = e1["case_id"].str.split("_").str[1]

    clusters = sorted(iod["patient"].unique())
    assert len(clusters) == 39, len(clusters)

    nnunet_fwd = e2[(e2["model"] == "nnunet") & (e2["prompt"] == "none")]
    medsam_p4 = e2[(e2["model"] == "medsam") & (e2["prompt"] == "p4")]
    medsam_p3 = e2[(e2["model"] == "medsam") & (e2["prompt"] == "p3")]

    # align forward/reverse slices to iod patients (inner join keeps 39 clusters)
    rev2 = rev2[rev2["patient"].isin(clusters)]
    rev3 = rev3[rev3["patient"].isin(clusters)]
    nnunet_fwd = nnunet_fwd[nnunet_fwd["patient"].isin(clusters)]
    iod_al = iod[iod["patient"].isin(clusters)]

    out = {"n_boot": N_BOOT, "clusters": 39,
           "normalized_external_joint": joint_ratio_ci(nnunet_fwd, iod_al, clusters),
           "normalized_p4_joint": joint_ratio_ci(medsam_p4, iod_al, clusters),
           "normalized_oracle_joint": joint_ratio_ci(medsam_p3, iod_al, clusters)}

    # internal (93 OKD clusters, numerator-only: IoD is TRUSTED-only)
    nnunet_int = e1[(e1["model"] == "nnunet") & (e1["prompt"] == "none")]
    pv = nnunet_int.groupby("patient")["dice"].mean()
    rng = np.random.default_rng(42)
    c93 = pv.index.values
    stats = np.array([pv.loc[rng.choice(c93, len(c93), replace=True)].mean() for _ in range(N_BOOT)])
    out["normalized_internal"] = {"point": round(float(pv.mean() / IOD), 4),
                                  "ci95": [round(float(np.percentile(stats / IOD, 2.5)), 4),
                                           round(float(np.percentile(stats / IOD, 97.5)), 4)],
                                  "note": "numerator bootstrap only (OKD clusters; "
                                          "IoD is a TRUSTED constant here)"}

    # domain legs
    out["domain_2d_slice_joint"] = joint_diff_ci(nnunet_fwd, rev2, clusters)
    # 3d volume: forward = per-kidney stacked dice (json) -> patient
    fk = pd.DataFrame([{"kidney_id": k, "dice": v} for k, v in
                       fwd3["42"]["per_kidney"].items()])
    fk["patient"] = fk["kidney_id"].str.replace("TRUSTED_", "").str[:3].astype(int)
    rk3 = e4kv[(e4kv["config"] == "3d_fullres")].rename(columns={"dice3d": "dice"})
    rk3["patient"] = rk3["patient"].astype(int)
    out["domain_3d_volume_joint"] = joint_diff_ci(fk, rk3, clusters)

    (ROOT / "outputs/summary/joint_bootstrap_cis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
