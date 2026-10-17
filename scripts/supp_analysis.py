#!/usr/bin/env python3
"""Items ① and ④ analysis (run AFTER supp_megapass.py completes):

① observer-stratified tables on pole vs discontinuity vs rest layers:
     - sonographer A vs B direct Dice and IoU (per layer)
     - model-vs-consensus / model-vs-A / model-vs-B for MedSAM P3, MedSAM P4,
       nnU-Net (plus the other megapass configs for completeness)
   plus four-way overlay PNGs (A / B / STAPLE contours + model prediction) for
   a small set of representative slices incl. 704L.
④ boundary metrics table: HD95 (valid-n / nan-n, conditional caliber) and
   NSD@1mm for nnU-Net, the three prompted models under P3 and P4-registered,
   and the TTA-R2 repair arm (3 models).

Outputs:
  outputs/summary/supp_stratified_tables.json + .md
  outputs/summary/supp_boundary_metrics.json + .md
  outputs/figures/supp/overlay_{kid}_z{z}.png
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "outputs/summary"
MET = ROOT / "outputs/metrics"
FIG = ROOT / "outputs/figures/supp"

CONFIGS = ["nnunet", "medsam_p3", "medsam_p4reg", "medsam_tta_r2s43",
           "samlora_p3", "samlora_p4reg", "samlora_tta_r2s43",
           "samus_p3", "samus_p4reg", "samus_tta_r2s43", "samus43_tta_r2s43"]
FILES = {"nnunet": "rev_nnunet_best.csv",
         **{c: f"rev_{c}.csv" for c in CONFIGS if c != "nnunet"}}


def load_all():
    out = {}
    for c, f in FILES.items():
        p = MET / f
        if not p.exists():
            print(f"[missing] {c} ({f})")
            continue
        df = pd.read_csv(p)
        assert len(df) == 18322, (c, len(df))
        out[c] = df
    return out


def layers():
    canon = json.loads((OUT / "canonical_pole_slices.json").read_text())
    pole = {f"{k}_z{z:03d}" for k, zs in canon["canonical_pole_slices"].items() for z in zs}
    ks = pd.read_csv(MET / "key_slices_trusted.csv")
    disc = {f"{r.kidney_id}_z{int(r.slice_idx):03d}" for r in ks.itertuples() if r.is_key == 1}
    return pole, disc


def main():
    data = load_all()
    iod = pd.read_csv(MET / "iod_ab_per_slice.csv")
    iod["cid"] = iod.apply(lambda r: f"{r.kidney_id}_z{int(r.slice_idx):03d}", axis=1)
    pole, disc = layers()
    iod["layer"] = ["pole" if c in pole else ("discontinuity" if c in disc else "rest")
                    for c in iod["cid"]]

    # ---- ① stratified tables ----
    strat = {"definition": {
        "pole": "canonical pole slices (GT area < 15% of kidney median), n=999",
        "discontinuity": "boundary-discontinuity detector flags, n=3456 (scored subset)",
        "rest": "all other scored slices"},
        "A_vs_B": {}, "models": {}}
    for layer in ("pole", "discontinuity", "rest"):
        sub = iod[iod["layer"] == layer]
        d = sub["dice_ab"].values
        iou = d / (2.0 - d)  # IoU = Dice/(2-Dice)
        strat["A_vs_B"][layer] = {"n": int(len(d)),
                                  "dice_mean": round(float(d.mean()), 4),
                                  "iou_mean": round(float(iou.mean()), 4),
                                  "iou_median": round(float(np.median(iou)), 4)}
    for c, df in data.items():
        df = df.copy()
        df["layer"] = ["pole" if cid in pole else ("discontinuity" if cid in disc else "rest")
                       for cid in df["case_id"]]
        rec = {}
        for layer in ("pole", "discontinuity", "rest"):
            s = df[df["layer"] == layer]
            rec[layer] = {"n": int(len(s)),
                          "dice_consensus": round(float(s["dice"].mean()), 4),
                          "dice_vs_A": round(float(s["dice_A"].mean()), 4),
                          "dice_vs_B": round(float(s["dice_B"].mean()), 4)}
        strat["models"][c] = rec
    (OUT / "supp_stratified_tables.json").write_text(json.dumps(strat, indent=1))

    md = ["# Observer-stratified tables (supplementary)\n",
          "## Sonographer A vs B (direct, by layer)", "",
          "| layer | n | Dice | IoU |", "|---|---|---|---|"]
    for layer in ("pole", "discontinuity", "rest"):
        v = strat["A_vs_B"][layer]
        md.append(f"| {layer} | {v['n']} | {v['dice_mean']} | {v['iou_mean']} |")
    md += ["", "_Layer note: pole slices take precedence; the discontinuity "
           "layer counts exclude the 420 slices that are both pole and "
           "discontinuity (3,456 flags - 420 = 3,036). _", "",
           "## Model vs consensus / A / B (by layer)", "",
           "| config | layer | n | vs consensus | vs A | vs B |", "|---|---|---|---|---|---|"]
    for c in ["nnunet", "medsam_p3", "medsam_p4reg", "medsam_tta_r2s43"] + \
             [x for x in strat["models"] if x not in ("nnunet", "medsam_p3", "medsam_p4reg", "medsam_tta_r2s43")]:
        for layer in ("pole", "discontinuity", "rest"):
            v = strat["models"][c][layer]
            md.append(f"| {c} | {layer} | {v['n']} | {v['dice_consensus']} | "
                      f"{v['dice_vs_A']} | {v['dice_vs_B']} |")
    (OUT / "supp_stratified_tables.md").write_text("\n".join(md) + "\n")

    # ---- ④ boundary metrics ----
    bnd = {}
    for c, df in data.items():
        h = df["hd95"].astype(float)
        bnd[c] = {"n": len(df),
                  "hd95_mean_valid": round(float(h.mean(skipna=True)), 2),
                  "hd95_valid_n": int(h.notna().sum()),
                  "hd95_nan_n": int(h.isna().sum()),
                  "nsd_mean": round(float(df["nsd"].mean()), 4),
                  "nsd_zero_n": int((df["nsd"] == 0).sum())}
    (OUT / "supp_boundary_metrics.json").write_text(json.dumps(bnd, indent=1))
    md4 = ["# Boundary metrics on TRUSTED (supplementary)\n",
           "HD95 = conditional (valid slices; NaN when prediction or GT empty). "
           "NSD@1 mm = normalized surface Dice vs STAPLE; empty prediction -> 0.\n",
           "| config | n | HD95 mean (mm) | HD95 valid n | HD95 NaN n | NSD@1mm | NSD=0 n |",
           "|---|---|---|---|---|---|---|"]
    for c in CONFIGS:
        if c not in bnd:
            continue
        v = bnd[c]
        md4.append(f"| {c} | {v['n']} | {v['hd95_mean_valid']} | {v['hd95_valid_n']} | "
                   f"{v['hd95_nan_n']} | {v['nsd_mean']} | {v['nsd_zero_n']} |")
    (OUT / "supp_boundary_metrics.md").write_text("\n".join(md4) + "\n")
    print(json.dumps({"stratified": "written", "boundary": "written"}, indent=1))

    # ---- ① four-way overlays (re-inference on 3 slices, incl. 704L) ----
    if "--overlays" in sys.argv:  # optional, off by default
        overlay_slices()


def overlay_slices():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import SimpleITK as sitk
    import torch
    from m3_common import iter_trusted, load_sam_family, infer_sam_family, MODEL_RES, oracle_box

    picks = [("704L", 283, "pole"),
             ("861L", 215, "pole (second kidney)"),
             ("200R", 298, "annotation-range edge")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_sam_family("medsam", 42, device)
    RAW = ROOT / "data/raw/TRUSTED/TRUSTED_dataset_for_nsd/US_DATA/US_masks"
    FIG.mkdir(parents=True, exist_ok=True)
    done = set()
    for cid, img, gt, spacing, meta in iter_trusted():
        key = (meta["kidney_id"], meta["slice_idx"])
        if key in [(k, z) for k, z, _ in picks]:
            kid, z = key
            A = sitk.GetArrayFromImage(sitk.ReadImage(
                str(RAW / "Annotator1" / f"{kid}1_maskUS.nii.gz")))[z] > 0.5
            B = sitk.GetArrayFromImage(sitk.ReadImage(
                str(RAW / "Annotator2" / f"{kid}2_maskUS.nii.gz")))[z] > 0.5
            box = oracle_box(gt)
            pred = infer_sam_family(model, img, MODEL_RES["medsam"], device,
                                    box=box).astype(bool)
            ys, xs = np.where(gt)
            cy, cx = int(ys.mean()), int(xs.mean())
            half = 90
            y0, y1 = max(0, cy - half), min(img.shape[0], cy + half)
            x0, x1 = max(0, cx - half), min(img.shape[1], cx + half)
            crop = img[y0:y1, x0:x1].astype(np.float32)
            crop = (crop - crop.min()) / max(crop.max() - crop.min(), 1e-6)
            up = 3
            crop = np.repeat(np.repeat(crop, up, 0), up, 1)
            fig, ax = plt.subplots(figsize=(4.4, 4.4))
            ax.imshow(crop, cmap="gray", interpolation="nearest")
            for mask, col, lw, lab in ((gt, "#2ecc71", 1.5, "STAPLE"),
                                       (A, "#3498db", 1.2, "annotator A"),
                                       (B, "#f39c12", 1.2, "annotator B"),
                                       (pred, "#e74c3c", 1.5, "MedSAM (oracle box)")):
                m = np.repeat(np.repeat(mask[y0:y1, x0:x1].astype(float), up, 0), up, 1)
                if m.max() > 0:
                    ax.contour(m, levels=[0.5], colors=[col], linewidths=lw)
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], color=c, lw=1.4, label=l)
                       for c, l in (("#2ecc71", "STAPLE"), ("#3498db", "annotator A"),
                                    ("#f39c12", "annotator B"), ("#e74c3c", "MedSAM (oracle box)"))]
            ax.legend(handles=handles, fontsize=7, loc="lower left", framealpha=0.9)
            ax.set_axis_off()
            p = FIG / f"overlay_{kid}_z{z:03d}.png"
            fig.savefig(p, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print("overlay:", p, flush=True)
            done.add(key)
    missing = [k for k, z, _ in picks if (k, z) not in done]
    if missing:
        print("WARN missing overlay slices:", missing)


if __name__ == "__main__":
    main()
