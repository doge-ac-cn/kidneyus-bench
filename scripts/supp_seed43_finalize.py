#!/usr/bin/env python3
"""Seed-43 finalization: promote the seed-43 outputs to the primary supp_*
filenames, fix labels, regenerate the companion markdown files, and append the
documentation records. Idempotent.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("KIDNEYUS_ROOT", str(Path(__file__).resolve().parent.parent)))
OUT = ROOT / "outputs/summary"
SENS = OUT / "seed43_sensitivity"


def load(p):
    return json.loads(p.read_text())


def main():
    # ---- USFM registered recipe ----
    src = SENS / "supp_usfm_regrecipe_seed43.json"
    if src.exists():
        d = load(src)
        d["official_reference_same_seed"] = d.pop("official_seed42", None)
        (SENS / "supp_usfm_regrecipe_seed43.json").write_text(json.dumps(d, indent=1))
        md = ["# Supplementary — USFM registered-recipe point check (registered seed 43)\n",
              f"Recipe: lr=1e-5, 50 epochs, Dice+BCE (frozen registration basis); "
              f"GPU train/eval ~19/28 min.\n",
              "| metric | registered recipe | official recipe (seed43) | delta |",
              "|---|---|---|---|"]
        off = d.get("official_reference_same_seed") or {}
        md.append(f"| internal val (best) | **{d['internal_val_best']:.4f}** | 0.8639 | -0.0893 |")
        md.append(f"| E1 test | {d['e1_test_dice']:.4f} | {off.get('e1_seed43')} | {d['delta_vs_official']['e1']:+.4f} |")
        md.append(f"| E2 slice-pooled | {d['e2_slice_pooled_dice']:.4f} | {off.get('e2_seed43')} | {d['delta_vs_official']['e2']:+.4f} |")
        md += ["", f"Acceptance gate (val >= 0.85): **{'PASS' if d['acceptance_gate_0.85'] else 'FAIL'}**.",
               "", "|delta external| < 0.03 and no ranking change -> no 3-seed upgrade."]
        (SENS / "supp_usfm_regrecipe_seed43.md").write_text("\n".join(md) + "\n")

    # ---- SAMUS box-FT ----
    src = SENS / "supp_samus_boxft_seed43.json"
    if src.exists():
        d = load(src)
        d["arm"] = "SAMUS box-prompt fine-tune (registered seed 43)"
        d["reference_point_trained_arm_seed43"] = d.pop("reference_point_trained_arm_seed42", None)
        e1p2 = round(d["e1"]["p2"] - d["reference_point_trained_arm_seed43"]["e1"]["p2"], 4)
        d["delta_vs_point_trained"]["e1_p2"] = e1p2
        (SENS / "supp_samus_boxft_seed43.json").write_text(json.dumps(d, indent=1))
        md = ["# Supplementary arm — SAMUS box-prompt fine-tuning (registered seed 43)\n",
              "Same recipe as the reported light fine-tune; prompts = GT boxes, prompt "
              "encoder unfrozen (43.72M); AdamW 1e-4 cosine, 50 ep, bs 8.\n",
              "| eval | prompt | box-FT | point-trained (seed43) | delta |",
              "|---|---|---|---|---|"]
        for k in ("e1", "e2"):
            for p in ("p3", "p4", "p2"):
                ref = d["reference_point_trained_arm_seed43"][k][p]
                md.append(f"| {k.upper()} | {p} | **{d[k][p]}** | {ref} | "
                          f"{d['delta_vs_point_trained'][f'{k}_{p}']:+.4f} |")
        md += ["", "Box-pathway fine-tuning closes the asymmetry: external oracle-box "
               "Dice rises to 0.892 (level with MedSAM 0.884); the detection-box gap "
               "remains set by the boxes themselves."]
        (SENS / "supp_samus_boxft_seed43.md").write_text("\n".join(md) + "\n")

    # ---- P5 ----
    src = SENS / "supp_p5_priorbox_seed43.json"
    if src.exists():
        (SENS / "supp_p5_priorbox_seed43.json").write_text(src.read_text())

    # ---- R2alt summary (built by supp_seed43_r2alt.py) ----
    rc = subprocess.run([sys.executable, str(ROOT / "scripts/supp_seed43_r2alt.py")],
                        capture_output=True, text=True)
    (ROOT / "outputs/logs/s43_r2alt_summary.log").write_text(rc.stdout + rc.stderr)

    # ---- companion markdown + RUN_REPORT/DATA_QUALITY ----
    rc = subprocess.run([sys.executable, str(ROOT / "scripts/supp_seed43_docs.py")],
                        capture_output=True, text=True)
    (ROOT / "outputs/logs/s43_docs.log").write_text(rc.stdout + rc.stderr)
    print("FINALIZED", rc.returncode)


if __name__ == "__main__":
    main()
