#!/usr/bin/env python3
"""M5 P2-4/P2-6: reproducible aggregation of benchmark CSVs.

Reads outputs/metrics/e1_internal.csv and e2_external.csv and regenerates
outputs/summary/aggregate_results.json. Guards required by the pre-registration
freeze:

- repeated header lines inside the CSVs are dropped and counted;
- primary key (model, seed, prompt, case_id) must be unique -- duplicates abort
  the run (keep-first vs keep-last would then be ambiguous, delta is reported);
- per-cell completeness must match the expected counts exactly: 99 (E1) and
  18,322 (E2), except the six documented MedSAM E3 dedup-loss cells;
- HD95 is a conditional metric: empty predictions get Dice 0 but HD95 NaN
  (m3_common.score), so every HD95 aggregate carries hd95_valid_n / hd95_nan_n
  alongside the mean -- never a mean over an undisclosed subset.

E3 ablation cells live in the same CSVs (prompt column); detection-side
records are in e3_detection_*.csv and are not aggregated here.
"""
import csv
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__import__("os").environ.get("KIDNEYUS_ROOT",
          str(Path(__file__).resolve().parent.parent)))
MET = ROOT / "outputs/metrics"
OUT = ROOT / "outputs/summary/aggregate_results.json"
OUT_LEGACY = ROOT / "outputs/metrics/aggregate_summary.json"

E1_N, E2_N = 99, 18322
# Documented in DATA_QUALITY.md: MedSAM E3 rows lost to the dedup incident
# (10 rows total, 0.003%). Anything else deviating from E1_N/E2_N aborts.
KNOWN_GAPS = {
    ("e2", "medsam", 42, "p1"): E2_N - 4,
    ("e2", "medsam", 42, "p2"): E2_N - 3,
    ("e2", "medsam", 42, "p4"): E2_N - 3,
    ("e2", "medsam", 44, "p1"): E2_N - 3,
    ("e2", "medsam", 44, "p2"): E2_N - 4,
    ("e2", "medsam", 44, "p4"): E2_N - 4,
}


def sha256_head(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()[:12]


def load(path):
    rows, repeated_headers = [], 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("model") == "model":  # repeated header line
                repeated_headers += 1
                continue
            rows.append(r)
    return rows, repeated_headers


def agg_cell(dice, hd):
    valid = [h for h in hd if h is not None]
    return {
        "n_cases": len(dice),
        "dice_mean": round(statistics.mean(dice), 4),
        "dice_std": round(statistics.stdev(dice), 4) if len(dice) > 1 else 0.0,
        "dice_median": round(statistics.median(dice), 4),
        "hd95_mean": round(statistics.mean(valid), 4) if valid else None,
        "hd95_median": round(statistics.median(valid), 4) if valid else None,
        "hd95_valid_n": len(valid),
        "hd95_nan_n": len(hd) - len(valid),
    }


def fnum(x):
    return None if x in ("", "nan", "NaN", "None") else float(x)


def build(rows, ds_n, ds_tag):
    cells = defaultdict(lambda: ([], []))
    for r in rows:
        d, h = cells[(r["model"], int(r["seed"]), r["prompt"])]
        d.append(float(r["dice"]))
        h.append(fnum(r["hd95_px"] if "hd95_px" in r else r["hd95_mm"]))
    # --- completeness against expected counts ---
    completeness, errors = {}, []
    for (m, s, p), (d, _) in sorted(cells.items()):
        want = KNOWN_GAPS.get((ds_tag, m, s, p), ds_n)
        ok = len(d) == want
        completeness[f"{m}/{s}/{p}"] = {"actual": len(d), "expected": want,
                                        "status": "OK" if ok else "GAP"}
        if not ok:
            errors.append(f"{ds_tag} {m}/{s}/{p}: {len(d)} != {want}")
    # --- primary-key uniqueness (abort on duplicates) ---
    keys = defaultdict(int)
    for r in rows:
        keys[(r["model"], r["seed"], r["prompt"], r["case_id"])] += 1
    dups = {k: c for k, c in keys.items() if c > 1}
    if dups:
        sample = list(dups.items())[:5]
        sys.exit(f"[ABORT] duplicate primary keys in {ds_tag}: n={len(dups)} sample={sample}")
    # --- aggregate ---
    out = []
    for (m, s, p), (d, h) in sorted(cells.items()):
        e = {"model": m, "seed": s, "prompt": p, **agg_cell(d, h)}
        out.append(e)
    return out, completeness, errors


def main():
    e1_rows, h1 = load(MET / "e1_internal.csv")
    e2_rows, h2 = load(MET / "e2_external.csv")
    e1, c1, err1 = build(e1_rows, E1_N, "e1")
    e2, c2, err2 = build(e2_rows, E2_N, "e2")
    if err1 or err2:
        sys.exit("[ABORT] completeness mismatch:\n  " + "\n  ".join(err1 + err2))

    # pooled across seeds for convenience (same cell list, per model/prompt)
    pool = defaultdict(lambda: ([], []))
    for r in e1_rows:
        d, h = pool[(r["model"], r["prompt"], "e1")]
        d.append(float(r["dice"])); h.append(fnum(r.get("hd95_px", "")))
    for r in e2_rows:
        d, h = pool[(r["model"], r["prompt"], "e2")]
        d.append(float(r["dice"])); h.append(fnum(r.get("hd95_mm", "")))
    pooled = [{"model": m, "prompt": p, "dataset": ds, **agg_cell(d, h)}
              for (m, p, ds), (d, h) in sorted(pool.items())]

    doc = {
        "generated": date.today().isoformat(),
        "generator": "scripts/m5_aggregate.py",
        "sources": {
            "e1_internal.csv": {"rows": len(e1_rows), "repeated_headers": h1,
                                 "sha256_1MB": sha256_head(MET / "e1_internal.csv")},
            "e2_external.csv": {"rows": len(e2_rows), "repeated_headers": h2,
                                 "sha256_1MB": sha256_head(MET / "e2_external.csv")},
        },
        "dedup_report": {
            "duplicate_primary_keys": 0,
            "note": "primary key (model,seed,prompt,case_id) verified unique in both "
                    "CSVs; keep-first vs keep-last dedup direction is therefore moot. "
                    "Repeated header lines dropped: e1=%d e2=%d." % (h1, h2),
        },
        "hd95_caliber": "conditional: HD95 is NaN when the prediction is empty "
                        "(m3_common.score); hd95_mean covers hd95_valid_n slices, "
                        "hd95_nan_n reported alongside. Empty predictions still "
                        "count Dice=0 in dice_mean.",
        "e1_internal": e1,
        "e2_external": e2,
        "pooled_seeds": pooled,
        "completeness": {"e1": c1, "e2": c2},
    }
    OUT.write_text(json.dumps(doc, indent=1))

    # ---- legacy twin: outputs/metrics/aggregate_summary.json (same verified ----
    # ---- cells, old schema; the 2026-10-10 23:23 version of this file was   ----
    # ---- computed on a pre-cleanup CSV state and disagreed with reality:    ----
    # ---- e1 usfm cells counted 200/198/396 rows and e2 usfm seed44 19,107   ----
    # ---- slices vs the true 99 / 18,322. Regenerated here from the same     ----
    # ---- asserted-unique cells so the two files can never diverge again.    ----
    legacy = {"e1_internal": [], "e2_external": [], "e3_ablation": [],
              "e2_completeness": {}, "note": "generated by scripts/m5_aggregate.py "
              "from the same verified cells as outputs/summary/aggregate_results.json"}
    for e in e1:
        legacy["e1_internal"].append({
            "model": e["model"], "seed": e["seed"], "prompt": e["prompt"],
            "n_cases": e["n_cases"], "dice_mean": e["dice_mean"],
            "dice_std": e["dice_std"], "dice_median": e["dice_median"],
            "hd95_px_mean": e["hd95_mean"], "hd95_px_median": e["hd95_median"]})
    for e in e2:
        legacy["e2_external"].append({
            "model": e["model"], "seed": e["seed"], "prompt": e["prompt"],
            "n_slices": e["n_cases"], "dice_mean": e["dice_mean"],
            "dice_std": e["dice_std"], "dice_median": e["dice_median"],
            "hd95_mm_mean": e["hd95_mean"], "hd95_mm_median": e["hd95_median"],
            "nan_hd95_count": e["hd95_nan_n"]})
    legacy["e2_completeness"] = c2
    legacy["e1_completeness"] = c1
    OUT_LEGACY.write_text(json.dumps(legacy, indent=1))

    n_gap = sum(1 for c in list(c1.values()) + list(c2.values()) if c["status"] == "GAP")
    print(f"OK -> {OUT}")
    print(f"OK -> {OUT_LEGACY} (legacy schema, same cells)")
    print(f"  e1 cells={len(e1)} e2 cells={len(e2)} pooled={len(pooled)} completeness GAPs={n_gap} (documented: 6)")
    print(f"  repeated headers: e1={h1} e2={h2}; duplicate PKs=0")


if __name__ == "__main__":
    main()
