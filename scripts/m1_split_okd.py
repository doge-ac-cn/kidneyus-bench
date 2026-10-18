#!/usr/bin/env python3
"""M1: canonical OKD patient-level 60/20/20 split (seed 42) -> outputs/splits.json.

Plan §4: patient-level, no cross-set leakage, stratified. The intended stratum
(native vs transplant) is absent from the distribution (plan §10 rule 3), so we
stratify on view type (Transverse/Longitudinal/Other) instead and document it.
The split is generated ONCE and is canonical for every model and seed.
"""
import csv
import json
import os
import random
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_CSV = os.path.join(ROOT, "outputs/okd_cohort.csv")
OUT = os.path.join(ROOT, "outputs/splits.json")

SEED = 42
RATIOS = (0.6, 0.2, 0.2)


def main():
    rows = list(csv.DictReader(open(IN_CSV)))
    by_patient = defaultdict(list)
    for r in rows:
        by_patient[r["patient_key"]].append(r)

    # stratum a patient belongs to = majority view among its images
    strata = defaultdict(list)
    for pk, recs in by_patient.items():
        views = defaultdict(int)
        for r in recs:
            views[r["view"]] += 1
        stratum = max(views.items(), key=lambda kv: kv[1])[0]
        strata[stratum].append(pk)

    rng = random.Random(SEED)
    split = {"train": [], "val": [], "test": []}
    for stratum in sorted(strata):
        pks = sorted(strata[stratum])
        rng.shuffle(pks)
        n = len(pks)
        n_tr = round(n * RATIOS[0])
        n_va = round(n * RATIOS[1])
        split["train"] += pks[:n_tr]
        split["val"] += pks[n_tr:n_tr + n_va]
        split["test"] += pks[n_tr + n_va:]

    # resolve to filenames
    out = {
        "seed": SEED,
        "granularity": "patient_key",
        "stratification": "view_type (native/transplant unavailable, plan §10 rule 3)",
        "counts": {},
        "train": [], "val": [], "test": [],
    }
    for k in ("train", "val", "test"):
        fns = sorted(r["filename"] for r in rows if r["patient_key"] in set(split[k]))
        out[k] = fns
        out["counts"][k] = {"patients": len(set(split[k])), "images": len(fns)}

    # leakage assertions: image-level AND patient-level (full IM-XXXX-YYYY key).
    # NOTE 2026-10-09 dHash adjudication: images sharing only the IM-XXXX prefix
    # but differing in the full key have min pairwise dHash distance 45/256
    # (near-duplicate threshold <=10; random cross-patient median ~94) -> the
    # full key is the case-unique identifier; prefix sharing is a numbering
    # coincidence, not same-patient. See RUN_REPORT review section.
    tr, va, te = (set(out[k]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te), "image-level leakage!"
    assert len(tr | va | te) == len(rows)
    pk_of = {r["filename"]: r["patient_key"] for r in rows}
    tr_p = {pk_of[f] for f in tr}
    va_p = {pk_of[f] for f in va}
    te_p = {pk_of[f] for f in te}
    assert not (tr_p & va_p) and not (tr_p & te_p) and not (va_p & te_p), "patient-level leakage!"

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["counts"], indent=2))


if __name__ == "__main__":
    main()
