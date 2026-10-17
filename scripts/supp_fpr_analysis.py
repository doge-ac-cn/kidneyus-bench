#!/usr/bin/env python3
"""Detector cross-rater overlap analysis (reads supp_detector_fpr_sets.json).

Computes exact and tolerance-dependent Jaccard overlaps of the discontinuity
flags across the three annotation sequences (STAPLE consensus / Annotator1 /
Annotator2), with analytic chance baselines and a within-kidney permutation
null. Output: outputs/summary/supp_detector_fpr_final.json.
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

d = json.loads(open("outputs/summary/supp_detector_fpr_sets.json").read())
S = {(k, z) for k, z in d["staple"]}
A = {(k, z) for k, z in d["A"]}
B = {(k, z) for k, z in d["B"]}

sl = pd.read_csv("outputs/trusted_us_slices.csv")
pool = defaultdict(set)
for k, z in zip(sl["kidney_id"], sl["slice_idx"]):
    pool[k].add(int(z))
N_pairs = sum(len(v) for v in pool.values())


def jac_tol(X, Y, tol):
    bykid = defaultdict(set)
    for k, z in Y:
        bykid[k].add(z)
    hitX = sum(1 for k, z in X if any(abs(zz - z) <= tol for zz in bykid.get(k, ())))
    bykid2 = defaultdict(set)
    for k, z in X:
        bykid2[k].add(z)
    hitY = sum(1 for k, z in Y if any(abs(zz - z) <= tol for zz in bykid2.get(k, ())))
    return (hitX + hitY) / (len(X) + len(Y))


def chance_jac(nX, nY, N, tol=0):
    ei = nX * nY / N * (2 * tol + 1) if tol else nX * nY / N
    return ei / (nX + nY - ei)


def main():
    res = {"set_sizes": {"staple": len(S), "A": len(A), "B": len(B)},
           "flag_rates": {"staple": round(len(S) / N_pairs, 4),
                          "A": round(len(A) / N_pairs, 4),
                          "B": round(len(B) / N_pairs, 4)},
           "documented_rater_flag_expectation": round(1 / 17, 4)}
    for name, X, Y in (("staple_vs_A", S, A), ("staple_vs_B", S, B), ("A_vs_B", A, B)):
        row = {f"J_tol{t}": round(jac_tol(X, Y, t), 3) for t in (0, 1, 2)}
        row.update({f"J_tol{t}_chance": round(chance_jac(len(X), len(Y), N_pairs, t), 3)
                    for t in (0, 1, 2)})
        res[name] = row
    rng = np.random.default_rng(42)
    ak = defaultdict(list)
    for k, z in A:
        ak[k].append(z)
    vals = []
    for _ in range(200):
        perm = set()
        for k, zs in ak.items():
            p_ = sorted(pool[k])
            pick = rng.choice(p_, size=min(len(zs), len(p_)), replace=False)
            perm |= {(k, int(z)) for z in pick}
        vals.append(jac_tol(S, perm, 0))
    res["permutation_null_staple_vs_A_exact"] = {
        "mean": round(float(np.mean(vals)), 3),
        "p95": round(float(np.percentile(vals, 95)), 3)}
    open("outputs/summary/supp_detector_fpr_final.json", "w").write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
