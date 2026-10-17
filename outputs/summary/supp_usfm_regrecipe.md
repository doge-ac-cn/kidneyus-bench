# Supplementary item 9 — USFM registered-recipe point check (seed 42)

**Recipe** (frozen registration basis, plan_A v1.4 §A1): lr = 1e-5, 50 epochs, Dice+BCE loss — aligned with the M3 fine-tuning protocol; acceptance gate val Dice >= 0.85.

Training details: epochs 50, lr 1e-05, layer_decay 1.0, warmup 0, wd 0.01, bs 8. GPU wall time: ~19 min train (18:51-19:10, V100-16GB) + ~28 min E1+E2 eval.

| metric | registered recipe (seed42) | official recipe (seed42) | delta |
|---|---|---|---|
| internal val Dice (best) | **0.7967** | 0.8639 | -0.0672 |
| E1 test Dice (99) | 0.8220 | 0.8622 | -0.0402 |
| E2 slice-pooled Dice (18,322) | 0.4585 | 0.4687 | -0.0102 |

**Acceptance gate**: val Dice 0.7967 < 0.85 → the registered recipe does NOT reach the pre-registered acceptance gate for this model (reported as a result, not hidden).

**Decision (writing-team rule)**: |delta external| = 0.0102 < 0.03 and no ranking/verdict change (USFM remains below the SAM family and above U-Net in E1; E2 ordering unchanged) → no 3-seed upgrade; one sentence + this supplementary table.

Provenance: `scripts/m6_usfm.py --tag _regrecipe` (train) → `outputs/checkpoints/usfm_regrecipe_seed42.pt`; `scripts/supp_usfm_regrecipe_eval.py` (E1/E2, registered inference pathway) → `outputs/summary/supp_usfm_regrecipe.json`. Official-recipe reference: seed-42 E1 0.8622 / E2 0.4687 from the registered `e1_internal.csv` / `e2_external.csv`.
