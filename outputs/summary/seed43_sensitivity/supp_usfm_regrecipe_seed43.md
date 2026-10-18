# Supplementary — USFM registered-recipe point check (registered seed 43)

Recipe: lr=1e-5, 50 epochs, Dice+BCE (frozen registration basis); GPU train/eval ~19/28 min.

| metric | registered recipe | official recipe (seed43) | delta |
|---|---|---|---|
| internal val (best) | **0.7746** | 0.8639 | -0.0893 |
| E1 test | 0.8048 | 0.8531 | -0.0483 |
| E2 slice-pooled | 0.4557 | 0.4543 | +0.0014 |

Acceptance gate (val >= 0.85): **FAIL**.

|delta external| < 0.03 and no ranking change -> no 3-seed upgrade.
