#!/usr/bin/env python3
"""M5 P3-9: provenance probe - stock SAM zero-shot with the GT box on TRUSTED.

Discriminative experiment for "was the annotation model-assisted?": if the
off-the-shelf SAM (vit_b, official predictor/preprocessing, no fine-tuning)
reproduces the STAPLE consensus near-perfectly from the consensus box alone,
the masks likely trace boundaries that a generic vision model already saturates
- a marker for (semi-)automated annotation. Conversely moderate scores suggest
human-drawn conventions SAM does not infer for free.

50 slices sampled with RNG seed 42 from the evaluation list (GT-box non-empty).
Stock SamPredictor (ImageNet normalization, 1024 letterbox) - deliberately NOT
our pipeline's preprocessing, so the probe reflects factory SAM behaviour.

Aggregate stats -> outputs/figures/sam_zeroshot_provenance.json (case-free,
committable). Per-slice detail goes to the log only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from m3_common import iter_trusted, score  # noqa: E402

ROOT = Path(__import__("os").environ.get("KIDNEYUS_ROOT",
          str(Path(__file__).resolve().parent.parent)))
N_SLICES, RNG_SEED = 50, 42
WEIGHTS = ROOT / "data/weights/sam_vit_b_01ec64.pth"

gt = np.load(ROOT / "outputs/prompts/p3_oracle_trusted.npz")
case_ids, boxes = list(gt["case_ids"]), gt["boxes"]
valid = [i for i in range(len(case_ids)) if not np.isnan(boxes[i]).any()]
rng = np.random.default_rng(RNG_SEED)
pick = sorted(rng.choice(valid, size=N_SLICES, replace=False).tolist())
pick_set = set(pick)
print(f"sampling {len(pick)}/{len(case_ids)} slices (seed {RNG_SEED})", flush=True)

from segment_anything import sam_model_registry, SamPredictor  # noqa: E402

device = "cpu"  # GPU0 is stacked with M4 fold1 (3 GB free < SAM 1024 peak);
                # 50 slices on CPU is minutes - keep GPU queue untouched
sam = sam_model_registry["vit_b"](checkpoint=str(WEIGHTS)).to(device)
sam.eval()
predictor = SamPredictor(sam)

dices = []
n_done = 0
for i, (cid, img, gt_mask, spacing, _meta) in enumerate(iter_trusted()):
    if i not in pick_set:
        continue
    rgb = np.stack([img] * 3, axis=-1)
    predictor.set_image(rgb)
    m, _, _ = predictor.predict(box=boxes[i].astype(np.float32), multimask_output=False)
    pred = m[0]
    d, _h = score(pred, gt_mask, spacing)
    dices.append(float(d))
    print(f"{cid}: dice={d:.4f}", flush=True)
    n_done += 1

import statistics as st
out = {
    "n": n_done, "rng_seed": RNG_SEED,
    "dice_mean": round(st.mean(dices), 4),
    "dice_median": round(st.median(dices), 4),
    "dice_std": round(st.stdev(dices), 4),
    "frac_gt_0.9": round(sum(1 for d in dices if d > 0.9) / n_done, 4),
    "frac_lt_0.5": round(sum(1 for d in dices if d < 0.5) / n_done, 4),
    "note": "stock SAM vit_b + GT (STAPLE-derived) box, official predictor "
            "preprocessing; provenance probe, not part of the benchmark table",
}
(ROOT / "outputs/figures/sam_zeroshot_provenance.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1), flush=True)
