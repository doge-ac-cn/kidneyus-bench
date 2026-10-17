# Supplementary methods materials

Sources for every number: `outputs/summary/supp_methods_facts.json`
(computed by `scripts/supp_methods_facts.py`), `outputs/summary/supp_detector_fpr.json`
(`scripts/supp_companion_cpu.py` / `scripts/supp_fpr2.py`), and the registered
runbooks listed per item.

## 1. Automatic detector (registered D1)

- Model: YOLOv8s, ultralytics 8.3.108 (version pinned at registration).
- Training: 50 epochs, imgsz 640 (library default), ultralytics default
  geometric/colour augmentation, three seeds (42/43/44). Boxes for training
  were derived automatically from the existing OKD capsule masks (registered
  zero-manual-annotation rule).
- Split: patient-level 293/94/99 (train/val/test) from `m1_split_okd.py`
  (seed 42), stratified by view type; verified leak-free by dHash
  (`outputs/RUN_REPORT.md`).
- Operating point: **conf >= 0.25** (ultralytics default), fixed at
  registration and used for every reported P4 number.
- Threshold justification: the sensitivity sweep
  (`scripts/m5_p4_conf_sensitivity.py`, `outputs/summary/p4_conf_sensitivity.json`)
  re-evaluated the full pipeline at tau = 0.10/0.15/0.25. Lowering the
  threshold adds < 6 pp of coverage and changes downstream strict Dice by only
  3.4-3.7 pp: the external failure is detection **absence**, not threshold
  placement.

## 2. P4-TTA self-training recipe (registered Amendment 1 §4.3)

- Optimizer: ultralytics defaults (SGD, momentum 0.937, weight decay 5e-4)
  with `lr0 = 1e-3` (10x below the library default).
- 15 epochs per round; loss = the standard ultralytics detection loss
  (box/cls/dfl); **no layers frozen** (full fine-tune from the current
  weights).
- Augmentation: registered defaults plus, with per-image probability 0.8, one
  or more of six domain operations sampled uniformly (console-text overlay,
  fan-edge vignetting, gamma/contrast jitter, vertical shadow gradient,
  speckle noise, cutout); parameters fixed in `scripts/m5_tta_aug.py`, exact
  ranges as registered.
- Pseudo-labels: top-1 box per slice at conf >= 0.50; slices without a
  detection are **excluded** from training (neither positive nor negative).
- 15-epoch choice: fixed at registration before any arm computation
  (`docs/tta_amendment_v1.md` §4.3) as a compute-bounded single-round budget;
  no tuning was performed on any outcome.
- Observed compute cost (per replicate, 3 shared V100-16GB, contended):
  round-1 window 14:19-23:28 (+0800, 2026-10-11) for three sequential
  train+dump replicates (including two pre-flight GPU-guard failures);
  approximately 1 h training + ~25 min detection dump per replicate in
  isolation.

## 3. Self-training slice-selection bias (quantified)

Detections at the pseudo-label threshold (conf >= 0.50) determine which
slices enter training; the excluded slices are systematically harder/smaller:

| round | kept slices | kept fraction | mean GT area kept | mean GT area excluded |
|---|---|---|---|---|
| R1 (each replicate, same set) | 8,142 / 18,322 | 44.4% | 26,187 px | 15,461 px |
| R2 s42 | 13,487 / 18,322 | 73.6% | 23,725 px | 10,471 px |
| R2 s43 | 12,623 / 18,322 | 68.9% | 24,137 px | 11,567 px |
| R2 s44 | 13,688 / 18,322 | 74.7% | 23,525 px | 10,486 px |

Interpretation: the exclusion rule concentrates training on the slices the
detector already finds (~2.2x larger cross-sections) and never teaches
negative evidence for the missed pole slices - a documented limitation of the
registered exclusion rule, quantified here. The R2 round-level criteria are
nevertheless evaluated on **all** 18,322 slices with empty-box -> Dice 0, so
the missed slices remain fully penalised.

## 4. Blind discontinuity detector (data-quality claim)

- Definition (`scripts/m5_key_slice_detect.py`): for each kidney, the
  adjacent-slice mask dissimilarity series s(z) = 1 - IoU(mask_z, mask_z+1)
  over consecutive annotated slices; a slice is flagged when s(z) is a local
  maximum and exceeds **median + 3 x MAD** of the series.
- Parameters: MAD multiplier 3 (primary); no hand-labelled key-slice reference
  exists in TRUSTED, so the detector is described as *blind* (mask-internal
  statistic only, no model, no GT comparison).
**Measured behaviour** (`supp_detector_fpr.json`, `supp_detector_fpr_final.json`,
`scripts/supp_fpr2.py`, `supp_fpr_analysis.py`):

- On either annotator's own mask sequence the detector flags **5.86% (A)** and
  **5.65% (B)** of adjacent-slice pairs - almost exactly the documented
  hand-drawn cadence (1/17 = 5.88%). On the **STAPLE consensus** map the rate
  is **18.86%**, i.e. ~3.2x the roughness of either input: the fusion +
  0.5-binarization process itself creates discontinuities where the two
  raters disagree. The 18.9% figure quoted in the manuscript is therefore a
  property of the *consensus product used as the scoring target*, not of a
  single annotator's drawing.
- **Noise floor**: under within-kidney slice-order shuffling the flag rate
  collapses to 0.03% - the criterion almost never fires on structureless
  sequences, so the observed flags are real sequence events.
- **Threshold stability**: flags at 2x / 3x / 4x MAD = 4,149 / 3,455 / 2,904
  (22.7% / 18.9% / 15.9%); the claim is not knife-edge on the multiplier.
- **Cross-rater overlap** (Jaccard, over kidneys x slices):
  A-vs-B 0.30 exact (chance 0.03; permutation null <= 0.095) rising to 0.64
  within +-1 slice and 0.79 within +-2; STAPLE-vs-rater 0.20 exact (chance
  0.05) rising to ~0.52 within +-1 slice. The flagged transitions are shared
  structure (7-10x chance at exact position, ~7x chance at +-1), with
  per-rater position offsets of one slice.
- **Reading for the manuscript**: the elevated model error on these slices
  (Dice 0.326 vs 0.688) is localized to slices where the *consensus scoring
  target* is locally ambiguous (raters disagree there, and each rater's own
  sequence is smooth where the consensus is jagged) - a property of
  consensus-derived ground truth, distinct from annotator sloppiness.
