Post-Registration Amendment 1 to OSF registration 10.17605/OSF.IO/TX7RN
Constructive arms for external detection-transfer failure: training-free
prior-box prompts (P5) and source-free detector self-training (P4-TTA)

Timestamp: [locked at OSF submission — before any Arm computation]

1. STATUS AND MOTIVATION

The registered benchmark pre-specified scenario B (generalization-warning
narrative) if normalized external Dice fell below 0.75. The completed
external evaluation (TRUSTED; 59 kidneys / 39 patients; 18,322 valid
slices) yields normalized Dice 0.624 / 0.9063 = 0.69 for the best
prompt-free model: scenario B is triggered. Failure attribution localizes
the failure of the fully-automatic prompt pathway (P4): the registered
seed-43 YOLOv8s detector produces no box on 48.5% of external slices
(detection rate 51.5% at the registered operating point, conf >= 0.25);
the failure is threshold-insensitive (lowering tau to 0.10 adds fewer than
6 percentage points of coverage and changes downstream strict Dice by only
+3.4 to +3.7 points across the three prompted models). With a
reference-quality box (P3, oracle), fine-tuned foundation models reach
Dice 0.884 — 97.6% of the direct annotator-agreement reference (IoD
0.9063) — and detector boxes on detected slices are of good quality (IoU
vs oracle, median 0.81–0.82). The binding constraint of the deployed
pipeline is therefore detection coverage, not segmentation quality and not
box-localization quality.

This amendment pre-specifies two constructive arms, registered before any
new computation, testing whether detection coverage on the external domain
can be recovered without any manual annotation. This is a registered
exploratory extension motivated by completed analyses: the triggering
statistics above existed before this timestamp; no outcome data from the
new arms exist at this timestamp; the amendment's purpose is to remove
post-hoc flexibility from the new arms' execution and interpretation.

2. SCOPE AND FRAMING

Both arms operate under the registered zero-manual-annotation constraint:
no human boxes, points, or masks are created; TRUSTED ground-truth
annotations are never accessed during arm construction or adaptation and
enter only the registered evaluation pipeline. Both arms are transductive
(label-free test-time adaptation): adaptation/derivation uses the
unlabeled external-domain images themselves and evaluation is on the same
slice set. Consequently, these arms make no claim about generalization to
a third cohort and are framed as proof-of-concept remedies for
same-deployment-domain operation. The registered P4 remains the headline
result unchanged; the registered six-model statistical family (models,
seeds, Friedman/Holm plan) is unchanged; both arms are disclosed in the
manuscript's post-registration deviations section with this timestamp.

3. ARM P5 — TRAINING-FREE PRIOR-BOX PROMPTS (DIAGNOSTIC)

Box source: binarized predictions of the registered nnU-Net v2 (2d) on
TRUSTED. Seed selection: the seed with the median internal-validation
Dice, computed from registered internal validation results only, blind to
external results 42. Box derivation: per slice,
largest-connected-component tight bounding box of the binarized prediction
(registered box_from_mask utility, pad = 0), native grid; slices with an
empty prediction yield no box and score Dice = 0 per the registered
empty-prompt rule.

Evaluation: MedSAM only (registered median-internal-val seed), single
pass, all 18,322 slices, registered scorer; report coverage, box IoU vs
oracle, and strict Dice. Role: diagnostic reference bounding the benefit
of a better prompt source without any detector training. Arm P5 runs
regardless of Arm P4-TTA outcomes.

Pre-specified reporting criterion (diagnostic, not a headline arm):
reported as a "viable alternative prompt source" if and only if coverage
>= 75% AND box-IoU-vs-oracle median >= 0.70 AND strict Dice >= 0.55;
otherwise reported as diagnostic/negative.

4. ARM P4-TTA — SOURCE-FREE DETECTOR SELF-TRAINING

4.1 Initialization. The registered seed-43 YOLOv8s detector; architecture,
imgsz = 640, and all registered detection settings unchanged.

4.2 Pseudo-labels (each round). Detections of the current detector on all
18,322 external slices; keep the single highest-confidence box per slice
if conf >= 0.50; slices without any detection are EXCLUDED from training
(neither positive nor negative — this avoids reinforcing absence).

4.3 Training (each round). Fine-tune from current weights: 15 epochs,
lr0 = 1e-3 (10x below the ultralytics default), all other ultralytics
8.3.108 hyperparameters at defaults, plus the registered geometric/color
augmentation, plus the following domain-specific augmentations (applied
with per-image probability 0.8; one or more sampled uniformly):
  (a) console-text overlay: 1-4 synthetic text lines (digits/latin),
      font height 10-40 px, opacity 0.3-0.9, uniform placement;
  (b) fan-edge vignetting: random sector-boundary darkening up to 40%;
  (c) gain/gamma jitter: gamma in [0.6, 1.6]; contrast +/- 30%;
  (d) vertical shadow gradient, strength <= 0.4;
  (e) multiplicative speckle noise, sigma <= 0.15;
  (f) random cutout: 1-3 rectangles, each <= 10% of image area.
Two rounds total. Each round comprises three replicates with data-order
seeds fixed at 42/43/44, all fine-tuned from the same current weights
(section 4.1); every replicate's detector checkpoint is archived and every
replicate's results are reported. Round-level criteria (section 4.4) are
evaluated on the per-criterion mean across a round's three replicates,
with per-replicate values reported alongside.

4.4 Sole decision rule (Round 2). Round 1 is process monitoring only; the
adoption decision is made exclusively on Round 2 outcomes. Adopt as a
secondary arm "P4-TTA" if and only if, on Round 2, ALL of:
  (i)   external no-box rate < 25%;
  (ii)  conditional-on-detection Dice >= 0.75;
  (iii) strict P4-TTA Dice >= 0.55;
  (iv)  in-domain non-regression: OKD test detection rate (conf >= 0.25)
        does not decrease by more than 2 percentage points, and OKD test
        mAP@0.5 does not decrease by more than 0.02, relative to the
        registered detector.
Criteria (ii) and (iii) are evaluated separately for each of the three
prompted models (MedSAM, SAM+LoRA, SAMUS); adoption requires them to hold
for ALL three models. Per-model, per-criterion values are means across the
round's three replicates (section 4.3).
Exploratory partial band (reporting only; no main-table placement; no
further rounds): no-box rate in [25%, 35%] AND conditional Dice >= 0.80.
Any other outcome: negative result, reported as such. No additional
rounds, thresholds, or configuration changes under any outcome.

4.5 Evaluation. Identical to the registered pipeline: same 18,322 slices,
same scorer, same STAPLE ground truth (binarized at 0.5), empty-box ->
Dice = 0; uncertainty by patient-level bootstrap (39 clusters); pairwise
comparisons vs registered P4 are labeled secondary/exploratory and lie
outside the registered Friedman/Holm family.

5. GOVERNANCE

The manuscript's post-registration deviations section will disclose: the
addition date, the triggering analysis (external failure attribution), the
timestamp of this amendment, and complete Round-1/Round-2 results for both
arms. Nothing in this amendment alters the registered headline, the
registered six-model family, the pre-specified statistics, or the
scenario-B primary narrative. Artifacts are pushed within 24 hours per the
registered cadence; RUN_REPORT logs each round. If Arm P5 meets its
reporting criterion, its relation to Arm P4-TTA (prompt-source bound vs
trained adaptation) is discussed explicitly; neither arm replaces the
registered P4 as the headline.

6. COMPLIANCE

No TRUSTED ground-truth annotation is read by any arm-construction or
adaptation code (images only; ground truth enters only the registered
scorer). No patient-level data or identifiers enter the repository. All
results are reported regardless of outcome.
