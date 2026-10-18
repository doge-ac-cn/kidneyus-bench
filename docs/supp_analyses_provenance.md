# Supplementary analyses — number provenance

**Registered caliber**: the registration fixes the inference/statistics seed at **42** (`REGISTRATION_DOCUMENT_EN.md` §3; `plan_A_public_v1.md` line 33). All primary `supp_*` files are seed-42 caliber. A complete seed-43 recomputation is archived as a sensitivity set under `outputs/summary/seed43_sensitivity/` (verdicts unchanged in direction and magnitude). Three-seed pooled main-table rows are unaffected.

Every number in the supplementary analyses traces to a script and an output
file. All scripts resolve paths from `KIDNEYUS_ROOT` or their own location.

| # | Analysis | Numbers | Produced by | Output |
|---|---|---|---|---|
| ① | Observer-stratified tables (pole / discontinuity / rest; model vs consensus / A / B) | A-vs-B by layer; per-config layer means | `scripts/supp_megapass.py` (per-slice dice vs GT/A/B), `scripts/supp_nnunet_score.py` (nnU-Net nii leg), `scripts/supp_analysis.py` | `outputs/summary/supp_stratified_tables.json` / `.md`; per-slice inputs `outputs/metrics/rev_*.csv` (gitignored) |
| ② | US-vs-CT volumetric error | bias, 95% CI, LoA, n, pairing rule | `scripts/supp_volume_pairing.py`, `scripts/supp_volume_3seed.py` | `outputs/summary/volume_bland_altman.json`, `supp_volume_pairing.md` |
| ③ | Paired bootstrap CIs | IoD CIs, R0-vs-R2 paired diffs, domain legs, CT r | `scripts/supp_bootstrap_suite.py`; joint legs originally in `scripts/m5_joint_bootstrap.py` | `outputs/summary/supp_bootstrap_cis.json` |
| ④ | Boundary metrics (HD95 / NSD@1mm) | per-config HD95 (valid-n / NaN-n) and NSD | `scripts/supp_megapass.py`, `scripts/supp_nnunet_score.py`, `scripts/supp_analysis.py` | `outputs/summary/supp_boundary_metrics.json` / `.md` |
| ⑤ | Clustering audit | full comparison list with units | `scripts/supp_cluster_audit.py` | `outputs/summary/supp_cluster_audit.md` |
| ⑥ | S-Table 4 per-replicate | strict / conditional / coverage per replicate | `scripts/supp_companion_cpu.py` (stable4) | `outputs/summary/supp_table_s4_per_replicate.csv` / `.md` |
| ⑦ | Methods materials (detector, TTA recipe, selection bias, FPR) | flags, ratios, costs | `scripts/supp_methods_facts.py`, `scripts/supp_fpr2.py`, `scripts/supp_fpr_analysis.py` | `outputs/summary/supp_methods.md`, `supp_methods_facts.json`, `supp_detector_fpr.json`, `supp_detector_fpr_final.json` |
| ⑧ | Sensitivity arms | LMM seed variance; SAMUS seed-43; R2 alternative initialization | `scripts/supp_companion_cpu.py` (LMM), `scripts/m5_tta_round.py --init-registered` + chain | `outputs/summary/supp_lmm_random_seed.json`, `supp_samus_seed_headline.json`, `supp_r2alt.json` |

## Headline numbers and their exact sources

- IoD direct A-vs-B 0.9063 [0.8943, 0.9164] (patient-39 cluster bootstrap):
  `supp_bootstrap_cis.json` → `iod_direct.patient39`; input
  `outputs/metrics/iod_ab_per_slice.csv` (gitignored, produced by
  `scripts/supp_pole_annotator_consistency.py` as a by-product... see note).
- Domain degradation: 2D slice −0.0496 [−0.0810, −0.0182];
  3D volume (3-seed forward) −0.0555 [−0.0923, −0.0208]:
  `supp_bootstrap_cis.json` → `domain_2d_slice_paired`, `domain_3d_volume_paired`.
- CT volume vs US Dice r = −0.0804 [−0.33, 0.23] (patient bootstrap):
  `supp_bootstrap_cis.json` → `ct_volume_vs_us_dice_r`.
- R0 (registered P4) vs R2 (paired two-round) differences:
  strict +0.2482 [0.2215, 0.2788]; conditional on the fixed R0-detection
  subset +0.0095 [0.0045, 0.0154]: `supp_bootstrap_cis.json` →
  `r0_vs_r2_strict_paired`, `r0_vs_r2_conditional_fixedsubset_paired`.
- Volume bias: nnU-Net zero-shot seed42 +147.85 [127.53, 167.84] mL;
  3-seed +123.86 [105.03, 142.62] mL; TTA-R2 +20.32 [3.99, 37.63] mL;
  US GT −26.98 [−38.10, −14.15] mL: `volume_bland_altman.json`.
- Observer stratification (pole): A-vs-B Dice 0.6182 / IoU 0.4858;
  MedSAM P3 0.7046 (vs consensus) / 0.5701 (vs A) / 0.5817 (vs B):
  `supp_stratified_tables.json`.
- Discontinuity detector: rater-own-mask flag rates 5.86% (A) / 5.65% (B)
  vs STAPLE consensus 18.86%; shuffled-order floor 0.03%; cross-rater
  Jaccard 0.30 exact (chance 0.03) / 0.64 at ±1 slice:
  `supp_detector_fpr.json`, `supp_detector_fpr_final.json`.
- LMM seed random-effect variance components: sd <= 0.0137 per model;
  MixedLM converges (seed variance component 0.0087 on patient-level
  vectors): `supp_lmm_random_seed.json`.
- R2 initialization sensitivity: single-start round-2 (three replicates
  from the registered detector) gives no-box 19.15%, strict 0.6323,
  conditional 0.7825 (MedSAM downstream caliber) vs the paired design's
  5.83% / 0.680 / 0.743: `supp_r2alt.json`.
- SAMUS seed-43 sensitivity under the TTA-R2 boxes: strict 0.6386,
  conditional 0.6963, coverage 0.917 (vs seed-42 0.5945 / 0.651 / 0.94):
  `outputs/metrics/rev_samus43_tta_r2s43.csv` (gitignored) aggregated in
  `supp_boundary_metrics.md` row `samus43_tta_r2s43`... note: the boundary
  table lists it; strict/conditional from the same file.

## Notes

- Per-slice metric tables (`outputs/metrics/rev_*.csv`, `iod_ab_per_slice.csv`,
  `supp_detector_fpr_sets.json`) carry dataset-internal slice keys and are
  gitignored; only aggregate outputs are committed.
- `iod_ab_per_slice.csv` is produced by `scripts/supp_pole_annotator_consistency.py`
  (per-slice A-vs-B Dice by-product), consumed by `supp_bootstrap_suite.py`.
