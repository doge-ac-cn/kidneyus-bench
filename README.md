# kidneyus-bench

**A cross-cohort benchmark of segmentation foundation models for ultrasound kidney
segmentation with a fully automatic, zero-manual-annotation prompting pipeline —
plus a pre-registered, source-free test-time-adaptation extension.**

Task-specific and foundation segmentation models (U-Net, nnU-Net v2, MedSAM,
SAM+LoRA, SAMUS, USFM) are trained on the
[Open Kidney Dataset](https://github.com/rsingla92/kidneyUS) (OKD) and evaluated
out-of-the-box on the external [TRUSTED](https://doi.org/10.6084/m9.figshare.27981050)
paired 3D-US/CT cohort, normalized by the radiographer inter-observer agreement (IoD).
A YOLO detector (boxes derived automatically from the existing segmentation masks)
provides fully automatic box prompts — the deployment pipeline never touches a manual
prompt or a ground-truth mask at inference time.

**Preregistration:**

- Main study protocol (v1.4) registered before external-validation results were
  aggregated: [DOI 10.17605/OSF.IO/TX7RN](https://doi.org/10.17605/OSF.IO/TX7RN)
- Post-registration Amendment 1 (constructive arms for the external detection
  failure: training-free prior-box prompts **P5** and source-free detector
  self-training **P4-TTA**), registered before any arm computation:
  [DOI 10.17605/OSF.IO/A8MKX](https://doi.org/10.17605/OSF.IO/A8MKX)
  ([text](docs/tta_amendment_v1.md))

The experimental protocol (v1.4, amended after journal benchmarking and an
advisor methodology review) is described in
[`docs/plan_A_v1.4.md`](docs/plan_A_v1.4.md) (Chinese) — the milestone/gate
structure of the study, written before the runs.

## Status: computation complete (2026-10-13)

| Milestone | Status |
|---|---|
| M0 data acquisition & verification (both cohorts) | ✅ passed |
| M1 canonical split + preprocessing + IoD reference | ✅ passed (IoD direct A-vs-B = 0.9063; consensus-derived = 0.9478; both within the pre-registered 0.90–0.96 gate) |
| M2 training — 6 models × 3 seeds | ✅ U-Net, nnU-Net v2, MedSAM, SAM+LoRA, SAMUS (official light recipe, 43.71M trainable), USFM (official recipe: AdamW 3e-4, layer-decay 0.65, 400 ep) + YOLOv8s detector (3 seeds) |
| M3 external validation (E2) + prompt-strategy ablation (E3) | ✅ 45/45 cells both cohorts, reconciled |
| M4 reverse direction: nnU-Net trained on TRUSTED, official 5-fold CV, 2d + 3d | ✅ 10/10 folds |
| M5 frozen statistics (Friedman+Holm, Wilcoxon, 39/93-cluster bootstrap n=1000, dual-caliber CI, LMM cross-check) | ✅ [`outputs/summary/stats_report.csv`](outputs/summary/stats_report.csv) |
| Amendment 1 arms (P5 + P4-TTA, 2 rounds × 3 replicates) | ✅ executed under the frozen criteria; verdict reported as measured |

## Headline results

External validation (TRUSTED, 18,322 valid slices / 59 kidneys / 39 patients;
mean Dice over 3 seeds; headline prompt in parentheses — p3 = oracle box,
p4 = fully automatic detector box):

All values are mean Dice over 3 seeds (slice-pooled caliber of
`aggregate_results.json`); retention = headline E2 / E1.

| Model | OKD test (E1) | TRUSTED ext. (E2, headline) | TRUSTED ext. (E2, oracle box) | Retention |
|---|---|---|---|---|
| MedSAM | 0.960 (p3) | 0.442 (p4) | **0.884** | 0.461 |
| SAM+LoRA | 0.956 (p3) | 0.440 (p4) | 0.872 | 0.460 |
| SAMUS | 0.903 (p3) | 0.418 (p4) | 0.762 | 0.462 |
| nnU-Net v2 | 0.933 (—) | **0.624** (—) | — | **0.669** |
| USFM | 0.862 (—) | 0.471 (—) | — | 0.546 |
| U-Net | 0.676 (—) | 0.289 (—) | — | 0.428 |

IoD reference (direct A-vs-B) = 0.9063; oracle-box MedSAM reaches 0.884 = 97.6% of IoD.

**Key findings**

1. **Cross-cohort domain gap** — every model degrades sharply on TRUSTED
   (best prompt-free retention 0.676); per the frozen §7 power statement, CIs
   are reported and overlapping CIs are described as "no difference detected".
2. **Prompt sensitivity under domain shift** — with an oracle box the
   foundation models reach 0.88 externally (0.44 with automatic detection
   boxes); the P3–P4 gap (0.33–0.44) localizes the deployed-pipeline failure
   to detection, not segmentation.
3. **Detection is the binding constraint** — the registered seed-43 YOLO
   detector finds no box on 48.5% of external slices; the failure is
   threshold-insensitive (AP@0.5 = 0.553, detected-box IoU median 0.81–0.82).
   Full metrics in [`outputs/summary/`](outputs/summary/).
4. **Failure anatomy** — 5.45% (999/18,322) of external scored slices are annotation-range-edge
   slices where every model scores ≈ 0 (pole cross-sections); boundary-
   discontinuity slices (18.9%) halve model Dice (0.326 vs 0.688).
5. **Amendment 1 verdict (registered, NEGATIVE — reported as measured)** —
   source-free detector self-training (P4-TTA) recovers coverage
   (no-box 48.5% → 5.8%) and lifts strict Dice to 0.69, but the frozen
   adoption criteria fail: conditional Dice 0.733/0.730/0.636 < 0.75
   (all three models required) and in-domain mAP regression 0.026 > 0.02.
   The P5 diagnostic arm (nnU-Net prior boxes → MedSAM, coverage 1.00,
   box-IoU 0.677, Dice 0.594) also reads diagnostic/negative, pinpointing
   box-localization quality — not coverage — as the remaining bottleneck.
   Complete Round-1/Round-2 results are in
   [`outputs/summary/tta_r2_verdict.json`](outputs/summary/tta_r2_verdict.json)
   and the RUN_REPORT.
6. **Reverse direction (E4)** — nnU-Net trained on TRUSTED, evaluated on its
   own official 5-fold holdouts (3D volume Dice, 59 kidneys):
   2d 0.7788 ± 0.1023 (HD95 70.9 mm), 3d 0.7776 ± 0.0938 (HD95 30.5 mm),
   cross-validated against nnU-Net's own evaluator.

## Repository layout

```
scripts/                  runbook (numbered by milestone, same names as the plan)
outputs/                  privacy-safe results: RUN_REPORT, aggregated metrics, env snapshot
docs/                     pre-registered protocol; Amendment 1 text; TRUSTED DUA copy
```

Per-case outputs (predictions, per-slice metrics, the patient-keyed split lists)
stay out of this repository on purpose; the canonical split is seed-reproducible
from scripts/m1_split_okd.py (seed 42) once the dataset is obtained — see *Privacy* below.

Aggregated, privacy-safe artifacts under `outputs/summary/` include:
`aggregate_results.json` (45 cells × both cohorts, regenerable via
`scripts/m5_aggregate.py`), `stats_report.csv` + `stats_detail.json`
(frozen §7 statistics), `e4_reverse_cv.csv` (per-kidney reverse CV),
`tta_r{1,2}_detection.json` / `tta_r2_downstream.json` / `tta_r2_verdict.json`
(Amendment 1), `p5_priorbox.json`, `pole_stratification.json`,
`box_iou_vs_oracle.json`, `nnunet_rater_aggregate.json`
(per-rater scoring closing the consensus-optimism objection),
`amendment_numbers_check.txt` (every number printed in the amendment,
mechanically re-derived).

## Data access

Both cohorts are public but **must be obtained through their own channels**; we do
not redistribute any imaging data or patient-level identifiers:

- **OKD** — register at the [kidneyUS project page](https://github.com/rsingla92/kidneyUS);
  the author emails a OneDrive share link. Required citation:
  *Singla, R., Ringstrom, C., Hu, G., Lessoway, V., Reid, J., Nguan, C., &
  Rohling, R. (2023, October). The open kidney ultrasound data set. In
  International Workshop on Advances in Simplifying Medical Ultrasound (ASMIC)
  (pp. 155–164). Cham: Springer Nature Switzerland.*
  (the arXiv preprint arXiv:2206.06657 may be cited in parallel) `scripts/download_okd.sh` consumes that
  link via environment variables (`OKD_ONEDRIVE_CID`, `OKD_DATA_SHARE`,
  `OKD_WEIGHTS_SHARE`) and never stores it.
  Facts recorded at verification: 534 PNGs distributed; 486 cases carry non-empty
  kidney-capsule masks from both annotators (the cohort used here); no
  native/transplant field exists in the distribution, so the pre-registered subgroup
  analysis was dropped (plan §10 rule 3) and the patient-level split is stratified by
  view type instead.
- **TRUSTED** — direct download from
  [figshare (10.6084/m9.figshare.27981050)](https://springernature.figshare.com/articles/dataset/TRUSTED_The_Paired_3D_Ultrasound_and_CT_Human_Data_for_Kidney_Segmentation_and_Registration_Research/27981050),
  CC BY 4.0; `scripts/download_trusted.sh`.
  The authors request a signed data-use agreement by email
  (`docs/trusted_data_use_agreement_archive.txt`).
  Facts recorded at verification: 59 US volumes (single kidney each) from 39 patients;
  Annotator1/Annotator2 + STAPLE consensus masks; 7 landmarks per kidney; the
  README inside the archive carries the official 5-fold patient split used for the
  reverse-direction experiment.

## Environment

- Python 3.10, PyTorch 2.7.1+cu126 (tested on 3× V100-16GB; fp16 AMP — Volta has no BF16)
- nnunetv2 2.8.1, MONAI 1.4.0, SimpleITK 2.5.6, statsmodels 0.15.0,
  ultralytics 8.3.108, segment-anything 1.0
- `outputs/requirements.lock` is the full snapshot

Reproduce:

```bash
# 1. environment (isolated conda env recommended)
conda create -n kidneyus2 python=3.10 -y && conda activate kidneyus2
pip install torch==2.7.1 torchvision==0.22.1
pip install nnunetv2 monai==1.4.0 SimpleITK pydicom scipy statsmodels einops \
            scikit-image pycocotools opencv-python-headless ultralytics==8.3.108 \
            openpyxl batchgenerators segment-anything
# (if your network cannot reach huggingface.co: export HF_ENDPOINT=https://hf-mirror.com)

# 2. data (see "Data access" above)
bash scripts/download_weights.sh
bash scripts/download_trusted.sh
bash scripts/download_okd.sh          # after exporting the OKD share variables
git clone --depth 1 https://github.com/rsingla92/kidneyUS external_kidneyUS   # labels/masks
unzip data/OKD_zips...                # place images under data/raw/OKD/kidneyUS_images_14_june_2022
unzip data/raw/TRUSTED.zip -d data/raw/TRUSTED/

# 3. milestones
python scripts/m0_verify_okd.py
python scripts/m0_verify_trusted.py data/raw/TRUSTED/TRUSTED_dataset_for_nsd
python scripts/m0_sanity_medsam.py
python scripts/m1_split_okd.py                       # seed-42 canonical split
python scripts/m1_trusted_slices_iod.py              # IoD reference (gate: 0.90–0.96)
bash scripts/m2_train_internal.sh                    # D1 + U-Net + nnU-Net, 3 seeds
# ... foundation-model fine-tuning and evaluation runbooks follow the same
# m*_*.sh / m*_*.py numbering; aggregation and frozen statistics:
python scripts/m5_aggregate.py                       # table reconciliation + assertions
python scripts/m5_stats_and_figures.py               # frozen section-7 statistics
python scripts/m5_e4_aggregate_fast.py               # reverse-direction 5-fold CV
```

Gotchas recorded while running (see `outputs/RUN_REPORT.md` for the full list):

- TRUSTED masks are **float STAPLE probability maps** — binarize at 0.5, not `>0`
- nnunetv2 2.8.x uses lowercase env vars (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`)
- **`nnUNetv2_train` uses positional arguments** (`DATASET CONFIG FOLD`),
  `nnUNetv2_predict` uses flags — mixing them fails with exit 2
- without `--c`, nnU-Net **retrains from scratch even if a checkpoint exists** —
  never run two trainers on the same fold
- the end-of-training validation needs `nnUNet_preprocessed/<dataset>/gt_segmentations/`
  (original-geometry GT copies); if the folder is missing, predictions are still
  written but metric computation fails — copy `labelsTr/*.nii.gz` there
- on hosts with a populated `~/.local/lib/python3.x/site-packages`, set `PYTHONNOUSERSITE=1`
  everywhere (including before `pip install`)
- PyTorch ≥ 2.8 / cu13x wheels drop Volta (sm_70) kernels — pin `torch==2.7.1+cu126` on V100s
- scripts resolve all paths from `KIDNEYUS_ROOT` or their own location — no
  absolute paths are stored anywhere

## Privacy statement

This repository contains **no patient imaging data and no patient-level identifiers**:

- no images, volumes, or masks from either dataset;
- no patient keys (e.g. OKD `IM-xxxx` codes, TRUSTED `xxxL/R` kidney IDs) — per-case
  tables, the split lists, and per-image results are intentionally kept out; the
  aggregate tables and the TRUSTED-internal kidney codes (e.g. `704L`, as named by
  the dataset itself) are the finest granularity present;
- no local filesystem paths; scripts resolve all paths from `KIDNEYUS_ROOT` or
  their own location;
- the OKD download script stores the (per-user) OneDrive share credentials only as
  environment variables.

Per-case artifacts and the patient-keyed canonical split are reproducible from the
seeded scripts once the cohorts are obtained through their official channels.

## License

Code: MIT (see `LICENSE`). The datasets remain under their own terms
(OKD: CC BY-NC-SA via registration; TRUSTED: CC BY 4.0 + signed DUA).
