# Clustering audit (Supplementary analysis (clustering audit))

All p-value-bearing comparisons are computed on patient-level (or kidney-level) cluster units; detection-, threshold- and stratified descriptive analyses carry no p-values and are labeled exploratory.

| Comparison | Source | Unit & clustering | Status | Action |
|---|---|---|---|---|
| E2 Friedman 6-model family | stats_report.csv | patient-39 seed-mean vectors | clustered | OK |
| E2 Wilcoxon pairwise (15 pairs, Holm) | stats_report.csv | patient-39 seed-mean vectors | clustered | OK |
| E1 Friedman 6-model family | stats_report.csv | patient-93 seed-mean vectors | clustered | OK |
| E1 CIs (6 models) | stats_report.csv | patient-93 bootstrap | clustered | OK |
| E2 CIs dual caliber (6 models) | stats_report.csv | patient-39 + kidney-59 bootstrap | clustered | OK |
| seed direction consistency (descriptive, no p) | stats_report.csv | per-seed medians | descriptive | OK |
| Required contrast MedSAM vs nnU-Net | stats_report.csv | patient-39 | clustered | OK |
| Prompt ranking P4/P1/P2 per model | stats_report.csv | patient-39 | clustered | OK |
| P3 vs P4 prompt gap per model | stats_report.csv | patient-39 | clustered | OK |
| B3a normalized/joint CIs | joint_bootstrap_cis.json | patient-39 joint bootstrap | clustered | OK |
| B3b TTA criteria CIs | tta_criteria_bootstrap.json | patient-39 bootstrap | clustered | OK |
| R0 vs R2 paired diffs (supplementary) | supp_bootstrap_cis.json | patient-39 paired bootstrap | clustered | OK |
| domain degradation CIs (supplementary) | supp_bootstrap_cis.json | patient-39 paired bootstrap | clustered | OK |
| CT correlation r CI (supplementary) | supp_bootstrap_cis.json | patient-39 bootstrap | clustered | OK |
| IoD CIs (supplementary) | supp_bootstrap_cis.json | patient-39 / kidney-59 | clustered | OK |
| P4 conf sensitivity (tau sweep) | p4_conf_sensitivity.json | full-population descriptive | descriptive, no p | OK |
| §4.4 criteria (frozen verdict) | tta_r2_verdict.json | replicate means + bootstrap CI | clustered (supplementary) | OK |
| pole vs rest stratified means | pole_stratification*.json | descriptive means only | exploratory (no p) | OK |
| observer-stratified tables (supplementary) | supp_stratified_tables.json | descriptive means / paired bootstrap | exploratory + clustered | OK |
| YOLO detection metrics | yolo_detection/summary.json | detection-set descriptive | exploratory (no p) | OK |
