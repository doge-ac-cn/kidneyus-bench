# Volumetric error (Supplementary analysis (volumetric error))

US kidney -> same-patient same-laterality CT component (GT_estimated_masksCT, R = smaller-x centroid); US volume = sum of per-slice predicted area x 0.09 mm2 x 0.3 mm; level = per-kidney (n=59); CI = patient-39 cluster bootstrap; US sub-cohort = 39 of the 48 CT patients

| arm | n pairs | bias mL | 95% CI (patient) | 95% LoA |
|---|---|---|---|---|
| nnunet_zeroshot_vs_ct | 59 | 147.85 | [127.53, 167.84] | [-54.66, 350.37] |
| tta_r2_vs_ct | 59 | 20.32 | [3.99, 37.63] | [-138.63, 179.28] |
| us_gt_vs_ct | 59 | -26.98 | [-38.1, -14.15] | [-128.7, 74.75] |
