# Boundary metrics on TRUSTED (supplementary)

HD95 = conditional (valid slices; NaN when prediction or GT empty). NSD@1 mm = normalized surface Dice vs STAPLE; empty prediction -> 0.

| config | n | HD95 mean (mm) | HD95 valid n | HD95 NaN n | NSD@1mm | NSD=0 n |
|---|---|---|---|---|---|---|
| nnunet | 18322 | 53.9 | 18322 | 0 | 0.2642 | 2153 |
| medsam_p3 | 18322 | 5.75 | 18318 | 4 | 0.4092 | 27 |
| medsam_p4reg | 18322 | 11.65 | 9434 | 8888 | 0.1857 | 8957 |
| medsam_tta_r2s43 | 18322 | 17.54 | 16802 | 1520 | 0.2684 | 2440 |
| samlora_p3 | 18322 | 9.13 | 18141 | 181 | 0.4066 | 264 |
| samlora_p4reg | 18322 | 12.15 | 9434 | 8888 | 0.193 | 8953 |
| samlora_tta_r2s43 | 18322 | 18.38 | 16802 | 1520 | 0.2718 | 2374 |
| samus_p3 | 18322 | 20.7 | 18322 | 0 | 0.1605 | 2774 |
| samus_p4reg | 18322 | 17.25 | 9434 | 8888 | 0.11 | 9469 |
| samus_tta_r2s43 | 18322 | 25.95 | 16802 | 1520 | 0.1391 | 4125 |
| samus43_tta_r2s43 | 18322 | 21.14 | 16802 | 1520 | 0.1871 | 2657 |
