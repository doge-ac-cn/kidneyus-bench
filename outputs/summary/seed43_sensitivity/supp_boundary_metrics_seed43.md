# Boundary metrics on TRUSTED (supplementary, registered seed 43)

HD95 = conditional (valid slices). NSD@1 mm vs STAPLE; empty prediction -> 0.

| config | n | HD95 mean (mm) | HD95 valid n | HD95 NaN n | NSD@1mm | NSD=0 n |
|---|---|---|---|---|---|---|
| nnunet | 18322 | 55.74 | 18322 | 0 | 0.2687 | 2387 |
| medsam_p3 | 18322 | 5.72 | 18317 | 5 | 0.4083 | 21 |
| medsam_p4reg | 18322 | 11.61 | 9434 | 8888 | 0.186 | 8962 |
| medsam_tta_r2 | 18322 | 17.46 | 16802 | 1520 | 0.2674 | 2434 |
| samlora_tta_r2 | 18322 | 18.07 | 16802 | 1520 | 0.2553 | 2465 |
| samus_tta_r2 | 18322 | 21.14 | 16802 | 1520 | 0.1871 | 2657 |
