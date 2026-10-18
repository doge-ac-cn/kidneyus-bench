# Observer-stratified tables (supplementary, registered seed 43)

## Sonographer A vs B (direct, by layer)

| layer | n | Dice | IoU |
|---|---|---|---|
| pole | 999 | 0.6182 | 0.4858 |
| discontinuity | 3036 | 0.8877 | 0.81 |
| rest | 14287 | 0.9304 | 0.875 |

_Layer note: pole slices take precedence (3,456 flags - 420 overlap = 3,036 discontinuity-only)._

## Model vs consensus / A / B (by layer)

| config | layer | n | vs consensus | vs A | vs B |
|---|---|---|---|---|---|
| nnunet | pole | 999 | 0.026 | 0.0343 | 0.0403 |
| nnunet | discontinuity | 3036 | 0.3817 | 0.3837 | 0.389 |
| nnunet | rest | 14287 | 0.7081 | 0.7042 | 0.7109 |
| medsam_p3 | pole | 999 | 0.7202 | 0.5834 | 0.5953 |
| medsam_p3 | discontinuity | 3036 | 0.8621 | 0.832 | 0.8362 |
| medsam_p3 | rest | 14287 | 0.8997 | 0.883 | 0.886 |
| medsam_p4reg | pole | 999 | 0.0008 | 0.0008 | 0.0015 |
| medsam_p4reg | discontinuity | 3036 | 0.1627 | 0.1614 | 0.1626 |
| medsam_p4reg | rest | 14287 | 0.5328 | 0.5288 | 0.5329 |
| medsam_tta_r2 | pole | 999 | 0.0447 | 0.0631 | 0.0591 |
| medsam_tta_r2 | discontinuity | 3036 | 0.5177 | 0.5196 | 0.5228 |
| medsam_tta_r2 | rest | 14287 | 0.7583 | 0.7543 | 0.7605 |
