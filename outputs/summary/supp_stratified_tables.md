# Observer-stratified tables (supplementary)

## Sonographer A vs B (direct, by layer)

| layer | n | Dice | IoU |
|---|---|---|---|
| pole | 999 | 0.6182 | 0.4858 |
| discontinuity | 3036 | 0.8877 | 0.81 |
| rest | 14287 | 0.9304 | 0.875 |

_Layer note: pole slices take precedence; the discontinuity layer counts exclude the 420 slices that are both pole and discontinuity (3,456 flags - 420 = 3,036). _

## Model vs consensus / A / B (by layer)

| config | layer | n | vs consensus | vs A | vs B |
|---|---|---|---|---|---|
| nnunet | pole | 999 | 0.0259 | 0.0339 | 0.0383 |
| nnunet | discontinuity | 3036 | 0.3915 | 0.3933 | 0.3981 |
| nnunet | rest | 14287 | 0.7113 | 0.7079 | 0.7139 |
| medsam_p3 | pole | 999 | 0.7046 | 0.5701 | 0.5817 |
| medsam_p3 | discontinuity | 3036 | 0.8605 | 0.8304 | 0.8349 |
| medsam_p3 | rest | 14287 | 0.8997 | 0.883 | 0.8859 |
| medsam_p4reg | pole | 999 | 0.0008 | 0.0008 | 0.0015 |
| medsam_p4reg | discontinuity | 3036 | 0.1626 | 0.1613 | 0.1625 |
| medsam_p4reg | rest | 14287 | 0.5328 | 0.5288 | 0.5327 |
| medsam_tta_r2s43 | pole | 999 | 0.0453 | 0.0637 | 0.0603 |
| medsam_tta_r2s43 | discontinuity | 3036 | 0.5187 | 0.5206 | 0.5239 |
| medsam_tta_r2s43 | rest | 14287 | 0.7588 | 0.7547 | 0.7609 |
| samlora_p3 | pole | 999 | 0.3722 | 0.3233 | 0.3066 |
| samlora_p3 | discontinuity | 3036 | 0.8361 | 0.8105 | 0.8167 |
| samlora_p3 | rest | 14287 | 0.8939 | 0.8789 | 0.8846 |
| samlora_p4reg | pole | 999 | 0.0007 | 0.0007 | 0.0014 |
| samlora_p4reg | discontinuity | 3036 | 0.1613 | 0.1601 | 0.1619 |
| samlora_p4reg | rest | 14287 | 0.5313 | 0.5276 | 0.5326 |
| samlora_tta_r2s43 | pole | 999 | 0.0522 | 0.0717 | 0.069 |
| samlora_tta_r2s43 | discontinuity | 3036 | 0.5179 | 0.52 | 0.5252 |
| samlora_tta_r2s43 | rest | 14287 | 0.7551 | 0.7515 | 0.7591 |
| samus_p3 | pole | 999 | 0.3672 | 0.409 | 0.4073 |
| samus_p3 | discontinuity | 3036 | 0.6253 | 0.6373 | 0.6445 |
| samus_p3 | rest | 14287 | 0.7495 | 0.7539 | 0.7613 |
| samus_p4reg | pole | 999 | 0.0008 | 0.0009 | 0.0015 |
| samus_p4reg | discontinuity | 3036 | 0.1447 | 0.1449 | 0.1469 |
| samus_p4reg | rest | 14287 | 0.4848 | 0.485 | 0.4905 |
| samus_tta_r2s43 | pole | 999 | 0.0329 | 0.0468 | 0.0467 |
| samus_tta_r2s43 | discontinuity | 3036 | 0.4317 | 0.4389 | 0.4457 |
| samus_tta_r2s43 | rest | 14287 | 0.6684 | 0.6715 | 0.6791 |
| samus43_tta_r2s43 | pole | 999 | 0.0359 | 0.0521 | 0.0489 |
| samus43_tta_r2s43 | discontinuity | 3036 | 0.4721 | 0.4777 | 0.4826 |
| samus43_tta_r2s43 | rest | 14287 | 0.7161 | 0.7159 | 0.7233 |
