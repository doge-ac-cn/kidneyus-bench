# OKD Access Email Record (redacted)

Access to the Open Kidney Dataset was granted by the dataset author,
Dr. Rohit Singla (University of British Columbia), by email on 2026-10-08,
in response to the registration form. The reply contained two private
OneDrive share links:

- dataset share link — **redacted**; consumed via the `OKD_DATA_SHARE`
  environment variable by `scripts/download_okd.sh`;
- pretrained-weights share link — **redacted**; consumed via the
  `OKD_WEIGHTS_SHARE` environment variable by `scripts/download_okd.sh`.

Neither link is stored anywhere in this repository. The author's requested
attribution:

> Singla, R., Ringstrom, C., Hu, G., Lessoway, V., Reid, J., Nguan, C., &
> Rohling, R. (2023, October). The open kidney ultrasound data set.
> In *International Workshop on Advances in Simplifying Medical Ultrasound
> (ASMIC)* (pp. 155–164). Cham: Springer Nature Switzerland.

Project page: https://github.com/rsingla92/kidneyUS
License: CC BY-NC-SA 4.0 (non-commercial; from the kidneyUS repository README).
Facts recorded at verification (2026-10-08): 534 PNGs distributed; 486 cases
carry non-empty kidney-capsule masks from both annotators (the cohort used in
this study); no native/transplant metadata field in the distribution.
