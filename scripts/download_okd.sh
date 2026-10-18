#!/usr/bin/env bash
# Download the OKD (Open Kidney Dataset) data + author-released weights from
# the author's OneDrive share, obtained after registration (see README).
#
# PRIVACY / ACCESS NOTE: the OneDrive share links are sent to registered users
# by the dataset author (Dr. Rohit Singla, UBC) and are intentionally NOT
# hardcoded here. Export the two variables below from your registration email.
#
# The share backend is a SharePoint-personal (migrated) account, so the
# api.onedrive.com shares endpoint returns 401; the working anonymous direct
# download for a redeemed share token is:
#   https://my.microsoftpersonalcontent.com/personal/<CID>/_layouts/15/download.aspx?share=<token>
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/outputs/logs"
OUT="$ROOT/scratch"
mkdir -p "$OUT" "$LOG"

CID="${OKD_ONEDRIVE_CID:?export OKD_ONEDRIVE_CID from your registration email}"
DATA_URL="${OKD_DATA_SHARE:?export OKD_DATA_SHARE (s!... share token)}"
WEIGHTS_URL="${OKD_WEIGHTS_SHARE:?export OKD_WEIGHTS_SHARE (s!... share token)}"

dl () { # $1 = share token, $2 = output file, $3 = label
  local url="https://my.microsoftpersonalcontent.com/personal/${CID}/_layouts/15/download.aspx?share=$1"
  echo "[$(date -Is)] $3 start -> $2" | tee -a "$LOG/m0_okd_download.log"
  curl -sL --retry 5 --retry-delay 15 -o "$2.part" --max-time 7200 "$url"
  local rc=$?
  local size=$(stat -c%s "$2.part" 2>/dev/null || echo 0)
  local magic=$(head -c2 "$2.part" 2>/dev/null)
  if [ $rc -eq 0 ] && [ "$size" -gt 1000 ] && [ "$magic" = "PK" ]; then
    mv "$2.part" "$2"
    sha256sum "$2" > "$2.sha256"
    echo "[$(date -Is)] $3 DONE size=$size" | tee -a "$LOG/m0_okd_download.log"
  else
    echo "[$(date -Is)] $3 FAILED rc=$rc size=$size magic='$magic'" | tee -a "$LOG/m0_okd_download.log"
    return 1
  fi
}

dl "$DATA_URL" "$OUT/OKD_data.zip" "OKD-data"
dl "$WEIGHTS_URL" "$OUT/OKD_weights.zip" "OKD-weights"
echo "[$(date -Is)] OKD downloads finished" | tee -a "$LOG/m0_okd_download.log"
