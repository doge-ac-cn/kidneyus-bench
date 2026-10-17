#!/usr/bin/env bash
# Download the TRUSTED dataset (14.89 GB zip) from Springer Nature figshare.
#
# NOTE: the springernature.figshare.com host sits behind an AWS WAF that
# challenges/blocks plain scripted requests from datacenter IPs. The legacy
# host ndownloader.figshare.com serves the same file without the WAF and
# 302-redirects to a short-lived signed S3 URL (follow it immediately).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/data/raw"
URL="https://ndownloader.figshare.com/files/51079133"
LOG="$ROOT/outputs/logs"
mkdir -p "$OUT" "$LOG"

echo "[$(date -Is)] TRUSTED download -> $OUT/TRUSTED.zip" | tee -a "$LOG/m0_trusted_download.log"
for attempt in 1 2 3 4 5; do
  curl -sL --retry 3 --retry-delay 10 -C - -o "$OUT/TRUSTED.zip.part" --max-time 28800 "$URL"
  rc=$?
  size=$(stat -c%s "$OUT/TRUSTED.zip.part" 2>/dev/null || echo 0)
  echo "[$(date -Is)] attempt=$attempt rc=$rc size=$size" | tee -a "$LOG/m0_trusted_download.log"
  if [ $rc -eq 0 ] && [ "$size" -gt 1000000000 ] && [ "$(head -c2 "$OUT/TRUSTED.zip.part")" = "PK" ]; then
    mv "$OUT/TRUSTED.zip.part" "$OUT/TRUSTED.zip"
    sha256sum "$OUT/TRUSTED.zip" > "$OUT/TRUSTED.zip.sha256"
    echo "[$(date -Is)] DONE size=$size" | tee -a "$LOG/m0_trusted_download.log"
    exit 0
  fi
  sleep 20
done
echo "[$(date -Is)] FAILED" | tee -a "$LOG/m0_trusted_download.log"
exit 1
