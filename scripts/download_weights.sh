#!/usr/bin/env bash
# M0: foundation-model checkpoint downloads.
# huggingface.co is unreachable from this host -> use hf-mirror.com for MedSAM.
set -u
ROOT="${KIDNEYUS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
W="$ROOT/data/weights"
LOG="$ROOT/outputs/logs/m0_weights_download.log"
mkdir -p "$W" "$ROOT/outputs/logs"

fetch () { # $1=url $2=out $3=label
  if [ -s "$2" ]; then echo "[$(date -Is)] $3 exists, skip" | tee -a "$LOG"; return 0; fi
  echo "[$(date -Is)] $3 start" | tee -a "$LOG"
  curl -sL --retry 5 --retry-delay 10 -o "$2.part" --max-time 3600 "$1"
  rc=$?; size=$(stat -c%s "$2.part" 2>/dev/null || echo 0)
  if [ $rc -eq 0 ] && [ "$size" -gt 10000000 ]; then
    mv "$2.part" "$2"; sha256sum "$2" > "$2.sha256"
    # official checksum spot-check where published (SAM URL embeds md5 prefix).
    # NOTE: match the full filename — "*sam_vit_b*" would also hit medsam_vit_b
    if [[ "$2" == *sam_vit_b_01ec64* ]]; then
      md5=$(md5sum "$2" | cut -d' ' -f1)
      case "$md5" in 01ec64*) : ;; *) echo "[$(date -Is)] $3 MD5 MISMATCH ($md5)" | tee -a "$LOG"; rm -f "$2" "$2.sha256"; return 1;; esac
    fi
    echo "[$(date -Is)] $3 DONE size=$size" | tee -a "$LOG"
  else
    echo "[$(date -Is)] $3 FAILED rc=$rc size=$size" | tee -a "$LOG"; return 1
  fi
}

# M3 MedSAM vit_b (official ckpt mirrored on HF; wanglab/medsam-vit-base)
fetch "https://hf-mirror.com/wanglab/medsam-vit-base/resolve/main/pytorch_model.bin" \
      "$W/medsam_vit_b.pth" "MedSAM-vitB"
# M4 SAM vit_b (official)
fetch "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
      "$W/sam_vit_b_01ec64.pth" "SAM-vitB"
# SAM 2.1 hiera base+ (backup / potential baseline)
fetch "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt" \
      "$W/sam2.1_hiera_base_plus.pt" "SAM2.1-B+"
echo "[$(date -Is)] weight downloads finished" | tee -a "$LOG"
