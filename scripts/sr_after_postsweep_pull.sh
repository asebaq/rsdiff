#!/usr/bin/env bash
# Local-side final pull. Polls box for POST_FINAL_DONE sentinel, then rsyncs
# CFG sweep TSV/log/samples, final 1093 PNG bundle, fid_result.json, final log.
#
# After this completes, box is safe to destroy (#34 + #37 + #38 evidence
# all local).
set -uo pipefail
SSH="ssh -p 18252 -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o TCPKeepAlive=yes -o ConnectTimeout=20"
HOST="${HOST:-root@ssh3.vast.ai}"
BR="${BR:-/workspace/rsdiff}"
LR="${LR:-/Users/asebaq/dev/rsdiff/outputs/vast}"
SENT="$BR/POST_FINAL_DONE"

mkdir -p "$LR/fid" \
         "$LR/legacy/DDPM/logs/full_sr_gdm/generated_images"

echo "=== local final pull start $(date -u) ==="
while ! $SSH "$HOST" "[ -f $SENT ]" 2>/dev/null; do
  echo "$(date -u): no sentinel"
  sleep 600
done
echo "=== POST_FINAL_DONE found $(date -u) ==="

RSYNC_OPTS="-av --inplace --partial -e"

# CFG TSV(s) — any matching
$SSH "$HOST" "ls $BR/outputs/fid_cfg_*.tsv 2>/dev/null" | tr '\n' '\0' | while IFS= read -r -d '' tsv; do
  rsync $RSYNC_OPTS "$SSH" "$HOST:$tsv" "$LR/fid/"
done

# Logs
rsync $RSYNC_OPTS "$SSH" "$HOST:$BR/cfgsweep.log" "$LR/fid/" || true
rsync $RSYNC_OPTS "$SSH" "$HOST:$BR/final1093.log" "$LR/fid/" || true
rsync $RSYNC_OPTS "$SSH" "$HOST:$BR/clipscore.log" "$LR/fid/" || true
rsync $RSYNC_OPTS "$SSH" "$HOST:$BR/afterpost.log" "$LR/fid/" || true

# CFG sample dirs (cfg64_step*_cs*)
$SSH "$HOST" "ls -d $BR/legacy/DDPM/logs/full_sr_gdm/generated_images/cfg*_step*_cs* 2>/dev/null" | while read -r d; do
  base=$(basename "$d")
  rsync $RSYNC_OPTS "$SSH" "$HOST:$d/" "$LR/legacy/DDPM/logs/full_sr_gdm/generated_images/$base/"
done

# Final 1093 bundle(s)
$SSH "$HOST" "ls -d $BR/legacy/DDPM/logs/full_sr_gdm/generated_images/final_test_step* 2>/dev/null" | while read -r d; do
  base=$(basename "$d")
  rsync $RSYNC_OPTS "$SSH" "$HOST:$d/" "$LR/legacy/DDPM/logs/full_sr_gdm/generated_images/$base/"
done

# Sweep gen sample dirs (fid128_step*) — ~1.5 GB total
$SSH "$HOST" "ls -d $BR/legacy/DDPM/logs/full_sr_gdm/generated_images/fid128_step* 2>/dev/null" | while read -r d; do
  base=$(basename "$d")
  rsync $RSYNC_OPTS "$SSH" "$HOST:$d/" "$LR/legacy/DDPM/logs/full_sr_gdm/generated_images/$base/"
done

# Inline training samples (sample-SR-UNet-*.png) — ~70 MB
rsync $RSYNC_OPTS "$SSH" --include='sample-SR-UNet-*.png' --include='*/' --exclude='*' \
  "$HOST:$BR/legacy/DDPM/logs/full_sr_gdm/generated_images/" \
  "$LR/legacy/DDPM/logs/full_sr_gdm/generated_images/"

# Postsweep + final log (whole-chain summary)
rsync $RSYNC_OPTS "$SSH" "$HOST:$BR/postsweep.log" "$LR/fid/" || true

# Sentinels (audit trail)
rsync $RSYNC_OPTS "$SSH" "$HOST:$BR/POST_SWEEP_DONE" "$LR/fid/" || true
rsync $RSYNC_OPTS "$SSH" "$HOST:$BR/POST_FINAL_DONE" "$LR/fid/" || true

echo "=== ALL DONE $(date -u) ==="
ls -la "$LR/fid/" | head -30
ls -ld "$LR/legacy/DDPM/logs/full_sr_gdm/generated_images/cfg"* \
       "$LR/legacy/DDPM/logs/full_sr_gdm/generated_images/final_test_step"* \
       "$LR/legacy/DDPM/logs/full_sr_gdm/generated_images/fid128_step"* 2>/dev/null
