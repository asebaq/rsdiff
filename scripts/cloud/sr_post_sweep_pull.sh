#!/usr/bin/env bash
# Local-side pull watcher. Polls box for POST_SWEEP_DONE sentinel, then
# rsyncs SR FID sweep artifacts + grid_sr_ep500 to outputs/vast/.
#
# Pair: scripts/cloud/sr_post_sweep.sh (box-side finalizer).
set -uo pipefail
SSH="ssh -p 18252 -o StrictHostKeyChecking=no -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o TCPKeepAlive=yes -o ConnectTimeout=20"
HOST="${HOST:-root@ssh3.vast.ai}"
BR="${BR:-/workspace/rsdiff}"
LR="${LR:-/Users/asebaq/dev/rsdiff/outputs/vast}"
SENT="$BR/POST_SWEEP_DONE"

mkdir -p "$LR/fid" \
         "$LR/ddpm/logs/full_sr_gdm/generated_images"

echo "=== local pull start $(date -u) ==="
echo "wait $HOST:$SENT"
while ! $SSH "$HOST" "[ -f $SENT ]" 2>/dev/null; do
  echo "$(date -u): no sentinel"
  sleep 300
done
echo "=== sentinel found $(date -u) ==="

RSYNC_OPTS="-av --inplace --partial -e"
rsync $RSYNC_OPTS "$SSH" "$HOST:$BR/outputs/fid_curve_full_sr_gdm.tsv" "$LR/fid/"
rsync $RSYNC_OPTS "$SSH" "$HOST:$BR/fidsweep.log" "$LR/fid/"
rsync $RSYNC_OPTS "$SSH" "$HOST:$BR/ddpm/logs/full_sr_gdm/srsnap.log" "$LR/ddpm/logs/full_sr_gdm/"
rsync $RSYNC_OPTS "$SSH" "$HOST:$BR/ddpm/logs/full_sr_gdm/generated_images/grid_sr_ep500/" \
  "$LR/ddpm/logs/full_sr_gdm/generated_images/grid_sr_ep500/"

echo "=== ALL DONE $(date -u) ==="
ls -la "$LR/fid/fid_curve_full_sr_gdm.tsv" "$LR/fid/fidsweep.log" \
       "$LR/ddpm/logs/full_sr_gdm/srsnap.log" \
       "$LR/ddpm/logs/full_sr_gdm/generated_images/grid_sr_ep500/" 2>/dev/null
