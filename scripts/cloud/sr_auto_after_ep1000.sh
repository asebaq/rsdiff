#!/usr/bin/env bash
# Auto-trigger: poll for ep1000 milestone, then kill srtrain/srsnap and launch
# the FID sweep. Run inside a detached tmux session ("fidwait") so it survives
# SSH disconnects.
#
# Sequence:
#   1. Wait until ckpt_sr_ep1000_step137000.pt exists in milestones/.
#   2. Sleep 120s to let srsnap flush + srtrain quiesce.
#   3. Kill srtrain + srsnap tmux sessions to free the GPU.
#   4. Invoke sr_fid_sweep.sh (which starts its own "fidsweep" tmux).
set -uo pipefail
ROOT="${ROOT:-/workspace/rsdiff}"
LOG_NAME="${LOG_NAME:-full_sr_gdm}"
TARGET_STEP="${TARGET_STEP:-137000}"
TARGET="$ROOT/ddpm/logs/${LOG_NAME}/milestones/ckpt_sr_ep1000_step${TARGET_STEP}.pt"

echo "=== fidwait start $(date -u) ==="
echo "watching: $TARGET"

while [ ! -f "$TARGET" ]; do
  echo "$(date -u): not present, sleep 60"
  sleep 60
done

echo "=== ep1000 milestone landed $(date -u) ==="
ls -la "$TARGET"

echo "=== sleep 120s to drain $(date -u) ==="
sleep 120

for sess in srtrain srsnap; do
  if tmux has-session -t "$sess" 2>/dev/null; then
    echo "killing tmux $sess"
    tmux kill-session -t "$sess" || true
  fi
done

echo "=== launching sr_fid_sweep.sh $(date -u) ==="
cd "$ROOT"
bash scripts/eval/sr_fid_sweep.sh
echo "=== fidwait done $(date -u) ==="
