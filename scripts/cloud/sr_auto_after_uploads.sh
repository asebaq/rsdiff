#!/usr/bin/env bash
# Auto-trigger: poll for ep150/200/250/300 milestone uploads to land byte-exact,
# then launch the SR FID sweep on the missing range ep150-700.
#
# Idempotent — fid_sweep.sh skips milestones already in
# outputs/fid_curve_full_sr_gdm.tsv.
set -uo pipefail
ROOT="${ROOT:-/workspace/rsdiff}"
LOG_NAME="${LOG_NAME:-full_sr_gdm}"
EXPECTED=1899494706
EPS_NEW="${EPS_NEW:-150 200 250 300}"
EPS_FULL="${EPS_FULL:-150 200 250 300 350 400 450 500 550 600 650 700}"
SPE="${SPE:-137}"
MS_DIR="$ROOT/ddpm/logs/${LOG_NAME}/milestones"

echo "=== fidwait_mid start $(date -u) ==="
echo "watching: $MS_DIR for ep{$EPS_NEW}"

all_ready() {
  for ep in $EPS_NEW; do
    step=$((ep * SPE))
    f="$MS_DIR/ckpt_sr_ep${ep}_step${step}.pt"
    [ -f "$f" ] || return 1
    sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    [ "$sz" = "$EXPECTED" ] || return 1
  done
  return 0
}

while ! all_ready; do
  echo "$(date -u): not all ready, sleep 60"
  for ep in $EPS_NEW; do
    step=$((ep * SPE))
    f="$MS_DIR/ckpt_sr_ep${ep}_step${step}.pt"
    sz=$(stat -c%s "$f" 2>/dev/null || echo MISSING)
    echo "  ep${ep}: $sz"
  done
  sleep 60
done

echo "=== all uploads landed $(date -u) ==="
sleep 30

echo "=== launching sr_fid_sweep.sh with EPS=$EPS_FULL $(date -u) ==="
cd "$ROOT"
EPS="$EPS_FULL" bash scripts/eval/sr_fid_sweep.sh
echo "=== fidwait_mid done $(date -u) ==="
