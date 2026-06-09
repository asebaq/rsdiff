#!/usr/bin/env bash
# Box-side post-sweep finalizer. Polls fid_curve_full_sr_gdm.tsv until
# header + 18 milestone rows present, then regenerates grid_sr_ep500
# (skipped during snap loop), and writes POST_SWEEP_DONE sentinel.
#
# Pair: scripts/sr_post_sweep_pull.sh (local-side rsync watcher).
set -uo pipefail
ROOT="${ROOT:-/workspace/rsdiff}"
LOG="$ROOT/legacy/DDPM/logs/full_sr_gdm"
TSV="$ROOT/outputs/fid_curve_full_sr_gdm.tsv"
EXPECTED_ROWS="${EXPECTED_ROWS:-19}"   # 1 header + 18 milestones
GRID_EP="${GRID_EP:-500}"
SPE="${SPE:-137}"

echo "=== post_sweep start $(date -u) ==="
echo "wait tsv >= $EXPECTED_ROWS rows: $TSV"

while :; do
  rows=$(wc -l < "$TSV" 2>/dev/null || echo 0)
  echo "$(date -u): tsv rows=$rows"
  [ "${rows:-0}" -ge "$EXPECTED_ROWS" ] && break
  sleep 300
done
echo "=== sweep done $(date -u) ==="

step=$((GRID_EP * SPE))
ckpt="$LOG/milestones/ckpt_sr_ep${GRID_EP}_step${step}.pt"
out="grid_sr_ep${GRID_EP}"

cd "$ROOT"
if [ -d "$LOG/generated_images/$out" ]; then
  echo "skip $out (exists)"
else
  echo "=== gen $out $(date -u) ==="
  python legacy/DDPM/sample_grid.py --log_dir "$LOG" \
    --data_root "$ROOT/legacy/RSICD_optimal" \
    --ckpt "$ckpt" --sr --n 16 --cols 4 --batch 2 --out_subdir "$out" \
    || echo "FAIL ep${GRID_EP}"
fi

touch "$ROOT/POST_SWEEP_DONE"
echo "=== post_sweep done $(date -u) ==="
