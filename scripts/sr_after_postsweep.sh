#!/usr/bin/env bash
# Box-side chain after POST_SWEEP_DONE: pick winner step from FID curve TSV,
# run CFG sweep on that milestone, pick winner cond_scale, run final 1093
# headline FID + asset bundle. Writes POST_FINAL_DONE sentinel.
#
# Pairs with sr_post_sweep.sh (which writes POST_SWEEP_DONE) and
# sr_after_postsweep_pull.sh (local-side rsync watcher).
set -uo pipefail
ROOT="${ROOT:-/workspace/rsdiff}"
TSV="$ROOT/outputs/fid_curve_full_sr_gdm.tsv"
SENTINEL_IN="$ROOT/POST_SWEEP_DONE"
SENTINEL_OUT="$ROOT/POST_FINAL_DONE"

echo "=== after_postsweep start $(date -u) ==="
echo "wait $SENTINEL_IN"
while [ ! -f "$SENTINEL_IN" ]; do
  sleep 60
done
echo "=== POST_SWEEP_DONE found $(date -u) ==="

# Winner = min FID row in sweep TSV
read winner_ep winner_step winner_fid < <(awk -F'\t' 'NR>1 && (min=="" || $3+0<min+0) {min=$3; ep=$1; step=$2} END {print ep, step, min}' "$TSV")
echo "=== winner: ep=$winner_ep step=$winner_step fid=$winner_fid ==="
if [ -z "$winner_step" ]; then
  echo "FAIL: no winner parsed from $TSV"; exit 1
fi

# CFG sweep on winner
echo "=== CFG sweep start $(date -u) ==="
cd "$ROOT"
WINNER_STEP="$winner_step" bash scripts/sr_cfg_sweep.sh 2>&1 | tee "$ROOT/cfgsweep.log"
cfg_status=${PIPESTATUS[0]}
if [ "$cfg_status" != "0" ]; then
  echo "FAIL: CFG sweep exit=$cfg_status"; exit 1
fi

cfg_tsv="$ROOT/outputs/fid_cfg_full_sr_gdm_step${winner_step}.tsv"
read winner_scale winner_cfg_fid < <(awk -F'\t' 'NR>1 && (min=="" || $4+0<min+0) {min=$4; scale=$3} END {print scale, min}' "$cfg_tsv")
echo "=== winner_scale=$winner_scale fid=$winner_cfg_fid ==="
if [ -z "$winner_scale" ]; then
  echo "FAIL: no winner_scale parsed from $cfg_tsv"; exit 1
fi

# Final 1093 with both winners
echo "=== final 1093 start step=$winner_step cs=$winner_scale $(date -u) ==="
WINNER_STEP="$winner_step" CFG_SCALE="$winner_scale" bash scripts/sr_final_1093.sh 2>&1 | tee "$ROOT/final1093.log"
final_status=${PIPESTATUS[0]}
if [ "$final_status" != "0" ]; then
  echo "FAIL: final 1093 exit=$final_status"; exit 1
fi

# CLIP-score on final 1093 bundle (#36)
final_dir="$ROOT/legacy/DDPM/logs/full_sr_gdm/generated_images/final_test_step${winner_step}_cs${winner_scale}"
echo "=== CLIP score start $(date -u) ==="
GEN_DIR="$final_dir" bash scripts/sr_clip_score.sh 2>&1 | tee "$ROOT/clipscore.log"
clip_status=${PIPESTATUS[0]}
if [ "$clip_status" != "0" ]; then
  echo "WARN: CLIP score exit=$clip_status (non-fatal, continue)"
fi

touch "$SENTINEL_OUT"
echo "=== after_postsweep done $(date -u) ==="
echo "Winner: ep=$winner_ep step=$winner_step cs=$winner_scale"
