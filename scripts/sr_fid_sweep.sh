#!/usr/bin/env bash
# Post-ep1000 SR milestone FID sweep wrapper.
#
# Prepares the train box so legacy fid_sweep.sh can score the SR milestones:
#   1. Symlinks data/RSICD_optimal -> legacy/RSICD_optimal (fid_sweep.sh expects data/).
#   2. Symlinks ckpt_sr_ep${ep}_step${step}.pt -> ckpt_step${step}.pt (fid_sweep.sh
#      expects the legacy LR naming).
#   3. Sanity-checks rsdiff[eval] import + RSICD test split row count.
#   4. Launches the sweep in tmux session "fidsweep".
#
# Sweep config: N=128 cascade-256, cond_scale=4.0, feature=2048.
# Wall: ~25 hr on 4090 (~4.1 hr/milestone x 6 milestones).
#
# Output: outputs/fid_curve_full_sr_gdm.tsv
set -euo pipefail
ROOT="${ROOT:-/workspace/rsdiff}"
LOG_NAME="${LOG_NAME:-full_sr_gdm}"
EPS="${EPS:-750 800 850 900 950 1000}"
SPE="${SPE:-137}"
cd "$ROOT"

mkdir -p "$ROOT/data"
[ -L "$ROOT/data/RSICD_optimal" ] || ln -sf "$ROOT/legacy/RSICD_optimal" "$ROOT/data/RSICD_optimal"

MS_DIR="$ROOT/legacy/DDPM/logs/${LOG_NAME}/milestones"
cd "$MS_DIR"
STEPS=""
for ep in $EPS; do
  step=$((ep * SPE))
  src="ckpt_sr_ep${ep}_step${step}.pt"
  dst="ckpt_step${step}.pt"
  if [ -f "$src" ]; then
    [ -L "$dst" ] || ln -sf "$src" "$dst"
    echo "linked $dst -> $src"
    STEPS="$STEPS $step"
  else
    echo "MISSING $src (skip)"
  fi
done
STEPS="${STEPS# }"
echo "STEPS=$STEPS"

cd "$ROOT"
python -c "from rsdiff.eval.fid import fid; import pandas as pd; df=pd.read_csv('data/RSICD_optimal/dataset_rsicd.csv'); assert (df.split=='test').sum()==1093, 'test split count mismatch'; print('precheck OK')"

tmux kill-session -t fidsweep 2>/dev/null || true
tmux new -d -s fidsweep "
cd $ROOT && \
LOG_NAME=${LOG_NAME} \
STEPS='${STEPS}' \
N=128 BATCH=2 FEATURE=2048 SIZE=256 SR=1 \
bash scripts/fid_sweep.sh 2>&1 | tee $ROOT/fidsweep.log
"
sleep 2
tmux ls
echo "--- launched ---"
echo "Watch: tail -f $ROOT/fidsweep.log"
echo "Result: $ROOT/outputs/fid_curve_${LOG_NAME}.tsv"
