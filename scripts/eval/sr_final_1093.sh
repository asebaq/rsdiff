#!/usr/bin/env bash
# Post-sweep: full 1093 cascade-256 asset gen on winner milestone.
# Saves PNGs as reusable bundle + computes FID against RSICD test split.
#
# Set WINNER_STEP env to the best step from fid_curve_full_sr_gdm.tsv
# before invoking. Defaults to 137000 (ep1000).
#
# Wall: ~35 hr on 4090 (or ~12 hr on H100).
set -euo pipefail
ROOT="${ROOT:-/workspace/rsdiff}"
LOG_NAME="${LOG_NAME:-full_sr_gdm}"
SPE="${SPE:-137}"
WINNER_STEP="${WINNER_STEP:-137000}"
CFG_SCALE="${CFG_SCALE:-4.0}"
WINNER_EP=$((WINNER_STEP / SPE))
OUT_SUBDIR="final_test_step${WINNER_STEP}_cs${CFG_SCALE}"

cd "$ROOT"
LOG="ddpm/logs/${LOG_NAME}"
DATA="$ROOT/data/RSICD_optimal"
CKPT="$LOG/milestones/ckpt_step${WINNER_STEP}.pt"

echo "=== final 1093 sample on ep${WINNER_EP} step${WINNER_STEP} $(date -u) ==="
[ -f "$CKPT" ] || { echo "MISSING $CKPT (run sr_fid_sweep.sh first to create symlink)"; exit 1; }
[ -L "$DATA" ] || ln -sf "$ROOT/ddpm/RSICD_optimal" "$DATA"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python ddpm/sample_grid.py \
  --log_dir "$LOG" \
  --data_root "$DATA" \
  --ckpt "$CKPT" \
  --n 1093 --batch 2 --no_grid \
  --cond_scale "$CFG_SCALE" \
  --img_sz 128 --sr_sz 256 --ts 1000 \
  --device cuda --sr --split test --seed 17 \
  --out_subdir "$OUT_SUBDIR"

GEN_DIR="$LOG/generated_images/$OUT_SUBDIR"
echo "=== compute FID $(date -u) ==="
python - "$GEN_DIR" "$DATA" <<'PY'
import sys, json
from rsdiff.eval.fid import fid
gendir, data = sys.argv[1], sys.argv[2]
r = fid(gen_dir=gendir, real_csv=f"{data}/dataset_rsicd.csv",
        real_dir=f"{data}/RSICD_images", split="test",
        feature=2048, image_size=256)
out = {"fid": r.fid, "n_gen": r.n_gen, "feature": 2048, "size": 256, "split": "test"}
print(json.dumps(out, indent=2))
with open(f"{gendir}/fid_result.json", "w") as f:
    json.dump(out, f, indent=2)
PY

echo "=== DONE $(date -u) ==="
echo "Bundle: $GEN_DIR"
echo "FID result: $GEN_DIR/fid_result.json"
