#!/usr/bin/env bash
# CFG scale sweep on best SR milestone. Iterates cond_scale over SCALES,
# generates N samples cascade-256 per scale, scores FID. Cheap pre-#34
# pick of guidance strength.
#
# Env knobs:
#   WINNER_STEP   step number of best milestone (required, e.g. 102750)
#   LOG_NAME      training log dir (default full_sr_gdm)
#   N             samples per scale (default 64)
#   BATCH         (default 2)
#   FEATURE       (default 2048)
#   SIZE          (default 256)
#   SCALES        space-separated cond_scale values (default "1 2 3 4 5 6 8")
#   SPE           steps per epoch (default 137)
set -euo pipefail
ROOT="${ROOT:-/workspace/rsdiff}"
LOG_NAME="${LOG_NAME:-full_sr_gdm}"
N="${N:-64}"
BATCH="${BATCH:-2}"
FEATURE="${FEATURE:-2048}"
SIZE="${SIZE:-256}"
SCALES="${SCALES:-1 2 3 4 5 6 8}"
SPE="${SPE:-137}"
WINNER_STEP="${WINNER_STEP:?set WINNER_STEP from fid_curve_full_sr_gdm.tsv winner}"
WINNER_EP=$((WINNER_STEP / SPE))

DATA="${ROOT}/data/RSICD_optimal"
LOGDIR="${ROOT}/legacy/DDPM/logs/${LOG_NAME}"
GENROOT="${LOGDIR}/generated_images"
CKPT="${LOGDIR}/milestones/ckpt_step${WINNER_STEP}.pt"
RESULTS="${ROOT}/outputs/fid_cfg_${LOG_NAME}_step${WINNER_STEP}.tsv"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${ROOT}"
[ -L "$DATA" ] || ln -sf "$ROOT/legacy/RSICD_optimal" "$DATA"
[ -f "$CKPT" ] || { echo "MISSING $CKPT — run sr_fid_sweep.sh first to symlink"; exit 1; }
mkdir -p "${ROOT}/outputs"
[ -f "${RESULTS}" ] || printf 'step\tepoch\tcond_scale\tfid\tfeature\tsize\tn_gen\n' > "${RESULTS}"

for scale in ${SCALES}; do
  sub="cfg${N}_step${WINNER_STEP}_cs${scale}"
  gendir="${GENROOT}/${sub}"
  if grep -q "^${WINNER_STEP}	${WINNER_EP}	${scale}	" "${RESULTS}" 2>/dev/null; then
    echo "=== cs=${scale}: already scored, skip ==="
    continue
  fi
  npng=$(ls "${gendir}"/*.png 2>/dev/null | wc -l || true)
  if [ "${npng}" -lt "${N}" ]; then
    echo "=== cs=${scale}: generating ${N} ($(date -u)) ==="
    python legacy/DDPM/sample_grid.py \
      --log_dir "${LOGDIR}" --data_root "${DATA}" \
      --ckpt "${CKPT}" \
      --n "${N}" --batch "${BATCH}" --no_grid --cond_scale "${scale}" \
      --img_sz 128 --sr_sz 256 --ts 1000 --device cuda \
      --sr --split test --seed 17 \
      --out_subdir "${sub}"
  else
    echo "=== cs=${scale}: ${npng} pngs present, skip gen ==="
  fi
  echo "=== cs=${scale}: scoring FID ($(date -u)) ==="
  fid=$(python - "$gendir" "$DATA" "$FEATURE" "$SIZE" <<'PY'
import sys
from rsdiff.eval.fid import fid
gendir, data, feature, size = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
r = fid(gen_dir=gendir, real_csv=f"{data}/dataset_rsicd.csv",
        real_dir=f"{data}/RSICD_images", split="test",
        feature=feature, image_size=size)
print(f"{r.fid:.4f}\t{r.n_gen}")
PY
)
  fidval="${fid%%	*}"; ngen="${fid##*	}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${WINNER_STEP}" "${WINNER_EP}" "${scale}" "${fidval}" "${FEATURE}" "${SIZE}" "${ngen}" >> "${RESULTS}"
  echo "=== cs=${scale}: FID=${fidval} (n_gen=${ngen}) ==="
done
echo "=== CFG SWEEP DONE $(date -u) ==="
cat "${RESULTS}"
