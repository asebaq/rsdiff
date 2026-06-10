#!/usr/bin/env bash
# FID-vs-epoch sweep: for each milestone checkpoint, generate N test-caption
# samples then score FID against the RSICD split. Streams one result row per
# milestone so a partial curve is readable while the rest still runs.
#
# Designed for the dedicated FID box (idle GPU). Idempotent: re-running skips
# milestones whose generations + FID row already exist, so it resumes cleanly.
#
# Env knobs:
#   LOG_NAME   training log dir under ddpm/logs (default full_lr_gdm)
#   STEPS      space-separated checkpoint step numbers (default = ep100..700)
#   N          samples per milestone (default 128)
#   BATCH      sampling chunk (default 4; safe on a 12GB card)
#   FEATURE    Inception feature dim (default 2048, paper-comparable)
#   SIZE       image size scored (default 128 for the LR base; 256 for SR)
#   STEPS_PER_EPOCH  for the epoch column (default 137)
#   SR         1 -> two-unet cascade sampling (256); default base-only
set -euo pipefail

ROOT="${ROOT:-/workspace/rsdiff}"
LOG_NAME="${LOG_NAME:-full_lr_gdm}"
N="${N:-128}"
BATCH="${BATCH:-4}"
FEATURE="${FEATURE:-2048}"
SIZE="${SIZE:-128}"
SPE="${STEPS_PER_EPOCH:-137}"
STEPS="${STEPS:-13700 27400 41100 54800 68500 82200 95900}"
DATA="${ROOT}/data/RSICD_optimal"
LOGDIR="${ROOT}/ddpm/logs/${LOG_NAME}"
GENROOT="${LOGDIR}/generated_images"
RESULTS="${ROOT}/outputs/fid_curve_${LOG_NAME}.tsv"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${ROOT}"
mkdir -p "${ROOT}/outputs"
[ -f "${RESULTS}" ] || printf 'epoch\tstep\tfid\tfeature\tsize\tn_gen\n' > "${RESULTS}"

sr_arg=""
[ "${SR:-0}" = "1" ] && sr_arg="--sr"

for step in ${STEPS}; do
  epoch=$(( step / SPE ))
  sub="fid${N}_step${step}"
  gendir="${GENROOT}/${sub}"
  if grep -q "^${epoch}	${step}	" "${RESULTS}" 2>/dev/null; then
    echo "=== ep${epoch} step${step}: already scored, skip ==="
    continue
  fi
  npng=$(ls "${gendir}"/*.png 2>/dev/null | wc -l || true)
  if [ "${npng}" -lt "${N}" ]; then
    echo "=== ep${epoch} step${step}: generating ${N} ($(date)) ==="
    python ddpm/sample_grid.py \
      --log_dir "${LOGDIR}" --data_root "${DATA}" \
      --ckpt "${LOGDIR}/milestones/ckpt_step${step}.pt" \
      --n "${N}" --batch "${BATCH}" --no_grid --cond_scale 4.0 \
      --img_sz 128 --sr_sz 256 --ts 1000 --device cuda \
      ${sr_arg} --out_subdir "${sub}"
  else
    echo "=== ep${epoch} step${step}: ${npng} pngs present, skip gen ==="
  fi
  echo "=== ep${epoch} step${step}: scoring FID ($(date)) ==="
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
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${epoch}" "${step}" "${fidval}" "${FEATURE}" "${SIZE}" "${ngen}" >> "${RESULTS}"
  echo "=== ep${epoch} step${step}: FID=${fidval} (n_gen=${ngen}) ==="
done
echo "=== SWEEP DONE $(date) ==="
column -t "${RESULTS}"
