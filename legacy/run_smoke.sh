#!/usr/bin/env bash
# Smoke run: short LR-GDM training on RSICD.
# Default 10 epochs (~1-2hr on RTX A6000), env-overridable. Verifies
# imagen-pytorch>=1.26 API still matches build_models() + loss curve is sane
# before committing to the full 1000-epoch run.
#
# Env knobs: EPOCHS BATCH_SZ IMG_SZ TS LOG_NAME
#
# Expects:
#   legacy/RSICD_optimal/  (symlink from vast_setup.sh --download-rsicd)
#   imagen-pytorch>=1.26 installed (legacy/requirements_dgx.txt)
#
# Logs: legacy/DDPM/logs/${LOG_NAME}/

set -euo pipefail

LEGACY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${LEGACY_ROOT}"

EPOCHS="${EPOCHS:-10}"
BATCH_SZ="${BATCH_SZ:-64}"
IMG_SZ="${IMG_SZ:-128}"
TS="${TS:-1000}"
LOG_NAME="${LOG_NAME:-smoke_lr_gdm}"
LOG_DIR="${LEGACY_ROOT}/DDPM/logs/${LOG_NAME}"

python DDPM/Imagen_text_pytorch.py \
  --img_sz "${IMG_SZ}" \
  --ts "${TS}" \
  --batch_sz "${BATCH_SZ}" \
  --epochs "${EPOCHS}" \
  --data_root "${LEGACY_ROOT}/RSICD_optimal" \
  --log_dir "${LOG_DIR}" \
  --device auto

echo "smoke done. inspect:"
echo "  tail -n 50 ${LOG_DIR}/*.log"
echo "  tensorboard --logdir ${LOG_DIR}/runs"
echo "  ls ${LOG_DIR}/generated_images/"
