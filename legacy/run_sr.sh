#!/usr/bin/env bash
# Full SR-GDM training (path B): freeze the reproduced lightweight LR base,
# train ONLY the super-resolution unet (128 -> 256). Unet 1 is seeded from the
# LR run's checkpoint (LR_CKPT) and frozen; see DDPM/Imagen_text_sr_pytorch.py.
#
# Env knobs: EPOCHS BATCH_SZ IMG_SZ SR_SZ TS LOG_NAME LR_CKPT
#
# Expects:
#   legacy/RSICD_optimal/  (symlink from vast_setup.sh --download-rsicd)
#   a finished LR run at legacy/DDPM/logs/full_lr_gdm/checkpoint.pt (or LR_CKPT)
#
# Logs: legacy/DDPM/logs/${LOG_NAME}/

set -euo pipefail

LEGACY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${LEGACY_ROOT}"

EPOCHS="${EPOCHS:-1000}"
BATCH_SZ="${BATCH_SZ:-64}"
IMG_SZ="${IMG_SZ:-128}"
SR_SZ="${SR_SZ:-256}"
TS="${TS:-1000}"
LOG_NAME="${LOG_NAME:-full_sr_gdm}"
LOG_DIR="${LEGACY_ROOT}/DDPM/logs/${LOG_NAME}"
LR_CKPT="${LR_CKPT:-${LEGACY_ROOT}/DDPM/logs/full_lr_gdm/checkpoint.pt}"

python DDPM/Imagen_text_sr_pytorch.py \
  --img_sz "${IMG_SZ}" \
  --sr_sz "${SR_SZ}" \
  --ts "${TS}" \
  --batch_sz "${BATCH_SZ}" \
  --epochs "${EPOCHS}" \
  --data_root "${LEGACY_ROOT}/RSICD_optimal" \
  --log_dir "${LOG_DIR}" \
  --lr_ckpt "${LR_CKPT}" \
  --device auto

echo "SR run done. inspect:"
echo "  tail -n 50 ${LOG_DIR}/*.log"
echo "  tensorboard --logdir ${LOG_DIR}/runs"
echo "  ls ${LOG_DIR}/generated_images/"
