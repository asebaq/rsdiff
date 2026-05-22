#!/usr/bin/env bash
# Joint fine-tune (paper phase 4): train BOTH unets, base unfrozen, combined
# objective L = L_LR + LAMBDA_SR * L_SR (LAMBDA_SR=0.8). Cheap legacy approx:
# both unets see GT low-res (not LR-GDM output), and the two losses step
# separately (LAMBDA_SR applied by scaling the SR unet grads). See
# DDPM/Imagen_text_joint_pytorch.py for the two shortcuts vs the paper.
#
# Env knobs: EPOCHS BATCH_SZ IMG_SZ SR_SZ TS LAMBDA_SR LOG_NAME INIT_CKPT
#
# Expects:
#   legacy/RSICD_optimal/  (symlink from vast_setup.sh --download-rsicd)
#   a finished SR run at legacy/DDPM/logs/full_sr_gdm/checkpoint.pt (or INIT_CKPT)
#
# Logs: legacy/DDPM/logs/${LOG_NAME}/

set -euo pipefail

LEGACY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${LEGACY_ROOT}"

EPOCHS="${EPOCHS:-200}"
BATCH_SZ="${BATCH_SZ:-64}"
IMG_SZ="${IMG_SZ:-128}"
SR_SZ="${SR_SZ:-256}"
TS="${TS:-1000}"
LAMBDA_SR="${LAMBDA_SR:-0.8}"
LOG_NAME="${LOG_NAME:-full_joint_gdm}"
LOG_DIR="${LEGACY_ROOT}/DDPM/logs/${LOG_NAME}"
INIT_CKPT="${INIT_CKPT:-${LEGACY_ROOT}/DDPM/logs/full_sr_gdm/checkpoint.pt}"

python DDPM/Imagen_text_joint_pytorch.py \
  --img_sz "${IMG_SZ}" \
  --sr_sz "${SR_SZ}" \
  --ts "${TS}" \
  --batch_sz "${BATCH_SZ}" \
  --epochs "${EPOCHS}" \
  --lambda_sr "${LAMBDA_SR}" \
  --data_root "${LEGACY_ROOT}/RSICD_optimal" \
  --log_dir "${LOG_DIR}" \
  --init_ckpt "${INIT_CKPT}" \
  --device auto

echo "Joint run done. inspect:"
echo "  tail -n 50 ${LOG_DIR}/*.log"
echo "  tensorboard --logdir ${LOG_DIR}/runs"
echo "  ls ${LOG_DIR}/generated_images/"
