#!/usr/bin/env bash
# Sample N test-set captions from a trained LR-GDM checkpoint and write a grid.
#
# Env knobs: LOG_DIR N COLS IMG_SZ TS COND_SCALE SPLIT
#
# Default: uses the smoke run's log dir.

set -euo pipefail

LEGACY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${LEGACY_ROOT}"

LOG_DIR="${LOG_DIR:-${LEGACY_ROOT}/DDPM/logs/smoke_lr_gdm}"
DATA_ROOT="${DATA_ROOT:-${LEGACY_ROOT}/RSICD_optimal}"
N="${N:-16}"
COLS="${COLS:-4}"
IMG_SZ="${IMG_SZ:-128}"
TS="${TS:-1000}"
COND_SCALE="${COND_SCALE:-4.0}"
SPLIT="${SPLIT:-test}"

python DDPM/sample_grid.py \
  --log_dir "${LOG_DIR}" \
  --data_root "${DATA_ROOT}" \
  --n "${N}" --cols "${COLS}" \
  --img_sz "${IMG_SZ}" --ts "${TS}" \
  --cond_scale "${COND_SCALE}" \
  --split "${SPLIT}" \
  --device auto

echo "grid done. see: ${LOG_DIR}/generated_images/grid_step*/"
