#!/usr/bin/env bash
# Server-side: snapshot legacy/DDPM/logs/<NAME>/checkpoint.pt -> checkpoint_step{N}.pt.
# Reads the step counter from the .pt file so each snapshot is uniquely tagged.
#
# Pruning policy: high/low watermark. Snapshots accumulate up to MAX (default 20).
# When the count exceeds MAX, prune the oldest until PRUNE_TO (default 10) remain.
# Disk ceiling = MAX * 416 MB (~8.3 GB at defaults).
#
# Milestones: any snapshot whose step crosses a multiple of MILESTONE_STEPS (env,
# default 13700 = 100 epochs at 137 batches/epoch) is *also* copied into
# LOG_DIR/milestones/ckpt_step{N}.pt. Milestones are never pruned.
#
# Usage:
#   bash scripts/remote_snapshot.sh LOG_DIR [MAX] [PRUNE_TO]

set -euo pipefail

LOG_DIR="${1:?usage: $0 LOG_DIR [MAX] [PRUNE_TO]}"
MAX="${2:-20}"
PRUNE_TO="${3:-10}"
CKPT="${LOG_DIR}/checkpoint.pt"

if [ ! -f "${CKPT}" ]; then
  echo "no checkpoint at ${CKPT}" >&2
  exit 1
fi

STEP=$(python3 -c "
import torch
ck = torch.load('${CKPT}', map_location='cpu', weights_only=False)
print(int(ck['steps'].sum().item()))
" 2>/dev/null || echo "unknown")

DST="${LOG_DIR}/checkpoint_step${STEP}.pt"
NEW_SNAPSHOT=0

if [ -f "${DST}" ]; then
  echo "rotating snapshot step=${STEP} already exists"
else
  cp "${CKPT}" "${DST}"
  size=$(du -h "${DST}" | cut -f1)
  echo "snapshot -> ${DST}  (${size})"
  NEW_SNAPSHOT=1
fi

# Milestone tier (never pruned). Checked every run, independent of rotating dedup.
MILESTONE_STEPS="${MILESTONE_STEPS:-13700}"
if [ "${MILESTONE_STEPS}" -gt 0 ] && [ "${STEP}" != "unknown" ]; then
  CURRENT_MILESTONE=$(( STEP / MILESTONE_STEPS * MILESTONE_STEPS ))
  if [ "${CURRENT_MILESTONE}" -gt 0 ]; then
    mkdir -p "${LOG_DIR}/milestones"
    MFILE="${LOG_DIR}/milestones/ckpt_step${CURRENT_MILESTONE}.pt"
    if [ ! -f "${MFILE}" ]; then
      cp "${CKPT}" "${MFILE}"
      echo "milestone -> ${MFILE}"
    fi
  fi
fi

# Prune rotating tier only when we added a new file.
if [ "${NEW_SNAPSHOT}" -eq 1 ]; then
  COUNT=$(ls -1 "${LOG_DIR}"/checkpoint_step*.pt 2>/dev/null | wc -l | tr -d ' ')
  if [ "${COUNT}" -gt "${MAX}" ]; then
    TO_DELETE=$((COUNT - PRUNE_TO))
    echo "prune: count=${COUNT} > max=${MAX}, deleting ${TO_DELETE} oldest"
    ls -1t "${LOG_DIR}"/checkpoint_step*.pt | tail -n "${TO_DELETE}" | xargs -r rm -v
  fi
fi
