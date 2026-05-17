#!/bin/bash
set -euo pipefail

# Run training locally on Mac (MPS / CPU).
# Ensure deps from requirements_mac.txt are installed in .venv/.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Starting training on Mac..."

if [ -d "${REPO_ROOT}/.venv" ]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.venv/bin/activate"
fi

python "${REPO_ROOT}/DDPM/Imagen_text_pytorch.py" --device mps "$@"

# Super-resolution stage:
# python "${REPO_ROOT}/DDPM/Imagen_text_sr_pytorch.py" --device mps "$@"
