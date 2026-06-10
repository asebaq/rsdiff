#!/bin/bash
set -euo pipefail

# Run training locally on Linux (No Slurm).
# Ensure deps from requirements.txt are installed in .venv/.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Starting training locally on Linux..."

if [ -d "${REPO_ROOT}/.venv" ]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.venv/bin/activate"
fi

python "${REPO_ROOT}/models/Imagen_text_pytorch.py" "$@"

# Super-resolution stage:
# python "${REPO_ROOT}/models/Imagen_text_sr_pytorch.py" "$@"
