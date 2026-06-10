#!/bin/bash
set -euo pipefail

# Run Evaluation Script.
# Uses the repo-local .venv if present.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Starting Evaluation (Zero-Shot Classification & OA)..."
echo "Note: This evaluates images at --images_dir against test-split labels."

PY="${REPO_ROOT}/.venv/bin/python"
[ -x "${PY}" ] || PY="python"

"${PY}" "${REPO_ROOT}/evaluate_model.py" "$@"

# Evaluate GENERATED images:
# "${PY}" "${REPO_ROOT}/evaluate_model.py" --images_dir /path/to/generated/images
