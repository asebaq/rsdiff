#!/usr/bin/env bash
# Convenience launcher. Forwards all args to `rsdiff train` via accelerate.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${1:-configs/rsicd_text_128.yaml}"
shift || true

accelerate launch -m rsdiff.cli train --config "${CONFIG}" "$@"
