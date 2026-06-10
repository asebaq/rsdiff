#!/usr/bin/env bash
# Bootstrap an rsdiff env on a fresh vast.ai instance.
#
# Pick a vast.ai template with a PyTorch 2.5 + CUDA 12.4 image (e.g.
# "PyTorch (cuda:12.4.1-cudnn-devel-ubuntu22.04)") or use this repo's
# Dockerfile directly via vast.ai's custom-image flow.
#
# Usage on the instance:
#   bash scripts/cloud/vast_setup.sh                       # rsdiff + legacy deps
#   bash scripts/cloud/vast_setup.sh --download-rsicd      # also fetch RSICD
#   bash scripts/cloud/vast_setup.sh --no-ddpm           # rsdiff only

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DOWNLOAD_RSICD=0
INSTALL_DDPM=1
for arg in "$@"; do
  case "${arg}" in
    --download-rsicd) DOWNLOAD_RSICD=1 ;;
    --no-ddpm) INSTALL_DDPM=0 ;;
    *) echo "unknown arg: ${arg}" >&2; exit 1 ;;
  esac
done

echo "== rsdiff vast.ai bootstrap =="
echo "  REPO_ROOT     = ${REPO_ROOT}"
echo "  install ddpm = ${INSTALL_DDPM}"
nvidia-smi | head -n 20 || echo "warn: nvidia-smi not found"

# Prefer uv if present (faster, isolates better); fall back to pip.
if command -v uv >/dev/null 2>&1; then
  PIP="uv pip"
else
  PIP="python -m pip"
  python -m pip install --upgrade pip
fi

${PIP} install -e ".[eval]"
if [ "${INSTALL_DDPM}" -eq 1 ]; then
  ${PIP} install -r ddpm/requirements_dgx.txt
fi

# Smoke imports.
python - <<'PY'
import importlib
mods = ["torch", "diffusers", "transformers", "accelerate", "rsdiff"]
try:
    import imagen_pytorch  # noqa: F401
    mods.append("imagen_pytorch")
except ModuleNotFoundError:
    pass
for m in mods:
    mod = importlib.import_module(m)
    print(f"{m:16s} {getattr(mod, '__version__', '?')}")
import torch
print("CUDA available:", torch.cuda.is_available(),
      "device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

mkdir -p data outputs

if [ "${DOWNLOAD_RSICD}" -eq 1 ]; then
  echo "== Fetching RSICD (HF mirror -> thesis-format CSV) =="
  # Materializes data/RSICD_optimal/{RSICD_images,dataset_rsicd.csv} in the
  # exact schema the legacy training scripts expect. Override the HF repo via
  # RSICD_HF_REPO and pass an authoritative split CSV via RSICD_SPLIT_CSV.
  REPO="${RSICD_HF_REPO:-arampacha/rsicd}"
  EXTRA_ARGS=()
  if [ -n "${RSICD_SPLIT_CSV:-}" ]; then
    EXTRA_ARGS+=(--split-csv "${RSICD_SPLIT_CSV}")
  fi
  python scripts/build_rsicd_csv.py --repo "${REPO}" --out data/RSICD_optimal "${EXTRA_ARGS[@]}"
  ln -sfn "$(pwd)/data/RSICD_optimal" ddpm/RSICD_optimal
  echo "  ddpm/RSICD_optimal -> data/RSICD_optimal"
fi

echo "== Done. Next: =="
echo "  # Legacy thesis repro (smoke run, ~2-4hr on A6000):"
echo "  bash ddpm/scripts/run_smoke.sh"
echo "  # Full LR-GDM run (multi-day):"
echo "  bash ddpm/scripts/lunch_training.sh"
echo "  # New rsdiff package (trainer is NotImplementedError until M1):"
echo "  accelerate config && rsdiff train --config configs/rsicd_text_128.yaml"
