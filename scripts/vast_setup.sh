#!/usr/bin/env bash
# Bootstrap an rsdiff env on a fresh vast.ai instance.
#
# Pick a vast.ai template with a PyTorch 2.5 + CUDA 12.4 image (e.g.
# "PyTorch (cuda:12.4.1-cudnn-devel-ubuntu22.04)") or use this repo's
# Dockerfile directly via vast.ai's custom-image flow.
#
# Usage on the instance:
#   bash scripts/vast_setup.sh
#   # or with RSICD download:
#   bash scripts/vast_setup.sh --download-rsicd

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DOWNLOAD_RSICD=0
for arg in "$@"; do
  case "${arg}" in
    --download-rsicd) DOWNLOAD_RSICD=1 ;;
    *) echo "unknown arg: ${arg}" >&2; exit 1 ;;
  esac
done

echo "== rsdiff vast.ai bootstrap =="
echo "  REPO_ROOT = ${REPO_ROOT}"
nvidia-smi | head -n 20 || echo "warn: nvidia-smi not found"

# Pin pip + install rsdiff with eval extras.
python -m pip install --upgrade pip
pip install -e ".[eval]"

# Smoke imports.
python - <<'PY'
import importlib
for m in ("torch", "diffusers", "transformers", "accelerate", "rsdiff"):
    mod = importlib.import_module(m)
    print(f"{m:14s} {getattr(mod, '__version__', '?')}")
import torch
print("CUDA available:", torch.cuda.is_available(),
      "device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

mkdir -p data outputs

if [ "${DOWNLOAD_RSICD}" -eq 1 ]; then
  echo "== Fetching RSICD =="
  # Public mirror on HuggingFace datasets; falls back to the original Google Drive
  # link if the env var RSICD_URL is set.
  python - <<'PY'
import os, pathlib, shutil
from huggingface_hub import snapshot_download
root = pathlib.Path("data/RSICD_optimal")
root.mkdir(parents=True, exist_ok=True)
url = os.environ.get("RSICD_URL")
if url:
    import urllib.request, zipfile, io
    print(f"downloading from {url}")
    data = urllib.request.urlopen(url).read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        z.extractall(root)
else:
    # HF mirror — repo id can be overridden via RSICD_HF_REPO env var.
    repo = os.environ.get("RSICD_HF_REPO", "arampacha/rsicd")
    print(f"snapshot_download {repo}")
    snapshot_download(repo, repo_type="dataset", local_dir=str(root))
print("done:", root)
PY
fi

echo "== Done. Next: =="
echo "  accelerate config           # one-time, picks GPU/precision"
echo "  rsdiff train --config configs/rsicd_text_128.yaml"
