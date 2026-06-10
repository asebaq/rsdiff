# rsdiff training image — designed for vast.ai / RunPod / any CUDA 12.4 host.
# Base: official PyTorch image with CUDA 12.4 + cuDNN. Multi-platform amd64.

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl wget unzip ca-certificates \
        build-essential libgl1 libglib2.0-0 \
        openssh-server tmux htop \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Repo layout — package metadata, the rsdiff scaffold, the cascade implementation,
# orchestration scripts, configs, and the standalone ddpm requirements pin.
COPY pyproject.toml /workspace/pyproject.toml
COPY README.md /workspace/README.md
COPY LICENSE /workspace/LICENSE
COPY src/ /workspace/src/
COPY ddpm/ /workspace/ddpm/
COPY scripts/ /workspace/scripts/
COPY configs/ /workspace/configs/

# Install rsdiff (eval extras = clean-fid, transformers, torchmetrics[image])
# plus the imagen-pytorch + extras pinned by ddpm/requirements_dgx.txt.
RUN pip install --upgrade pip && \
    pip install -e ".[eval]" && \
    pip install -r ddpm/requirements_dgx.txt

# accelerate default config = single GPU bf16 — overridden by `accelerate config`.
RUN python - <<'PY'
import os, pathlib
cfg_dir = pathlib.Path(os.path.expanduser("~/.cache/huggingface/accelerate"))
cfg_dir.mkdir(parents=True, exist_ok=True)
(cfg_dir / "default_config.yaml").write_text("""compute_environment: LOCAL_MACHINE
distributed_type: NO
mixed_precision: bf16
num_machines: 1
num_processes: 1
use_cpu: false
""")
PY

CMD ["bash"]
