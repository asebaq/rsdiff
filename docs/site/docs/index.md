---
title: rsdiff
hide:
  - navigation
---

# rsdiff

**Diffusion models for remote-sensing imagery.**

Clean rewrite of the 2024 master's thesis [*RSDiff: A Diffusion-Based Framework for Text-to-Satellite-Image Generation*](https://github.com/asebaq/rsdiff) on top of HuggingFace `diffusers` + `accelerate`.

<div class="grid cards" markdown>

-   :material-image-multiple:{ .lg .middle } __Text-to-satellite__

    ---

    Generate 256×256 satellite imagery from natural-language captions.
    Trained on RSICD (10,921 images / 5 captions each).

-   :material-chart-line:{ .lg .middle } __FID 66.49 baseline__

    ---

    Reproduction target from the original thesis on the RSICD test split.
    See [Results](results.md) for the full table.

-   :material-cube-outline:{ .lg .middle } __Cascaded LR + SR__

    ---

    128×128 base (LR-GDM, 27.2M) + 256×256 super-resolution (SRDM, 92.7M),
    both T5-conditioned. 119.9M total. Latent-diffusion variant in v2.

-   :material-package-variant:{ .lg .middle } __Open weights__

    ---

    All checkpoints released on HuggingFace Hub under `asebaq/rsdiff-*`.
    Apache 2.0 licensed.

</div>

## Quickstart

```bash
git clone https://github.com/asebaq/rsdiff && cd rsdiff
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,eval]"

# sample from a trained cascade (legacy engine)
python legacy/DDPM/sample_grid.py \
  --log_dir legacy/DDPM/logs/full_lr_gdm --data_root data/RSICD_optimal \
  --n 16 --cols 4 --cond_scale 4.0 --split test --sr
```

`pip install rsdiff` and a one-line `rsdiff sample` CLI land with the first release. See the [Usage](usage.md) page for the full installation, sampling, and training runbook.

## Status

| Milestone | Tag | Status |
|---|---|---|
| Thesis cascade reproduction (rsdiff1.5, 119.9M, FID ≤ 70) | v0 | SR stage training |
| Paper-faithful cascade (rsdiff1, 723.2M) | v0 | optional |
| `diffusers`-native trainer (DDIM/DPM-Solver sampling) | v0.x | planned |
| Latent diffusion (VAE-encoded, single stage 256²) | v2 | planned |
| ControlNet (layout / segmentation conditioning) | v1 | planned |

See [`docs/roadmap.md`](https://github.com/asebaq/rsdiff/blob/main/docs/roadmap.md) for the full milestone list.

## Acknowledgments

Builds on `lucidrains/imagen-pytorch` (legacy thesis baseline) and the HuggingFace `diffusers` library (rsdiff package). The 2024 thesis was supervised at Nile University.
