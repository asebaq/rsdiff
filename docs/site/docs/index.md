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

    128×128 base model (LR-GDM) + 256×256 super-resolution model (SRDM),
    both T5-conditioned. Latent-diffusion variant in v1.x.

-   :material-package-variant:{ .lg .middle } __Open weights__

    ---

    All checkpoints released on HuggingFace Hub under `asebaq/rsdiff-*`.
    Apache 2.0 licensed.

</div>

## Quickstart

```bash
pip install rsdiff
rsdiff sample --prompt "dense residential area near the port" --out sample.png
```

See the [Usage](usage.md) page for installation, sampling, and training instructions.

## Status

| | Version | Notes |
|---|---|---|
| Thesis reproduction (cascade, T5-base, FID ≤ 70) | v0.x | in progress |
| Latent diffusion (VAE-encoded, single stage 256²) | v1.0 | planned |
| RemoteCLIP text encoder | v1.x | planned |
| ControlNet (layout / segmentation conditioning) | v2.0 | planned |

See [`docs/roadmap.md`](https://github.com/asebaq/rsdiff/blob/main/docs/roadmap.md) for the full milestone list.

## Acknowledgments

Builds on `lucidrains/imagen-pytorch` (legacy thesis baseline) and the HuggingFace `diffusers` library (rsdiff package). The 2024 thesis was supervised at Nile University.
