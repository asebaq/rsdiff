# Results

!!! warning "Reproduction in progress"
    The v0 reproduction run is currently training on a single RTX A6000. Final numbers will replace the placeholders in this page once the 1000-epoch run completes.

## Quantitative

FID and CLIP-score reported on the **RSICD test split** (1,093 images), with `cond_scale=4.0`, 1000-step DDPM sampling, 5 captions per image.

| Model | Resolution | FID ↓ | CLIP-score ↑ | Zero-shot OA ↑ | Notes |
|---|---|---|---|---|---|
| Thesis original (2024) | 256² | **66.49** | — | — | from `~/dev/ms/code/Generative-Models/` |
| rsdiff v0 (this repo, legacy) | 256² | _TBD_ | _TBD_ | _TBD_ | reproduction target ≤ 70 |
| rsdiff v1 (latent diffusion) | 256² | _TBD_ | _TBD_ | _TBD_ | planned, target ≤ 50 |

Metric definitions:

- **FID** — Fréchet Inception Distance, Inception-V3 features. We additionally report RS-FID (Inception fine-tuned on AID) once the eval harness lands.
- **CLIP-score** — cosine similarity between CLIP text-embedding(caption) and CLIP image-embedding(sample).
- **Zero-shot OA** — overall accuracy of an off-the-shelf CLIP ViT-L/14 classifier on generated samples, using the 30 RSICD class labels. Tests whether generated content is semantically classifiable as the intended land-cover type.

## Qualitative samples

_Sample grids will be added here once the v0 reproduction lands. Layout sketch:_

```
+--------------------------------------------------+
| caption: "many planes are parked next to a       |
|           long building in an airport"           |
|                                                  |
|   +-------+   +-------+   +-------+   +-------+  |
|   |       |   |       |   |       |   |       |  |
|   | s1    |   | s2    |   | s3    |   | s4    |  |
|   |       |   |       |   |       |   |       |  |
|   +-------+   +-------+   +-------+   +-------+  |
+--------------------------------------------------+
```

Per-caption, 4 samples generated at `cond_scale ∈ {3, 4, 5, 7}` to visualise guidance sensitivity.

## Inference cost

| Stage | Steps | Time / image (A6000) | Time / image (4090) |
|---|---|---|---|
| LR-GDM 128² | 1000 | ~45 s | _TBD_ |
| SRDM 256² | 1000 | _TBD_ | _TBD_ |
| Full cascade | 2000 | ~2 min | _TBD_ |

DPM-Solver++ 25-step sampler in v1 should bring this under 5 s per image.

## Training cost (this reproduction)

| Stage | GPU | Wall-clock | Cost (vast.ai spot) |
|---|---|---|---|
| LR-GDM 1000 ep | RTX A6000 | ~185 h | ~$74 |
| SRDM 1000 ep | RTX A6000 | ~250 h (projected) | ~$100 |

Numbers will be backfilled when training completes.
