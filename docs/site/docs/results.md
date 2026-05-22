# Results

!!! warning "Reproduction in progress"
    The LR base run (rsdiff1.5) is complete; the super-resolution stage is training on a single RTX 4090. The headline 256² numbers below are placeholders until the SR run finishes and the full-test-split eval runs.

## Headline FID

The paper-comparable number is computed **once**, on the final checkpoint: generate the full **RSICD test split** (1,093 images, `cond_scale=4.0`, 1000-step DDPM), score FID against the 1,093 real test images with Inception-V3 features (`feature=2048`) at the cascade output resolution.

| Model | Resolution | FID ↓ | CLIP-score ↑ | Zero-shot OA ↑ | Notes |
|---|---|---|---|---|---|
| Thesis original (2024) | 256² | **66.49** | — | — | reference target |
| rsdiff1.5 (this repo) | 256² | _TBD_ | _TBD_ | _TBD_ | SR run in progress, target ≤ 70 |
| rsdiff1 (paper-faithful) | 256² | _TBD_ | _TBD_ | _TBD_ | optional head-to-head |
| rsdiff v2 (latent diffusion) | 256² | _TBD_ | _TBD_ | _TBD_ | planned, target ≤ 50 |

## FID-vs-epoch (model selection)

A cheap, **small-N** FID curve (N=64 generations per checkpoint, base 128² only) is tracked during training for early-stop / checkpoint selection. These values are biased high by the small sample count — only the **trend** is meaningful, not the absolute number, and they are **not** comparable to the headline above.

| Epoch | FID (N=64, 128²) |
|---|---|
| 100 | 276.8 |
| 200 | 227.1 |
| 300 | 221.7 |
| 400 | 218.9 |
| 500 | 211.1 |
| 600 | 208.9 |
| … | _curve continues to ep1000_ |

Monotone descent, flattening past ep500 — healthy, no divergence.

!!! danger "No selection on test"
    Checkpoint selection and early-stop use the **val** split only. The test split is touched once, for the headline number, to avoid leakage.

Metric definitions:

- **FID** — Fréchet Inception Distance, Inception-V3 features (`feature=2048`). RS-FID (Inception fine-tuned on AID) reported once the eval harness lands.
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

1000-step DDPM, RTX 4090:

| Stage | Steps | Time / image |
|---|---|---|
| LR-GDM 128² | 1000 | ~15 min (batched: ~110 s/img at batch 8) |
| SRDM 256² | 1000 | _TBD_ |
| Full cascade | 2000 | _TBD_ |

The 1000-step sampler is the bottleneck. A DPM-Solver++ / DDIM 25-step sampler (in the `diffusers` rewrite) should cut this ~40× — under 5 s per image.

## Training cost (rsdiff1.5)

vast.ai on-demand RTX 4090 @ ~\$0.56/h:

| Stage | GPU | Epochs | Wall-clock | Cost |
|---|---|---|---|---|
| LR-GDM | RTX 4090 | 1000 | ~119 h | ~\$66 |
| SRDM (path B) | RTX 4090 | 1000 | _TBD_ | _TBD_ |

Numbers backfilled when the SR run completes.
