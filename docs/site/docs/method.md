# Method

## Overview

`rsdiff` v0 follows the cascaded text-to-image diffusion recipe of [Imagen (Saharia 2022)](https://arxiv.org/abs/2205.11487), adapted to remote-sensing data:

1. **T5-base text encoder** (frozen) maps the caption to a sequence of context embeddings.
2. **LR-GDM** (Low-Resolution Generation Diffusion Model) — UNet trained to denoise 128×128 images conditioned on the T5 embeddings.
3. **SRDM** (Super-Resolution Diffusion Model) — second UNet conditioned on both the T5 embeddings and the LR sample, upscaling 128² → 256².

Inference samples through both stages with [classifier-free guidance](https://arxiv.org/abs/2207.12598), `cond_scale ∈ [3, 5]`.

```
caption  ──▶ T5-base ──┬──▶ LR-GDM (128²)  ──▶  x_lr
                       │                              │
                       └──────────────────────────────┴──▶ SRDM (256²)  ──▶  x_hr
```

## Architectures

| Stage | Resolution | Params (measured) | Conditioning | Loss |
|---|---|---|---|---|
| LR-GDM | 128² | 27.2M | T5 cross-attn (CFG p=0.1) | ε-prediction MSE |
| SRDM | 256² | ~27M | T5 cross-attn + LR image | ε-prediction MSE |
| T5-base | — | 220M (frozen) | — | — |
| **Total** | | **~274M** | | |

UNet topology (both stages):

- `dim=128`, `cond_dim=256`, `dim_mults=(1, 2, 2, 2)`
- `num_resnet_blocks=0` — attention-only blocks, no ResNet stacks per level
- Attention + cross-attention from level 1 onward (`layer_attns=(False, True, True, True)`)
- 1000 timesteps, linear β schedule

!!! note "Code vs paper divergence"
    The thesis paper reports 260M-param UNets and Adafactor optimizer. The shipped training code uses `num_resnet_blocks=0` (which collapses each level to attention-only, ~27M params) and the imagen-pytorch default Adam optimizer. The reported FID 66.49 was produced by the code, not the paper recipe. See [`legacy/README.md`](https://github.com/asebaq/rsdiff/blob/main/legacy/README.md) for the full table.

## Dataset

- **RSICD** ([Lu et al. 2018](https://arxiv.org/abs/1712.07835)) — 10,921 satellite images, 5 captions each.
- Splits: 8,734 train / 1,093 test / 1,094 val.
- 30 land-cover classes inferred from filename prefix (airport, baseball field, …).
- Captions normalised to lowercase + trailing ` .` to match the thesis tokenizer convention.

## Training recipe (legacy thesis)

| | LR-GDM | SRDM |
|---|---|---|
| Batch size | 64 | 64 |
| Epochs | 1000 | 1000 |
| Optimizer | Adam, lr=1e-4, β=(0.9, 0.99) | Adam, lr=1e-4 |
| Warmup | 0 | 0 |
| Weight decay | 0 | 0 |
| CFG drop prob | 0.1 | 0.1 |
| Hardware | 1× A6000 (or A100) | 1× A6000 |
| Wall-clock | ~7 days | ~10 days |

## v1 design (planned)

Switch to a single-stage **latent diffusion** model: encode 256² images through a pretrained SD-v1.5 VAE (8× downsample → 32² latents), train UNet in latent space, decode back with the same VAE. Replaces the cascade with one UNet at 32², cuts compute ~10×, and is the modern baseline for FID < 50 on satellite domains.
