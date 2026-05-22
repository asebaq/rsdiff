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

`rsdiff` ships two configs — same code path, two points on the size/quality
curve. **rsdiff1.5** is the optimized cascade we train and release; **rsdiff1**
is the paper-faithful (heavyweight) cascade kept for a head-to-head row.

=== "rsdiff1.5 (optimized — released)"

    | Stage | Resolution | Params | Conditioning | Optimizer |
    |---|---|---|---|---|
    | LR-GDM | 128² | 27.2M | T5 cross-attn (CFG p=0.1) | Adam |
    | SRDM | 256² | 92.7M | T5 cross-attn + LR image | Adam |
    | T5-base | — | 220M (frozen) | — | — |
    | **Cascade total** | | **119.9M** | | |

    - LR-GDM: `dim=128`, `cond_dim=256`, `dim_mults=(1,2,2,2)`, `num_resnet_blocks=0` — attention-only, the net that produced the thesis FID 66.49.
    - SRDM: `dim=128`, `cond_dim=512`, `dim_mults=(1,2,3,4)`, `num_resnet_blocks=(2,2,2,2)` — shrunk Efficient-U-Net (deepest stage 8×→4×).

=== "rsdiff1 (paper-faithful)"

    | Stage | Resolution | Params | Conditioning | Optimizer |
    |---|---|---|---|---|
    | LR-GDM | 128² | 260.8M | T5 cross-attn (CFG p=0.1) | Adafactor |
    | SRDM | 256² | 462.4M | T5 cross-attn + LR image | Adam |
    | T5-base | — | 220M (frozen) | — | — |
    | **Cascade total** | | **723.2M** | | |

    ≈ the abstract's stated 0.75 B parameters. `dim_mults=(1,2,4,8)`, deeper ResNet stacks. See [`configs/rsdiff1.yaml`](https://github.com/asebaq/rsdiff/blob/main/configs/rsdiff1.yaml) for the full topology and the note on the paper's internal 520M-vs-0.75B inconsistency.

All stages: 1000 timesteps, ε-prediction MSE, classifier-free guidance (`cond_drop_prob=0.1`).

!!! note "Code vs paper divergence"
    The thesis paper describes ~260M UNets with Adafactor. The training code that produced the reported **FID 66.49** actually ran a 27.2M attention-only base (`num_resnet_blocks=0`) on Adam defaults. `rsdiff1.5` encodes that real, lightweight base; `rsdiff1` encodes the paper-prose architecture. See [`legacy/README.md`](https://github.com/asebaq/rsdiff/blob/main/legacy/README.md) for the full table.

## Dataset

- **RSICD** ([Lu et al. 2018](https://arxiv.org/abs/1712.07835)) — 10,921 satellite images, 5 captions each.
- Splits: 8,734 train / 1,093 test / 1,094 val.
- 30 land-cover classes inferred from filename prefix (airport, baseball field, …).
- Captions normalised to lowercase + trailing ` .` to match the thesis tokenizer convention.

## Training recipe (rsdiff1.5)

The two stages train **sequentially (path B)**: train the 27.2M base to 1000
epochs, then seed and **freeze** it and train only the 92.7M SR UNet on top. The
base never sees the SR optimizer — this keeps the FID-66.49 base intact and
isolates SR quality.

| | LR-GDM (stage 1) | SRDM (stage 2, path B) |
|---|---|---|
| Batch size | 64 | 64 |
| Epochs | 1000 | 1000 |
| Optimizer | Adam, lr=1e-4 | Adam, lr=1e-4 |
| Warmup | 10k steps | 10k steps |
| Weight decay | 0 | 0 |
| Mixed precision | bf16 | bf16 |
| CFG drop prob | 0.1 | 0.1 |
| Base | trained from scratch | **seeded + frozen** from stage 1 |
| Hardware | 1× RTX 4090 (vast.ai) | 1× RTX 4090 |
| Wall-clock | ~119 h | _TBD_ |

## v1 design (planned)

Switch to a single-stage **latent diffusion** model: encode 256² images through a pretrained SD-v1.5 VAE (8× downsample → 32² latents), train UNet in latent space, decode back with the same VAE. Replaces the cascade with one UNet at 32², cuts compute ~10×, and is the modern baseline for FID < 50 on satellite domains.
