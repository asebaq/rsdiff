# Method

## Overview

RSDiff is a two-stage Imagen-style cascaded diffusion model conditioned on a frozen T5-base text encoder, trained on the RSICD remote-sensing image–caption dataset.

1. **T5-base text encoder** (frozen) maps the caption to a sequence of context embeddings.
2. **LR base UNet** denoises 128×128 noise into a low-resolution image, conditioned on the T5 embeddings via cross-attention.
3. **SR UNet** denoises 256×256 noise, conditioned on the T5 embeddings *and* the low-resolution sample from stage 2.

Both stages use 1000-step DDPM with ε-prediction MSE and classifier-free guidance (`cond_drop_prob=0.1`). The cascade output is the SR UNet's 256² sample.

```
caption  ──▶ T5-base ──┬──▶ LR base UNet (128²)  ──▶  x_lr
                       │                                   │
                       └───────────────────────────────────┴──▶ SR UNet (256²)  ──▶  x_hr
```

## Architecture

| Stage | Resolution | Params | Conditioning | Block scaffold |
|---|---|---|---|---|
| **LR base UNet** | 128² | 27.18 M | T5-base cross-attn, `p_uncond=0.1` | `dim=128`, `cond_dim=256`, `dim_mults=(1,2,2,2)`, attention-only |
| **SR UNet** | 128 → 256 | 92.66 M | T5-base + LR image, `p_uncond=0.1` | `dim=128`, `cond_dim=512`, `dim_mults=(1,2,3,4)`, `num_resnet_blocks=(2,2,2,2)` |
| **T5-base** | — | 220 M (frozen) | — | — |
| **Cascade total** | | **119.9 M** | | |

A larger 723 M-parameter configuration ([`configs/rsdiff1.yaml`](https://github.com/asebaq/rsdiff/blob/main/configs/rsdiff1.yaml)) shares the same code path; weights are not yet released.

## Dataset

[RSICD](https://huggingface.co/datasets/arampacha/rsicd) — 10,921 paired satellite images and natural-language captions, 30 inferred land-cover classes from filename prefix, official 8/1/1 train/val/test split (1,093 test). Captions are short, declarative, and mostly land-cover oriented. The first caption per image (`sent1`) is used as the conditioning text at training time. No augmentation, no caption shuffling.

## Training

Two-phase decoupled-cascade training: LR base first, SR second on top of a *frozen* LR base.

| | LR base (stage 1) | SR UNet (stage 2) |
|---|---|---|
| Resolution | 128² | 128 → 256 |
| Batch size | 64 | 64 |
| Epochs | 1000 | 1000 |
| Optimizer | Adam, lr=1e-4 | Adam, lr=1e-4 |
| Warmup | 10k steps | 10k steps |
| Weight decay | 0 | 0 |
| Mixed precision | bf16 | bf16 |
| CFG drop probability | 0.1 | 0.1 |
| LR base | trained from scratch | **seeded + frozen** from stage 1 |
| LR conditioning | — | ground-truth low-resolution (no noise aug) |
| Hardware | 1× RTX 4090 | 1× RTX 4090 |
| Wall-clock | ~119 h | ~115 h |

Milestones are snapshotted every 50 SR epochs. A `merge_base.py` utility reinjects the frozen LR base into each slim SR checkpoint so every milestone is a self-contained cascade for evaluation. The released checkpoint is the merged SR epoch-650 milestone (`step=89050`).

## Inference

Cascade sampling is sequential: the LR base UNet samples a 128² image conditioned only on text; the SR UNet samples a 256² image conditioned on both text and the LR sample. Classifier-free guidance applies to both stages with a shared `cond_scale`; null-conditioning is taken from the `p_uncond=0.1` channel learnt at training time.

| | DDPM steps | Time per image (RTX 4090, batch=8) |
|---|---|---|
| LR base (128²) | 1000 | ~110 s |
| SR UNet (128 → 256²) | 1000 | ~120 s |
| **Full cascade** | 2000 | ~230 s |

A 1000-step sampler is the bottleneck. A DDIM / DPM-Solver fast-sampler port is on the roadmap.

## Design choices

**Pixel-space cascade.** Two stages in pixel space, no VAE encoder. Slower at inference than a latent-diffusion port, but lets us train the entire pipeline from scratch on a single RSICD-scale dataset without a separate VAE pre-training step.

**Decoupled stages.** Stage 2 is trained on ground-truth low-resolution images, not on stage-1 samples. This keeps the SR stage's training distribution clean, and isolates the LR vs SR quality contributions for ablation.

**T5 cross-attention, not CLIP.** Following Imagen, the text encoder is frozen T5-base (not CLIP). T5 captions are linguistically richer than CLIP's contrastive objective allows, which matters more on the captioned-by-humans RSICD distribution than on web-scraped photo corpora.

**Classifier-free guidance, both stages.** A single `cond_scale` parameter at inference time controls the strength of text adherence in both stages. The CFG sweep ([Results](results.md#cfg-cond_scale-ablation)) shows the bowl bottoms at `cs=5`.

## Limitations

- **Overfit drift past SR ep650.** No augmentation or weight decay, train set is small (10,921 images). FID climbs from 156.73 → 167.46 between SR ep650 → ep1000.
- **Single-caption conditioning.** RSICD provides 5 captions per image; this run uses only `sent1`.
- **No memorisation probe.** Partial training-set memorisation is not ruled out — a perceptual-hash audit is on the roadmap.
- **Pixel-space cascade is inference-slow.** A latent-diffusion rewrite is on the roadmap.
