---
license: apache-2.0
library_name: rsdiff
tags:
  - diffusion
  - text-to-image
  - remote-sensing
  - satellite-imagery
  - cascaded-diffusion
  - rsicd
datasets:
  - arampacha/rsicd
language:
  - en
pipeline_tag: text-to-image
inference: false
---

# rsdiff-sr-cascade-ep650

A T5-conditioned cascaded diffusion model for **text-to-satellite-image
generation** at 256×256, trained on RSICD.

- **FID 65.70** on the full RSICD test split (N=1093, torchmetrics
  Inception feature=2048, both sides resized to 256, `cond_scale=5.0`).
- **CLIP-score 0.278** (OpenAI ViT-B/32), with a +0.046 lift over a
  shuffled-caption null baseline.

Code & full tech report: https://github.com/asebaq/rsdiff
([REPORT.md](https://github.com/asebaq/rsdiff/blob/main/docs/REPORT.md)).

## Architecture

Two-stage Imagen-style cascade conditioned on a frozen T5-base text
encoder.

| Stage | Params | Resolution | Conditioning |
| --- | --- | --- | --- |
| LR base UNet | 27.18 M | 128×128 | T5-base, `p_uncond=0.1` |
| SR UNet | 92.66 M | 128 → 256 | T5-base + LR image, `p_uncond=0.1` |

Total **119.9 M params**.¹ Sampler: DDPM, T=1000 denoising steps.

¹ 119.9 M counts the two standalone UNets. Loaded inside the cascade, the SR UNet gains
low-resolution conditioning layers (105.0 M), so the cascade holds 132.2 M trainable
parameters, plus the frozen T5-base text encoder.

## Files

| File | Size | What it is |
| --- | --- | --- |
| `ckpt_sr_ep650_step89050.pt` | ~1.9 GB | Merged cascade weights (LR base + SR) |
| `samples/` | ~2 MB | 16 demo PNGs at 256² + 4×4 grid |
| `captions.txt` | 72 KB | 1093 RSICD-test captions matching the demo and FID PNGs |
| `fid_result.json` | — | Headline FID (Inception feature=2048) |
| `fid_result_f768.json` | — | Cross-comparison FID (feature=768) |
| `clip_result.json` | — | OpenAI CLIP ViT-B/32 score + shuffled-baseline null |

## Usage

```bash
pip install rsdiff
rsdiff sample --prompt "many buildings and green trees are around a playground ." -n 4 --out-dir generated
```

The CLI downloads `ckpt_sr_ep650_step89050.pt` from this repository at a pinned revision,
verifies its SHA-256, and writes 256×256 PNGs plus `captions.txt`. To sample RSICD test
captions instead, pass `--rsicd data/RSICD_optimal` (first caption per image, shuffled
with `--seed 17`). Score a bundle with `rsdiff eval fid` (requires `pip install
"rsdiff[eval]"`) and `rsdiff eval clip`. Full options:
[README](https://github.com/asebaq/rsdiff#use).

Sampling runs on CUDA when available and otherwise on CPU (slow: 2,000 denoising steps
per image). Apple MPS is not supported because imagen-pytorch 2.1.0 and the torchmetrics
FID use float64.

**Synthetic imagery.** Every output is synthetic. It is not an observation or a
prediction of any real place. Each PNG written by `rsdiff sample` carries `Software`,
`Description` (a synthetic-content notice), and `Prompt` PNG text chunks; keep them when
redistributing images.

## Training data

[RSICD](https://huggingface.co/datasets/arampacha/rsicd) — 10 921 paired
satellite images and natural-language captions, official 8/1/1
train/val/test split (1093 test). At training time the first caption per
image (`sent1`) is used as the conditioning text.

## Intended use & limitations

**Intended use.** Research artefact for studying small-scale text-to-RS
generation. Useful as a baseline for new remote-sensing diffusion work
and as a starting point for downstream tasks (augmentation, change-
detection priors).

**Out of scope.**

- Operational or commercial remote-sensing imagery synthesis — visual
  fidelity is well below modern web-scale models.
- Generating imagery intended to be mistaken for real satellite data.
- Anything safety-critical (disaster response, surveillance, defence).

**Known limitations.**

- **Overfit drift past SR ep650.** Validation FID climbs slightly after
  the bowl (see [REPORT.md §4](https://github.com/asebaq/rsdiff/blob/main/docs/REPORT.md)).
  No augmentation or weight decay; the train set is small (10 921 images).
- **Single-caption conditioning.** RSICD provides 5 captions per image;
  this run uses only the first.
- **Pixel-space cascade.** Slower at inference than a latent-diffusion
  port; a latent-space rewrite is on the project roadmap.
- **No memorisation probe.** Partial training-set memorisation is not
  ruled out — pHash audit is on the roadmap.

## License

The repository's authored code is Apache-2.0. Dataset, dependency, and derived-weight
rights are separate; see the repository's
[`LICENSE_AUDIT.md`](https://github.com/asebaq/rsdiff/blob/main/docs/LICENSE_AUDIT.md)
before commercial use.

## Citation

```bibtex
@article{sebaq2024rsdiff,
  title   = {RSDiff: remote sensing image generation from text using diffusion model},
  author  = {Sebaq, Ahmad and ElHelw, Mohamed},
  journal = {Neural Computing and Applications},
  volume  = {36},
  number  = {36},
  pages   = {23103--23111},
  year    = {2024},
  doi     = {10.1007/s00521-024-10363-3}
}
```
