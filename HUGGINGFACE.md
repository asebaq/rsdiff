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

- **FID 65.70** on the full RSICD test split (N=1093, Inception
  feature=2048, cascade-256, `cond_scale=5.0`).
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

Total ≈ **120 M params**. Sampler: DDPM, T=1000 denoising steps.

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
git clone https://github.com/asebaq/rsdiff
cd rsdiff
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,eval]"

# pull the checkpoint
hf download asebaq/rsdiff-sr-cascade-ep650 ckpt_sr_ep650_step89050.pt -o ddpm/ckpts/

# sample 16 captions from the RSICD test split
python ddpm/sample_grid.py \
  --log_dir ddpm/logs/full_sr_gdm \
  --data_root data/RSICD_optimal \
  --ckpt ddpm/ckpts/ckpt_sr_ep650_step89050.pt \
  --n 16 --cols 4 --batch 2 --cond_scale 5.0 \
  --img_sz 128 --sr_sz 256 --ts 1000 \
  --sr --split test --seed 17
```

A `diffusers`-native sampling path is on the project roadmap; for now the
bundled cascade runner (`ddpm/`) loads this checkpoint directly.

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

Apache 2.0 — see [`LICENSE`](https://github.com/asebaq/rsdiff/blob/main/LICENSE).

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
