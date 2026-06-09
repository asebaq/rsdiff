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

Open-source reproduction checkpoint for the 2024 *RSDiff* thesis cascade —
text-to-satellite imagery, 256×256, T5-base conditioning. **FID 65.70** on
the full RSICD test split (N=1093), slightly better than the published 66.49.

- Code & methodology: https://github.com/asebaq/rsdiff
- Full report (curves, parity, costs): [`docs/REPORT.md`](https://github.com/asebaq/rsdiff/blob/main/docs/REPORT.md)
- Reproducibility runbook: [`docs/reproducibility.md`](https://github.com/asebaq/rsdiff/blob/main/docs/reproducibility.md)

## Highlights

| Metric | Value | Reference |
| --- | --- | --- |
| **FID** (cascade-256, N=1093, feature=2048) | **65.70** | thesis 66.49 |
| FID (feature=768) | 0.275 | — |
| CLIP-score (OpenAI ViT-B/32) | 0.278 | shuffled baseline 0.232 |
| CLIP delta | **+0.046** | text↔image alignment vs null |

`cond_scale=5.0` (winner of a CFG sweep on the best SR milestone, ep650).

## Architecture

| Stage | Params | Resolution | Conditioning |
| --- | --- | --- | --- |
| LR base UNet | 27.18 M | 128×128 | T5-base, `p_uncond=0.1` |
| SR UNet | 92.66 M | 128→256 | T5-base + LR image, `p_uncond=0.1` |

Both unets follow the `lucidrains/imagen-pytorch` cascade scaffolding.
Training: Adam, T=1000 DDPM steps. Path B — LR base trained 1000 epochs
first, then frozen at `ep700` (LR FID winner), then SR unet trained 1000
epochs on top using GT-lowres targets. Best SR milestone: ep650.

## Files

| File | Size | What |
| --- | --- | --- |
| `ckpt_sr_ep650_step89050.pt` | ~1.9 GB | merged cascade (LR base + SR) — load with the legacy trainer |
| `fid_result.json` | — | headline FID (feature=2048) |
| `fid_result_f768.json` | — | feature=768 head |
| `clip_result.json` | — | OpenAI CLIP ViT-B/32 score |
| `captions.txt` | — | 1093 RSICD-test captions matching the demo PNGs (sorted) |
| `samples/` | ~2 MB | 16 cherry-picked demo PNGs |

The two slimmer companion checkpoints (`ckpt_step95900.pt` LR base ep700,
slim SR-only milestones) are not uploaded here; build them from the
training command in the [reproducibility doc](https://github.com/asebaq/rsdiff/blob/main/docs/reproducibility.md).

## Usage

> The clean `diffusers`-native trainer is still on the roadmap. For now use
> the bundled legacy engine.

```bash
git clone https://github.com/asebaq/rsdiff
cd rsdiff
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,eval]"

# pull the checkpoint
hf download asebaq/rsdiff-sr-cascade-ep650 ckpt_sr_ep650_step89050.pt -o legacy/DDPM/ckpts/

# sample (1 batch of 16 captions from RSICD test split)
python legacy/DDPM/sample_grid.py \
  --log_dir legacy/DDPM/logs/full_sr_gdm \
  --data_root data/RSICD_optimal \
  --ckpt legacy/DDPM/ckpts/ckpt_sr_ep650_step89050.pt \
  --n 16 --cols 4 --batch 2 --cond_scale 5.0 \
  --img_sz 128 --sr_sz 256 --ts 1000 \
  --sr --split test --seed 17
```

## Training data

[RSICD](https://huggingface.co/datasets/arampacha/rsicd) — 10 921 paired
satellite images and natural-language captions, split 8 / 1 / 1 (train / val
/ test). Only the first caption per image (`sent1`) is used as
conditioning at train time, matching the thesis protocol.

## Intended use & limitations

**Intended use.** Research artefact for studying small-scale text-to-RS
generation, reproducibility of the 2024 thesis, and as a baseline for
future remote-sensing diffusion work.

**Out of scope.**

- Operational/commercial RS imagery synthesis — fidelity is too low.
- Producing imagery that could be mistaken for real, unaltered satellite
  data. The model is small (120 M params) and outputs are visibly
  diffusion-generated.
- Anything safety-critical (disaster response, surveillance, etc.).

**Known limitations.**

- **Overfit drift past SR ep650.** FID climbs after the bowl; v0 will ship
  weight decay, augmentation, val-FID, early stop, memorization probe.
- Single-caption conditioning — no caption-augmentation diversity.
- Pixel-space cascade — slower at sample time than a latent diffusion port
  (planned in v1).
- CFG sweep was scored at N=64 (rank-only, not headline).
- No memorisation probe yet — small training set + no augmentation means
  partial memorisation is possible.

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

@software{rsdiff2026,
  title  = {rsdiff: open-source diffusion models for remote sensing},
  author = {Sebaq, Ahmad},
  url    = {https://github.com/asebaq/rsdiff},
  year   = {2026},
}
```

## Acknowledgements

- `lucidrains/imagen-pytorch` for the cascade scaffolding.
- Nile University AI program for hosting the thesis work.
- vast.ai for cheap RTX 4090 hourly compute (~$166 total).
