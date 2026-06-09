# rsdiff

Open-source text-to-satellite-image diffusion models. Generate 256×256
overhead imagery from natural-language captions; trained on RSICD.

> **Headline.** `rsdiff1.5` — a 120 M-param T5-conditioned cascade — reaches
> **FID 65.70** and **CLIP-score 0.278** on the full RSICD test split
> (N=1093, cascade-256, `cond_scale=5`). Pretrained weights, configs,
> training and eval runbooks, and the full tech report are all open.
> Methodology, curves, and discussion in [`docs/REPORT.md`](docs/REPORT.md).
> See also [`docs/roadmap.md`](docs/roadmap.md).

Pretrained weights: [`asebaq/rsdiff-sr-cascade-ep650`](https://huggingface.co/asebaq/rsdiff-sr-cascade-ep650) on the HF Hub.
Project site: **https://asebaq.github.io/rsdiff**.

## What ships

Two configs sit at different points on the size/quality curve. Same code,
same data, different UNet topology:

| Config | Pipeline | Params | What it is |
|---|---|---|---|
| [`configs/rsdiff1.5.yaml`](configs/rsdiff1.5.yaml) | 27.2 M base + 92.7 M SR | **119.9 M** | Lightweight cascade. This is the released `rsdiff-sr-cascade-ep650` model. |
| [`configs/rsdiff1.yaml`](configs/rsdiff1.yaml) | 260.8 M base + 462.4 M SR | **723.2 M** | Larger cascade (≈ 0.75 B) — code path identical, training run pending. |

Both are Imagen-style: frozen **T5-base** text encoder → **LR base** UNet
(128²) → **SR** UNet (256²), classifier-free guidance, 1000-step DDPM.

## Results

Numbers below are on the **RSICD test split** (1,093 images), Inception
feature=2048, `cond_scale=5` (selected by a 7-scale CFG sweep on the best
SR milestone).

| Model | Res | FID ↓ | CLIP ↑ |
|---|---|---|---|
| **rsdiff1.5** (released) | 256² | **65.70** | **0.278** |

CLIP shuffled-caption null baseline = 0.232 → real-pair lift **+0.046**.
Per-epoch sweep + CFG sweep + headline panel:

![Headline](docs/figures/headline.png)

| Curve | Figure |
|---|---|
| SR FID-vs-epoch sweep (ep50–1000) | [`docs/figures/sr_fid_curve.png`](docs/figures/sr_fid_curve.png) |
| CFG `cond_scale` sweep on best SR milestone | [`docs/figures/cfg_sweep.png`](docs/figures/cfg_sweep.png) |
| LR base FID-vs-epoch sweep | [`docs/figures/lr_fid_curve.png`](docs/figures/lr_fid_curve.png) |
| Random 9-sample montage (RSICD test) | [`docs/figures/sample_montage.png`](docs/figures/sample_montage.png) |

Committed numerics: [`results/`](results/). Full discussion of curves,
CFG choice, overfit drift, and ablations: [`docs/REPORT.md`](docs/REPORT.md).
Project site (downloads + interactive views): [project site → Results](https://asebaq.github.io/rsdiff/results/).

## Quick start

```bash
git clone https://github.com/asebaq/rsdiff
cd rsdiff
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,eval]"
pytest -q                                   # smoke tests, no GPU/network
```

Sample 16 captions from the RSICD test split with the released cascade:

```bash
hf download asebaq/rsdiff-sr-cascade-ep650 ckpt_sr_ep650_step89050.pt -o legacy/DDPM/ckpts/
python legacy/DDPM/sample_grid.py \
  --log_dir legacy/DDPM/logs/full_sr_gdm \
  --data_root data/RSICD_optimal \
  --ckpt legacy/DDPM/ckpts/ckpt_sr_ep650_step89050.pt \
  --n 16 --cols 4 --batch 2 --cond_scale 5.0 \
  --img_sz 128 --sr_sz 256 --ts 1000 \
  --sr --split test --seed 17
```

`rsdiff train` (a `diffusers`-native trainer) is in active development;
until then, use the bundled cascade runner in `legacy/`. Full
end-to-end training + eval reproducibility runbook (env, dataset, sweeps,
sample times, costs): [`docs/reproducibility.md`](docs/reproducibility.md).
Cloud-GPU automation (vast.ai / RunPod): [`scripts/vast_run.sh`](scripts/vast_run.sh).

## Why

Generic web-scale diffusion models trained on natural photos degrade on
overhead views — road lattices, field textures, and building grids sit
outside the training distribution. RS-specific generation is useful for
augmentation, simulation, and change-detection priors, and the open-source
landscape is fragmented across one-off paper repos. `rsdiff` is one repo,
one CLI, with a clear path to more datasets, larger backbones, and richer
conditioning modes.

## License

Apache-2.0. See [`LICENSE`](LICENSE).

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

Built on `lucidrains/imagen-pytorch` (cascade scaffolding) and HuggingFace
`diffusers` (current rewrite target).
