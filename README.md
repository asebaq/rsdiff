# rsdiff

Open-source diffusion models for remote-sensing imagery — an open, reproducible
build of the master's thesis *"RSDiff: A Diffusion-Based Framework for
Text-to-Satellite-Image Generation"* (Nile University, 2024).

Generate 256×256 satellite/aerial imagery from a natural-language caption,
trained on RSICD. Weights, configs, and the full training runbook are open.

> **Status:** reproduction of the thesis cascade. The LR base + super-resolution
> stages train and sample today via the bundled `legacy/` engine; a clean
> `diffusers`-native trainer is on the roadmap. See [`docs/roadmap.md`](docs/roadmap.md).

Project site: **https://asebaq.github.io/rsdiff**

## What ships

Two configs encode two points on the size/quality curve. Same code path, same
data, different UNet topology:

| Config | Pipeline | Params | What it is |
|---|---|---|---|
| [`configs/rsdiff1.5.yaml`](configs/rsdiff1.5.yaml) | 27.2M base + 92.7M SR | **119.9M** | Optimized cascade. The lightweight base is the net that produced the thesis FID 66.49; SR is a shrunk Efficient-U-Net. |
| [`configs/rsdiff1.yaml`](configs/rsdiff1.yaml) | 260.8M base + 462.4M SR | **723.2M** | Paper-faithful cascade (≈ the abstract's "0.75 B"). |

Both are Imagen-style: frozen **T5-base** text encoder → **LR-GDM** (128²) →
**SRDM** (256²), classifier-free guidance, 1000-step DDPM.

## Results

Reported on the **RSICD test split** (1,093 images), `cond_scale=4.0`.

| Model | Res | FID ↓ | Notes |
|---|---|---|---|
| Thesis original (2024) | 256² | **66.49** | reference target |
| rsdiff1.5 (this repo) | 256² | _TBD_ | super-resolution run in progress |

Full table, FID-vs-epoch curve, and sample grids: [project site → Results](https://asebaq.github.io/rsdiff/results/).

## Quick start

```bash
git clone https://github.com/asebaq/rsdiff
cd rsdiff
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,eval]"
pytest -q                                   # smoke tests, no GPU/network
```

Sampling from a trained cascade (legacy engine, until the `diffusers` trainer lands):

```bash
python legacy/DDPM/sample_grid.py \
  --log_dir legacy/DDPM/logs/full_lr_gdm \
  --data_root data/RSICD_optimal \
  --n 16 --cols 4 --cond_scale 4.0 --split test --sr
```

> `pip install rsdiff` and `huggingface_hub` weight downloads land with the
> first tagged release. Until then, install from source and train (or pull
> checkpoints once published under `asebaq/rsdiff-*`).

Cloud-GPU runbook (vast.ai / RunPod): [`scripts/vast_run.sh`](scripts/vast_run.sh).
Full instructions: [project site → Usage](https://asebaq.github.io/rsdiff/usage/).

## Why

Image diffusion models trained on natural photos degrade on overhead views —
roads, field texture, and building grids are out of distribution. RS-specific
generation is useful for augmentation, simulation, and change-detection priors,
and the open-source landscape is fragmented across one-off paper repos. `rsdiff`
is one repo, one CLI, with a clear path to more datasets and conditioning modes.

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

Built on `lucidrains/imagen-pytorch` (cascade engine) and HuggingFace
`diffusers` (rewrite target). Full bibliography on the [Cite](https://asebaq.github.io/rsdiff/citation/) page.
