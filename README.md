# RSDiff: Remote Sensing Image Generation from Text<br><sub>Official PyTorch Implementation</sub>

[![Paper](https://img.shields.io/badge/DOI-10.1007%2Fs00521--024--10363--3-blue)](https://doi.org/10.1007/s00521-024-10363-3)
[![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/asebaq/rsdiff-sr-cascade-ep650)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)

<p align="center">
  <img src="docs/figures/sample_montage.png" width="780">
</p>

> [**RSDiff: Remote Sensing Image Generation from Text Using Diffusion Model**](https://doi.org/10.1007/s00521-024-10363-3)<br>
> [Ahmad Sebaq](https://github.com/asebaq), [Mohamed ElHelw](https://nu.edu.eg/)<br>
> Center for Informatics Science, Nile University<br>
> Neural Computing and Applications, 2024

A T5-conditioned cascaded diffusion model for text-to-satellite-image generation at 256×256, trained on RSICD. The released cascade reaches **FID 65.70** and **CLIP-score 0.278** on the full RSICD test split (N=1,093, `cond_scale=5`).

This repository contains:

* A PyTorch [implementation](ddpm/) of the two-stage Imagen-style cascade (T5-base → 128² base UNet → 256² SR UNet)
* Pre-trained 120 M-parameter cascade weights on [HF Hub](https://huggingface.co/asebaq/rsdiff-sr-cascade-ep650)
* A self-contained [sampling script](ddpm/sample_grid.py) for the released checkpoint
* FID + CLIP-score [evaluation pipeline](scripts/eval/) reproducing every number in [`docs/REPORT.md`](docs/REPORT.md)
* Cloud-GPU [orchestration scripts](scripts/cloud/) (vast.ai / RunPod) for end-to-end training and evaluation


## Setup

```bash
git clone https://github.com/asebaq/rsdiff
cd rsdiff
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,eval]"
```

Place the RSICD dataset (10,921 image–caption pairs) at `data/RSICD_optimal/`. See [`docs/reproducibility.md`](docs/reproducibility.md) for dataset preparation.


## Pre-trained Models

Released checkpoint, on the RSICD test split (1,093 images, Inception feature=2048, cascade-256, `cond_scale=5`):

| Model | Resolution | FID ↓ | CLIP ↑ | Params | Weights |
|---|---|---|---|---|---|
| **RSDiff Cascade** | 256² | **65.70** | **0.278** | 119.9 M | [🤗 `asebaq/rsdiff-sr-cascade-ep650`](https://huggingface.co/asebaq/rsdiff-sr-cascade-ep650) |

CLIP shuffled-caption null baseline = 0.232 → text-image alignment lift **+0.046**.

A larger 723 M-parameter configuration ([`configs/rsdiff1.yaml`](configs/rsdiff1.yaml)) shares the same code path; weights are not yet released.


## Sampling

Sample 16 captions from the RSICD test split with the released cascade:

```bash
hf download asebaq/rsdiff-sr-cascade-ep650 ckpt_sr_ep650_step89050.pt -o ddpm/ckpts/

python ddpm/sample_grid.py \
  --log_dir ddpm/logs/full_sr_gdm \
  --data_root data/RSICD_optimal \
  --ckpt ddpm/ckpts/ckpt_sr_ep650_step89050.pt \
  --n 16 --cols 4 --batch 2 --cond_scale 5.0 \
  --img_sz 128 --sr_sz 256 --ts 1000 \
  --sr --split test --seed 17
```


## Training

Two-phase decoupled-cascade training. LR base first, then SR on top of the frozen LR base:

```bash
# Phase 1 — 128² LR base UNet, 1000 epochs
bash ddpm/scripts/run_training_local.sh

# Phase 2 — 256² SR UNet (frozen LR base, GT-low-res conditioning), 1000 epochs
bash ddpm/scripts/run_sr.sh
```

For cloud-GPU runs (vast.ai / RunPod):

```bash
bash scripts/cloud/vast_setup.sh           # provisions env + dataset
bash scripts/cloud/vast_run.sh             # end-to-end train + snapshot
```

A `diffusers`-native trainer is in active development under [`src/rsdiff/`](src/rsdiff/) — until then, the cascade scripts above are the reference.


## Evaluation

FID over the full RSICD test split (N=1,093) on the best SR milestone with `cond_scale=5`:

```bash
bash scripts/eval/sr_final_1093.sh         # generate 1093-image bundle
bash scripts/eval/fid_sweep.sh             # FID at feature=2048 + feature=768
bash scripts/eval/sr_clip_score.sh         # CLIP score + shuffled-caption null
```

Per-milestone FID sweep, CFG `cond_scale` ablation, and the full tech report:

| Curve | Figure |
|---|---|
| SR FID-vs-epoch (ep50–1000) | [`docs/figures/sr_fid_curve.png`](docs/figures/sr_fid_curve.png) |
| CFG `cond_scale` sweep on best SR milestone | [`docs/figures/cfg_sweep.png`](docs/figures/cfg_sweep.png) |
| LR-base FID-vs-epoch sweep | [`docs/figures/lr_fid_curve.png`](docs/figures/lr_fid_curve.png) |

Committed numerics: [`results/`](results/). Full methodology, curves, and discussion: [`docs/REPORT.md`](docs/REPORT.md). End-to-end reproducibility runbook (env, training, sweeps, costs): [`docs/reproducibility.md`](docs/reproducibility.md).


## Acknowledgments

This implementation builds on `lucidrains/imagen-pytorch` for the cascade scaffolding and the HuggingFace `datasets` mirror of RSICD. The 2024 paper was developed at the Center for Informatics Science, Nile University.


## Citation

If you use this code or the released weights, please cite:

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


## License

Apache-2.0. See [`LICENSE`](LICENSE).
