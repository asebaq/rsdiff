# Usage

## Installation

```bash
git clone https://github.com/asebaq/rsdiff
cd rsdiff
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,eval]"
```

Requires Python ≥ 3.10. The `[eval]` extra adds `clean-fid`, `transformers`, and `torchmetrics[image]` for the FID + CLIP-score pipeline.

## Dataset

Place the RSICD dataset (10,921 image–caption pairs) at `data/RSICD_optimal/`:

```bash
bash scripts/cloud/vast_setup.sh --download-rsicd     # downloads HF mirror + writes the local-format CSV
```

The local layout is `data/RSICD_optimal/imgs/*.jpg` + `dataset_rsicd.csv`. See [`docs/reproducibility.md`](https://github.com/asebaq/rsdiff/blob/main/docs/reproducibility.md) for the full dataset-prep walkthrough.

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

Outputs land in `<log_dir>/generated_images/grid_step89050/`:

- `00_<caption_slug>.png`, `01_…` per-image samples
- `_grid.png` 4×4 grid
- `captions.txt` prompts in order

### Python API

```python
import torch
from imagen_pytorch import Imagen, ImagenTrainer, Unet
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download("asebaq/rsdiff-sr-cascade-ep650",
                       "ckpt_sr_ep650_step89050.pt")
base = Unet(dim=128, cond_dim=256, dim_mults=(1,2,2,2),
            num_resnet_blocks=0,
            layer_attns=(False, True, True, True),
            layer_cross_attns=(False, True, True, True))
sr = Unet(dim=128, cond_dim=512, dim_mults=(1,2,3,4),
          num_resnet_blocks=(2,2,2,2))
imagen = Imagen(text_encoder_name="t5-base",
                unets=(base, sr),
                image_sizes=(128, 256),
                timesteps=1000, cond_drop_prob=0.1)
trainer = ImagenTrainer(imagen=imagen).cuda()
trainer.load(ckpt)
imgs = trainer.sample(texts=["dense residential area near the port"],
                      cond_scale=5.0, return_pil_images=True)
imgs[0].save("sample.png")
```

## Training

Reproducing the released cascade on a single GPU. Two-phase decoupled training: 27 M-param LR base first, then 92 M-param SR UNet on the frozen LR base.

### Single GPU (local)

```bash
# Phase 1 — 128² LR base UNet, 1000 epochs (~119 h on RTX 4090)
bash ddpm/scripts/run_training_local.sh

# Phase 2 — 256² SR UNet (frozen LR base, GT-low-res conditioning), 1000 epochs
bash ddpm/scripts/run_sr.sh
```

### Cloud (vast.ai / RunPod)

```bash
export VAST_API_KEY=...                            # read from .env
bash scripts/cloud/vast_run.sh launch              # picks a cheap RTX 4090
bash scripts/cloud/vast_run.sh wait
bash scripts/cloud/vast_run.sh rsync
bash scripts/cloud/vast_run.sh bootstrap

# smoke check (10 epochs)
bash scripts/cloud/vast_run.sh run 10 smoke_lr_gdm

# stage 1 — LR base, 1000 epochs
bash scripts/cloud/vast_run.sh run 1000 full_lr_gdm

# stage 2 — SR UNet, 1000 epochs
bash scripts/cloud/vast_run.sh run-sr 1000 full_sr_gdm

# pull artefacts back to local before destroying
bash scripts/cloud/vast_run.sh pull
bash scripts/cloud/vast_run.sh destroy
```

The script reads `VAST_API_KEY` from `.env`. Long runs go in a detached `tmux` session. See `bash scripts/cloud/vast_run.sh help` for the full subcommand list.

A `diffusers`-native trainer under [`src/rsdiff/`](https://github.com/asebaq/rsdiff/tree/main/src/rsdiff) is in active development; until then, the cascade scripts above are the reference.

## Evaluation

FID over the full RSICD test split (N=1,093) on the best SR milestone with `cond_scale=5`:

```bash
bash scripts/eval/sr_final_1093.sh         # generate 1093-image bundle
bash scripts/eval/fid_sweep.sh             # FID at feature=2048 + feature=768
bash scripts/eval/sr_clip_score.sh         # CLIP score + shuffled-caption null
```

Per-milestone FID sweeps and the CFG `cond_scale` ablation:

```bash
bash scripts/eval/sr_fid_sweep.sh          # SR ep150–1000 stride 50, N=128
bash scripts/eval/sr_cfg_sweep.sh          # cond_scale ∈ {1,2,3,4,5,6,8}, N=64
```

Numerics drop into `results/`. See [Results](results.md) for the full numbers + curves.

## Publishing inference weights

Strip optimizer state from a trained checkpoint before HuggingFace Hub upload:

```bash
python scripts/strip_checkpoint.py \
  outputs/vast/ddpm/logs/full_sr_gdm/checkpoint.pt \
  outputs/rsdiff_sr_v0.pt \
  --keep ema --fp16
# 1.9 GB → ~470 MB (75% reduction)
```

Merge the frozen LR base back into a slim SR checkpoint for a self-contained cascade:

```bash
python scripts/merge_base.py \
  --sr  outputs/vast/ddpm/logs/full_sr_gdm/checkpoint.pt \
  --base outputs/vast/ddpm/logs/full_lr_gdm/checkpoint.pt \
  --out ckpt_sr_ep650_step89050.pt
```
