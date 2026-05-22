# Usage

## Installation

Install from source until the first tagged release; `pip install rsdiff` lands then.

=== "from source (current)"

    ```bash
    git clone https://github.com/asebaq/rsdiff
    cd rsdiff
    uv venv && source .venv/bin/activate
    uv pip install -e ".[dev,eval]"
    pytest -q                               # smoke tests, no GPU/network
    ```

=== "pip (after release)"

    ```bash
    uv pip install rsdiff                   # or: pip install rsdiff
    ```

Requires Python ≥ 3.10.

## Sampling

!!! warning "rsdiff trainer not yet implemented"
    The `rsdiff` CLI under `src/rsdiff/` raises `NotImplementedError` for `train`. Use the legacy thesis path below until v0 reproduction lands.

### Legacy thesis cascade

```bash
# Inside the repo, after running `bash scripts/vast_setup.sh --download-rsicd`:
python legacy/DDPM/sample_grid.py \
  --log_dir legacy/DDPM/logs/full_lr_gdm \
  --data_root data/RSICD_optimal \
  --n 16 --cols 4 \
  --cond_scale 4.0 --split test
```

Outputs go to `<log_dir>/generated_images/grid_step{N}/`:

- `00_<caption_slug>.png`, `01_…` per-image samples
- `_grid.png` 4×4 grid
- `captions.txt` prompts in order

### From a published checkpoint (after v0 release)

```python
import torch
from imagen_pytorch import Imagen, ImagenTrainer, Unet
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download("asebaq/rsdiff-lr-gdm-128", "checkpoint.pt")
unet = Unet(dim=128, cond_dim=256, dim_mults=(1,2,2,2),
            num_resnet_blocks=0,
            layer_attns=(False, True, True, True),
            layer_cross_attns=(False, True, True, True))
imagen = Imagen(text_encoder_name="t5-base", unets=unet,
                image_sizes=128, timesteps=1000, cond_drop_prob=0.1)
trainer = ImagenTrainer(imagen=imagen).cuda()
trainer.load(ckpt)
imgs = trainer.sample(texts=["dense residential area near the port"],
                      cond_scale=4.0, return_pil_images=True)
imgs[0].save("sample.png")
```

## Training (legacy path)

Reproducing `rsdiff1.5` on a single GPU. The cascade trains in two stages
(path B): the 27.2M base first, then the 92.7M SR UNet on the frozen base.

```bash
# 1. Provision a vast.ai GPU (see scripts/vast_run.sh)
export VAST_API_KEY=...                    # read from .env
./scripts/vast_run.sh launch              # picks a cheap RTX 4090
./scripts/vast_run.sh wait
./scripts/vast_run.sh rsync
./scripts/vast_run.sh bootstrap

# 2. Smoke check (10 epochs)
./scripts/vast_run.sh run 10 smoke_lr_gdm
./scripts/vast_run.sh logs

# 3. Stage 1 — LR-GDM base, 1000 epochs (~119 h on a 4090)
./scripts/vast_run.sh run 1000 full_lr_gdm

# 4. Stage 2 — SR UNet on the frozen base (path B)
./scripts/vast_run.sh run-sr 1000 full_sr_gdm

# 5. Pull artifacts back to local before destroying
./scripts/vast_run.sh pull
./scripts/vast_run.sh destroy
```

The script reads `VAST_API_KEY` from `.env`. Long runs go in a detached `tmux`
session (the SSH wrapper drops on long commands). See `scripts/vast_run.sh help`
for the full subcommand list.

## Evaluation

Zero-shot overall accuracy with off-the-shelf CLIP:

```bash
python -m rsdiff.eval.zeroshot_oa --images path/to/samples --captions captions.txt
```

FID + CLIP-score evaluation harness lands with v0; track [issue #TBD](https://github.com/asebaq/rsdiff/issues).

## Publishing inference weights

Strip optimizer state from a trained checkpoint before HuggingFace Hub upload:

```bash
python scripts/strip_checkpoint.py \
  outputs/vast/legacy/DDPM/logs/full_lr_gdm/checkpoint.pt \
  outputs/rsdiff_lr_gdm_v0.pt \
  --keep ema --fp16
# 435 MB -> 109 MB (75% reduction)
```
