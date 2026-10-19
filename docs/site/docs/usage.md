# Usage

## Installation

```bash
pip install rsdiff            # sampling + CLIP-score
pip install "rsdiff[eval]"    # adds torchmetrics for FID
```

Requires Python ≥ 3.10. Sampling runs on CUDA when available and otherwise on CPU (slow:
2,000 denoising steps per image). Apple MPS is not supported: imagen-pytorch 2.1.0 and
the torchmetrics FID use float64.

For development, clone the repository and run `uv sync`, then `uv run pytest -q`.

## Sampling

```bash
rsdiff sample --prompt "many buildings and green trees are around a playground ." -n 4 --out-dir generated
```

The first run downloads the released checkpoint (1.9 GB) from
[`asebaq/rsdiff-sr-cascade-ep650`](https://huggingface.co/asebaq/rsdiff-sr-cascade-ep650)
at a pinned revision and verifies its SHA-256. Outputs in `generated/`:

- one 256×256 PNG per image, named `<index>_<caption_slug>.png`
- `captions.txt`, one prompt per line in file order

Every image is synthetic, not an observation or prediction of a real place. Each PNG
carries `Software`, `Description` (a synthetic-content notice), and `Prompt` text chunks.

`--prompt` can be repeated. To sample RSICD captions instead (first caption per image,
shuffled with `--seed`, default 17):

```bash
rsdiff sample --rsicd data/RSICD_optimal --split test -n 16 --out-dir generated
```

| Option | Default | Meaning |
|---|---|---|
| `--cond-scale` | 5 | classifier-free guidance scale, both stages |
| `--batch-size` | 2 | images per sampling batch |
| `--lr-only` | off | stop at the 128×128 base stage |
| `--cpu` | off | force CPU |
| `--config` | `rsdiff1.5` | preset name (`rsdiff1.5`, `rsdiff1`) or YAML path |
| `--checkpoint` | released | checkpoint path; required for non-default configs |

Interrupted runs resume: batches whose PNGs already exist are skipped.

### Python API

```python
from rsdiff.cascade import download_release, generate, load_cascade
from rsdiff.config import load_preset

cfg = load_preset("rsdiff1.5")
trainer = load_cascade(cfg, download_release())
generate(trainer.sample, ["dense residential area near the port ."], "generated", cond_scale=5.0)
```

## Dataset

RSICD (10,921 image–caption pairs) belongs at `data/RSICD_optimal/` as
`RSICD_images/*.jpg` plus `dataset_rsicd.csv`. Build it from the Hugging Face mirror with
`scripts/build_rsicd_csv.py`; see
[`docs/reproducibility.md`](https://github.com/asebaq/rsdiff/blob/main/docs/reproducibility.md).

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

Training is not part of the `rsdiff` package yet: it runs through the thesis-era training code in [`ddpm/`](https://github.com/asebaq/rsdiff/tree/main/ddpm).

## Evaluation

Score a generated directory against the RSICD test split; both commands print JSON:

```bash
rsdiff sample --rsicd data/RSICD_optimal --cond-scale 5 --seed 17 --out-dir final_test_cs5
rsdiff eval fid final_test_cs5 --rsicd data/RSICD_optimal   # --feature 2048 --size 256
rsdiff eval clip final_test_cs5                            # clip_score, clip_score_shuffled, delta
```

FID is torchmetrics `FrechetInceptionDistance` (feature 2048, both sides resized to 256),
the metric behind the published 65.70. CLIP-score uses OpenAI CLIP ViT-B/32 with a
seed-17 shuffled-caption baseline. The historical cloud scripts (`scripts/eval/*.sh`)
sampled with `ddpm/sample_grid.py`.

Per-milestone FID sweeps and the CFG `cond_scale` ablation (historical cloud scripts):

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
