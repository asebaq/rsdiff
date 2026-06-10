# ddpm/

Patched thesis-era reproduction baseline. Lifted from
`~/dev/ms/code/Generative-Models/` and modified to be runnable on a fresh
machine in 2026:

- Hard-coded `/home/asebaq/...` paths → repo-relative.
- `Imagen_text_sr_pytorch.py` `.cuda()` calls → `--device {auto,cuda,mps,cpu}`.
- `start_epoch` default 16 → 0 (resume happens via checkpoint presence).
- `loss / (len(dl) / bs)` → `loss / len(dl)`.
- `evaluate_model.py` `images_dir` default typo → `RSICD_images`.
- Bash launchers re-rooted from `$(dirname …)`.
- `utils/seed_everything.py`: dropped unused `import tensorflow` (TF was never
  exercised; the active `seed_everything()` only calls basic+torch).
- Added `requirements_dgx.txt` (torch 2.5+cu124, imagen-pytorch≥1.26.0),
  `REPRODUCE.md` (full runbook), and `run_smoke.sh` (env-tunable 10-epoch
  smoke for cloud-GPU bring-up).

**The unmodified thesis code lives at `~/dev/ms/code/Generative-Models/`.**
Treat that tree as a historical artifact; do not edit it. Patches and any
follow-up belong here.

Why this exists in rsdiff at all: the thesis numbers (FID 66.49) are the
acceptance bar for rsdiff v0. This dir lets us re-run the original cascade
end-to-end whenever we need to confirm a regression or compare against the
new diffusers-based implementation.

## Layout

```
ddpm/
  models/
    Imagen_text_pytorch.py       # LR base UNet (text → 128x128)
    Imagen_text_sr_pytorch.py    # SR UNet (128 → 256)
    Imagen_text_joint_pytorch.py # joint fine-tune entry
  scripts/
    lunch_training.sh            # SLURM-friendly LR launcher
    lunch_training_sr.sh         # SLURM-friendly SR launcher
    run_training_local.sh        # bare-Linux launcher
    run_training_mac.sh          # MPS launcher
    run_sr.sh                    # SR-stage launcher (path B)
    run_joint.sh                 # joint fine-tune launcher
    run_smoke.sh                 # 10-epoch smoke test
    run_evaluation.sh            # eval wrapper
    sample_grid.sh               # sample_grid.py wrapper
  utils/                         # logging, seeding, dataset stats
  sample_grid.py                 # caption-grid sampling entry point
  evaluate_model.py              # CLIP zero-shot OA on test split
  requirements_dgx.txt           # newer-torch pin for aarch64 / vast.ai
  REPRODUCE.md                   # full runbook
```

## Quick start

```bash
cd ddpm
# point at the existing RSICD copy in the ms repo:
export DATA_ROOT="$HOME/dev/ms/code/Generative-Models/RSICD_optimal"
python models/Imagen_text_pytorch.py --data_root "$DATA_ROOT" --log_dir runs/lr_gdm
```

See `REPRODUCE.md` for full instructions (env pin, SRDM stage, eval).

## Known divergences (code ≠ paper)

The thesis paper and the shipped training scripts disagree on several
hyperparameters. We treat the **code as authoritative** because FID 66.49
came from running it; the paper text appears post-hoc / aspirational. The
rsdiff v0 rewrite picks its own training recipe rather than inheriting
either side.

### Optimizer / schedule (Ch. 4 §4.2.1, Table 4.1)

| | Paper | Code (`models/Imagen_text_pytorch.py:122`, `ImagenTrainer(imagen=imagen)`) |
|---|---|---|
| LR-GDM optimizer | Adafactor | **Adam** (imagen-pytorch default) |
| SRDM optimizer | Adam | Adam |
| Learning rate | 1e-4 | 1e-4 |
| Warmup steps | 10000 | **0** |
| Weight decay | 0.01 | **0** |
| Adam betas | n/a | (0.9, 0.99) |

### Model size (Ch. 4 §4.2.1, §4.5)

| | Paper | Code (`build_models()` in `models/Imagen_text_pytorch.py:71`) |
|---|---|---|
| LR-GDM params | 260M | **27.2M** (measured at training start) |
| SRDM params | 260M | ~27M (analogous build) |
| RSDiff total | 0.75B | ~274M (27M+27M+220M T5-base) |

Cause: `Unet(..., num_resnet_blocks=0, ...)`. With zero ResNet stacks per
level the model is attention-only, an order of magnitude lighter than the
paper's claimed footprint. The 0.75B total in the abstract is overstated by
~3x as a result.

## Original sources

The full original repo (untouched) is at `~/dev/ms/code/Generative-Models/`.
The accompanying thesis PDF + LaTeX are at `~/dev/ms/docs/nu_msc_thesis/`.

```bibtex
@mastersthesis{sebaq2024rsdiff,
  title  = {RSDiff: A Diffusion-Based Framework for Text-to-Satellite-Image Generation},
  author = {Sebaq, Ahmad},
  school = {Nile University},
  year   = {2024}
}
```
