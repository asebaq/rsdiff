# legacy/

Patched thesis-era reproduction baseline. Lifted from
`~/dev/ms/code/Generative-Models/` and modified to be runnable on a fresh
machine in 2026:

- Hard-coded `/home/asebaq/...` paths → repo-relative.
- `Imagen_text_sr_pytorch.py` `.cuda()` calls → `--device {auto,cuda,mps,cpu}`.
- `start_epoch` default 16 → 0 (resume happens via checkpoint presence).
- `loss / (len(dl) / bs)` → `loss / len(dl)`.
- `evaluate_model.py` `images_dir` default typo → `RSICD_images`.
- Bash launchers re-rooted from `$(dirname …)`.
- Added `requirements_dgx.txt` (torch 2.5+cu124, imagen-pytorch≥1.26.0) and
  `REPRODUCE.md` (end-to-end runbook for DGX Spark / vast.ai / Mac eval).

**The unmodified thesis code lives at `~/dev/ms/code/Generative-Models/`.**
Treat that tree as a historical artifact; do not edit it. Patches and any
follow-up belong here.

Why this exists in rsdiff at all: the thesis numbers (FID 66.49) are the
acceptance bar for rsdiff v0. This dir lets us re-run the original cascade
end-to-end whenever we need to confirm a regression or compare against the
new diffusers-based implementation.

## Layout

```
legacy/
  DDPM/
    Imagen_text_pytorch.py       # LR-GDM (text → 128x128)
    Imagen_text_sr_pytorch.py    # cascaded SR (128 → 256)
  evaluate_model.py              # CLIP zero-shot OA on test split
  lunch_training.sh              # SLURM-friendly LR-GDM launcher
  lunch_training_sr.sh           # SLURM-friendly SR launcher
  run_training_local.sh          # bare-Linux launcher
  run_training_mac.sh            # MPS launcher
  run_evaluation.sh              # eval wrapper
  requirements_dgx.txt           # newer-torch pin for aarch64 / vast.ai
  REPRODUCE.md                   # full runbook
```

## Quick start

```bash
cd legacy
# point at the existing RSICD copy in the ms repo:
export DATA_ROOT="$HOME/dev/ms/code/Generative-Models/RSICD_optimal"
python DDPM/Imagen_text_pytorch.py --data_root "$DATA_ROOT" --log_dir runs/lr_gdm
```

See `REPRODUCE.md` for full instructions (env pin, SRDM stage, eval).

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
