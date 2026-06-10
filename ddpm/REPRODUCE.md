# Reproducing RSDiff (thesis)

Two-stage cascaded diffusion on RSICD: LR-GDM (text → 128×128) then SRDM (→ 256×256).
Reported: **FID 66.49** with **~0.75B** params.

## 0. Data

Dataset already present at `RSICD_optimal/`:
- `RSICD_images/` — 10921 JPGs
- `dataset_rsicd.csv` — splits, captions (`sent1..sent5`), class label
- Splits: train / val / test as in the CSV

## 1. Env

### DGX Spark (aarch64, GB10, CUDA 12.4+)
Original `requirements.txt` pins `torch==1.13.0+cu117` (x86_64) — **will not run on Spark**.

```bash
# Easiest: use NGC container, then pip-install the rest.
docker run --gpus all -it --rm \
  -v "$PWD":/workspace \
  nvcr.io/nvidia/pytorch:24.10-py3 bash

cd /workspace
pip install -r requirements_dgx.txt
```

Or in a venv:
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -r requirements_dgx.txt
```

### Linux x86_64 GPU
Either `requirements_dgx.txt` (recommended — newer pins) or the legacy `requirements.txt`
if torch 1.13+cu117 still installs cleanly.

### Mac (sampling/eval only — training too slow)
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements_mac.txt
```

## 2. Train

### Stage 1 — LR-GDM (128×128, T5-base, ~1000 epochs)
```bash
./run_training_local.sh --img_sz 128 --ts 1000 --batch_sz 64 --epochs 1000
```

Resume picks up automatically from `<log_dir>/checkpoint.pt`.

### Stage 2 — SRDM (128→256 cascade)
```bash
python models/Imagen_text_sr_pytorch.py \
  --img_sz 128 --sr_sz 256 --ts 1000 --batch_sz 32 --epochs 1000
```

Note: `Imagen_text_sr_pytorch.py:118` trains UNets in order `(2, 1)` — SRDM first, then
LR-GDM. Matches thesis cascade-init protocol. Don't change without reading ch. 3.

Logs + TensorBoard runs land under `logs/<exp_name>/`.

## 3. Evaluate

```bash
./run_evaluation.sh
# CLIP zero-shot OA on the test split using *real* RSICD images (sanity check)
# For generated samples:
./run_evaluation.sh --images_dir /path/to/generated
```

The repo only ships zero-shot OA. **FID/IS/CLIP-score** required to match thesis numbers —
those are not in this repo and must be added (see `rsdiff` OSS rewrite roadmap).

## 4. Known repro blockers (already patched)

- Hard-coded `/home/asebaq/...` paths → repo-relative.
- `Imagen_text_sr_pytorch.py` `.cuda()` calls → `--device {auto,cuda,mps,cpu}`.
- `--start_epoch` default 16 → 0 (resume happens via checkpoint presence, not arg).
- `loss / (len(dl) / bs)` → `loss / len(dl)`.
- `evaluate_model.py` default `images_dir` typo → `RSICD_images`.

## 5. Open risks

- `imagen-pytorch` 1.1.4 (2022) likely incompatible with torch ≥ 2.1.
  `requirements_dgx.txt` bumps to `>=1.26.0`; model configs in
  `build_models()` may need small kwargs tweaks if the API drifted.
- T5-base download (~900MB) on first run.
- Single-GPU 24h SLURM time was the thesis budget — Spark single-node will be tighter
  than the original 8-GPU runs. Watch loss curves at the same step count, not wall time.
