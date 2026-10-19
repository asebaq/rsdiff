# Reproducing the RSDiff baseline end to end

This doc walks through every step that produced the FID 65.70 / CLIP 0.278
number in [`REPORT.md`](REPORT.md), with exact commands and a per-phase
cost table. The whole pipeline runs on a single RTX 4090; total wall time
~233 hr / ~$166 at vast.ai's $0.717/hr.

Two boxes are typical: a **training box** for the 1000-ep LR + 1000-ep SR
runs (long, can be the same instance throughout) and a **FID box** for the
post-train sweep + headline (can re-use). All scripts are idempotent;
sentinels (`POST_SWEEP_DONE`, `POST_FINAL_DONE`) make the chain
resume-friendly across SSH drops and host reboots.

## 0. Env (local + box)

Dependencies are locked in `uv.lock`. Local development, sampling, and scoring:

```bash
git clone https://github.com/asebaq/rsdiff
cd rsdiff
uv sync                                     # Python 3.10+, dev group included
uv run pytest -q                            # smoke tests, no GPU/network
uv run ruff check . && uv run ruff format --check .
```

Training needs a CUDA box. The training image [`docker/Dockerfile`](../docker/Dockerfile)
targets any CUDA 12.4 host (vast.ai, RunPod, bare metal). Without Docker, on a freshly
provisioned instance (PyTorch 2.5 + CUDA 12.4 works) after rsync'ing this repo to
`/workspace/rsdiff`:

```bash
cd /workspace/rsdiff
bash scripts/cloud/vast_setup.sh --download-rsicd  # installs rsdiff[eval] + ddpm deps + RSICD
```

`ddpm/` pins its own newer-torch dependencies in `ddpm/requirements_dgx.txt` (torch 2.5 +
cu124, imagen-pytorch). On an aarch64 DGX Spark the thesis-era `torch==1.13.0+cu117`
pins do not install; use the NGC container instead:

```bash
docker run --gpus all -it --rm -v "$PWD":/workspace nvcr.io/nvidia/pytorch:24.10-py3 bash
pip install -r /workspace/ddpm/requirements_dgx.txt
```

Apple MPS is not supported for packaged sampling or FID (float64). The Mac is fine for
planning, plotting, and CLIP scoring.

Cloud-side orchestration (search → launch → bootstrap → ssh):

```bash
bash scripts/cloud/vast_run.sh search
bash scripts/cloud/vast_run.sh launch <OFFER_ID>
bash scripts/cloud/vast_run.sh wait
bash scripts/cloud/vast_run.sh rsync
bash scripts/cloud/vast_run.sh bootstrap
```

## 1. Dataset — RSICD

Either grab the HuggingFace mirror or use the original ZIP. Both the
`ddpm/` trainer and `rsdiff sample --rsicd` / `rsdiff eval fid` expect a `dataset_rsicd.csv` next to a `RSICD_images/` directory.

```bash
# from inside the repo
uv run --with datasets python scripts/build_rsicd_csv.py            # builds data/RSICD_optimal/{dataset_rsicd.csv,RSICD_images/}
```

Expected layout:

```
data/RSICD_optimal/
├── dataset_rsicd.csv     # filename, sent1..sent5, split
└── RSICD_images/
    ├── airport_1.jpg
    └── ...               # 10 921 JPEGs
```

`split` column values: `train` / `val` / `test` (1093 test images). The HF mirror splits
differ slightly (train/test/valid = 8730/1090/1090 versus 8734/1093/1094); pass
`--split-csv` to `build_rsicd_csv.py` to take splits from the original local CSV, which
the published N=1,093 protocol uses.

The real-image side of FID must also come from this builder. It re-encodes the mirror's
images as JPEG quality 95, and the published 65.70 was scored against those files. Scoring
the same generated bundle against the original RSICD JPEGs gives FID 70.15 instead. With
`--split-csv`, the builder's `sent1` captions also match the published bundle's
`captions.txt` exactly for `--seed 17`.

## 2. Train the LR base — 1000 ep

Runs `ddpm/models/Imagen_text_pytorch.py` end-to-end with milestone
snapshotting every 100 epochs.

```bash
# on box, inside tmux
bash scripts/cloud/vast_run.sh run 1000 full_lr_gdm  # nohup-detached, survives SSH drop
bash scripts/cloud/vast_run.sh logs full_lr_gdm      # follow logfile.log
```

Locally, without the cloud wrapper: `bash ddpm/scripts/run_training_local.sh`. Resume is
automatic from `<log_dir>/checkpoint.pt`.

Wall ~50 hr / ~$36 on a 4090. Outputs land in
`ddpm/logs/full_lr_gdm/{checkpoint.pt, milestones/*.pt, logfile.log}`.

Periodically pull milestones to local for safekeeping:

```bash
bash scripts/cloud/vast_run.sh pull-milestones full_lr_gdm
```

## 3. Train the SR unet — Path B, 1000 ep

Freezes the chosen LR base seed (we used `ep700` = LR FID winner) and
trains the SR unet only on GT-lowres targets.

```bash
# point LR_CKPT at the chosen base milestone (ep700 = step 95900)
LR_CKPT=ddpm/logs/full_lr_gdm/milestones/ckpt_step95900.pt \
  bash scripts/cloud/vast_run.sh run-sr 1000 full_sr_gdm
bash scripts/cloud/vast_run.sh logs full_sr_gdm
```

Locally: `bash ddpm/scripts/run_sr.sh`. Note that
`ddpm/models/Imagen_text_sr_pytorch.py` trains UNets in the order `(2, 1)`, SR first,
matching the thesis cascade-init protocol.

Wall ~85 hr / ~$61 on a 4090. Outputs:
`ddpm/logs/full_sr_gdm/{checkpoint.pt, milestones/*.pt, logfile.log}`.

The SR milestone files are *slim* — they only carry the SR unet weights,
not the frozen LR base. Use the merge utility to produce self-contained
cascade checkpoints for evaluation:

```bash
python scripts/merge_base.py \
  --base   ddpm/logs/full_lr_gdm/milestones/ckpt_step95900.pt \
  --slim   ddpm/logs/full_sr_gdm/milestones/ \
  --suffix _merged
```

## 4. Snapshot epoch grids (optional but useful)

`ddpm/sample_grid.py` makes a small 16-caption visual grid per
milestone. Useful for spotting overfitting visually. The
`snapshot`/`SNAP_WATCH` subcommand keeps the loop running while training
proceeds.

```bash
bash scripts/cloud/vast_run.sh snapshot full_sr_gdm
```

## 5. SR FID sweep — 18 milestones

The post-training FID sweep that produced the SR FID curve. Idempotent: if
the TSV already contains a row for a milestone, it is skipped.

```bash
bash scripts/eval/sr_fid_sweep.sh      # symlinks data/ to ddpm/RSICD_optimal
# inside, this launches the fidsweep tmux running scripts/fid_sweep.sh with:
#   STEPS=20550 27400 ... 137000   (ep150..ep1000 stride 50)
#   N=128 BATCH=2 FEATURE=2048 SIZE=256 SR=1
```

Watch progress via `tmux a -t fidsweep` or peek at the streaming TSV:

```bash
cat outputs/fid_curve_full_sr_gdm.tsv
```

Wall ~52 hr for 18 milestones (~2.9 hr/milestone). Cost ~$37.

When done, kick off the post-sweep watchers (grid500 backfill + sentinel):

```bash
tmux new -d -s postsweep  'bash scripts/cloud/sr_post_sweep.sh 2>&1 | tee postsweep.log'
tmux new -d -s afterpost  'bash scripts/cloud/sr_after_postsweep.sh 2>&1 | tee afterpost.log'
```

`sr_after_postsweep.sh` reads the SR TSV, picks the FID winner (ep650),
fires the CFG sweep + final 1093 + CLIP automatically.

## 6. CFG `cond_scale` sweep

Cheap-N picker (N=64) on the SR winner to choose `cond_scale` for the
headline. Reads `WINNER_STEP` from env or the watcher derives it.

```bash
WINNER_STEP=89050 bash scripts/eval/sr_cfg_sweep.sh  # sweeps cs ∈ {1,2,3,4,5,6,8}
```

Wall ~10 hr / ~$7. Output: `outputs/fid_cfg_full_sr_gdm_step89050.tsv`.

## 7. Final 1093 + FID + CLIP

Sample the headline bundle with the packaged CLI (defaults to the released checkpoint;
pass `--checkpoint` for your own merged milestone):

```bash
rsdiff sample --rsicd data/RSICD_optimal --split test --cond-scale 5 --seed 17 \
  --out-dir outputs/final_test_cs5
rsdiff eval fid  outputs/final_test_cs5 --rsicd data/RSICD_optimal --feature 2048 --size 256
rsdiff eval clip outputs/final_test_cs5
```

Without `-n`, `--rsicd` takes every caption in the split (1,093). Interrupted runs resume.
`rsdiff eval fid` is torchmetrics `FrechetInceptionDistance` with both sides resized to
256; it is the metric behind the published 65.70. `rsdiff eval clip` uses OpenAI CLIP
ViT-B/32 with a seed-17 shuffled-caption baseline.

The published bundle was produced with the historical cloud scripts, which call
`ddpm/sample_grid.py`:

```bash
WINNER_STEP=89050 CFG_SCALE=5 bash scripts/eval/sr_final_1093.sh
GEN_DIR=ddpm/logs/full_sr_gdm/generated_images/final_test_step89050_cs5 \
  bash scripts/eval/sr_clip_score.sh
```

Wall ~24 hr (sampling on a 4090) + a couple of minutes (FID + CLIP). Cost ~$17.

Outputs land in the gen dir:

```
final_test_step89050_cs5/
├── 0000_*.png … 1092_*.png        # 1093 PNGs (~127 MB)
├── captions.txt
├── fid_result.json                 # feature=2048 → 65.70
├── fid_result_f768.json            # feature=768  → 0.275
└── clip_result.json                # ViT-B/32 → 0.278 (shuffled 0.232)
```

## 8. Pull everything back to local

Two watcher scripts pull as soon as sentinels land — no babysitting:

```bash
# local Mac
bash scripts/cloud/sr_post_sweep_pull.sh      &    # waits on POST_SWEEP_DONE
bash scripts/cloud/sr_after_postsweep_pull.sh &    # waits on POST_FINAL_DONE
```

Both rsync to `outputs/vast/...` with `--inplace --append --partial` so
they tolerate vast.ai's flaky SSH.

## 9. Plots

Once the pulls land, regenerate every figure in `docs/figures/` from the
small artifacts in `results/`:

```bash
uv run python scripts/make_plots.py
```

Outputs:

```
docs/figures/
├── sr_fid_curve.png
├── cfg_sweep.png
├── lr_fid_curve.png
├── headline.png
└── sample_montage.png
```

## 10. Cost summary

| Phase | Wall | $ | Box |
| --- | --- | --- | --- |
| LR base 1000-ep train | ~50 hr | $36 | training |
| SR 1000-ep Path B train | ~85 hr | $61 | training |
| LR FID sweep (9 milestones, N=64) | ~3 hr | $2 | FID |
| SR FID sweep (18 milestones, N=128) | ~52 hr | $37 | FID |
| CFG sweep (7 scales, N=64) | ~10 hr | $7 | FID |
| Final 1093 cascade + FID + CLIP | ~24 hr | $17 | FID |
| Snap-loop grid generation (overlap) | ~10 hr | $7 | training |
| **Total** | **~234 hr** | **~$167** | |

All on a single RTX 4090 at $0.717/hr. The pipeline is built to run
unsupervised — sentinel watchers chain LR → SR → sweep → CFG → final → CLIP
end to end. The most expensive step that *can't* be deferred to a smaller
GPU is SR training; everything else fits comfortably on a 24 GB 4090.

## 11. Known reproducibility caveats

- **N=64 CFG FID has upward bias.** It is only used to rank scales here;
  treat its absolute numbers as ordinal, not metric.
- **Single caption per image (`sent1`).** RSICD has 5 captions per image;
  augmenting across captions changes both train and test signal. We took
  the thesis-conservative choice.
- **No augmentation, no weight decay.** Same as the thesis. Overfit drift
  past SR ep650 is visible in `docs/figures/sr_fid_curve.png`.
- **`column -t` quirk.** Original `fid_sweep.sh` / `sr_cfg_sweep.sh` had a
  trailing `column -t` that fails on minimal vast.ai images and aborts
  the calling watcher under `set -e`. Patched to plain `cat` — if you
  fork an older revision, drop the `column` line.
- **In-memory image accumulation in `sample_grid.py`.** The original
  collected all PILs in RAM and wrote only at the end — a host migration
  mid-run wiped ~16 hr of work. The patch in this repo flushes PNGs per
  batch and resumes from existing files; recommended for any > 1 hr
  sampling job. `rsdiff sample` behaves the same way.
- **Thesis code versus paper.** The thesis paper and its training scripts disagree on
  several hyperparameters (the code uses Adam with no warmup and no weight decay where
  the paper lists Adafactor, 10k warmup, and 0.01 decay). The code is authoritative: it
  produced the numbers. The unmodified thesis tree is kept outside this repository.
- **`ddpm/` patches.** Hard-coded `/home/asebaq/...` paths were made repo-relative,
  `.cuda()` calls became `--device {auto,cuda,mps,cpu}`, `--start_epoch` defaults to 0
  (resume comes from checkpoint presence), the epoch loss divides by `len(dl)`, and
  `evaluate_model.py` defaults to `RSICD_images`.
- **T5-base download.** About 900 MB on first run.

## Reproduction status

Packaged path (`rsdiff sample` / `rsdiff eval`) versus the published release. Complete
as of 2026-10-18 (PLAN B2).

| Check | Published | Packaged | Status |
| --- | --- | --- | --- |
| Release weights loaded into the `rsdiff1.5` preset | — | strict load, all keys matched | Matches |
| CLIP-score on the published 1,093-image bundle (`rsdiff eval clip`) | 0.27771 | 0.27771 (abs. diff 5e-7); shuffled 0.23219 | Matches |
| FID on the published 1,093-image bundle (`rsdiff eval fid`, CPU, torch 2.12, real images from `build_rsicd_csv.py --split-csv`) | 65.70 (65.698) | 65.688 | Matches (abs. diff 0.010) |
| Caption draw for `--rsicd`, seed 17, full test split | published `captions.txt` | all 1,093 identical, same order | Matches |
| One RSICD caption, seed 17, CPU: `rsdiff sample` versus `ddpm/sample_grid.py` on the same checkpoint | identical image | max abs. pixel diff 0 | Matches |

A fresh 1,093-image `rsdiff sample` run was not repeated. The packaged sampler produces
the same pixels as the sampler that made the published bundle, and the packaged metrics
reproduce the published scores on that bundle, so a re-run would only measure GPU
nondeterminism (about 24 GPU-hours, roughly $17 on an RTX 4090).
