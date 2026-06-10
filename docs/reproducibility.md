# Reproducing the rsdiff baseline end-to-end

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

Local Mac (planning, plotting, light eval) — `uv` is preferred over `pip`.

```bash
git clone https://github.com/asebaq/rsdiff
cd rsdiff
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,eval]"
pytest -q                                   # smoke tests, no GPU/network
```

Box (vast.ai RTX 4090 — PyTorch 2.5 + CUDA 12.4 image works):

```bash
# On the freshly provisioned instance, after rsync'ing this repo to /workspace/rsdiff
cd /workspace/rsdiff
bash scripts/vast_setup.sh --download-rsicd     # installs rsdiff[eval] + legacy deps + RSICD
```

Cloud-side orchestration (search → launch → bootstrap → ssh):

```bash
bash scripts/vast_run.sh search
bash scripts/vast_run.sh launch <OFFER_ID>
bash scripts/vast_run.sh wait
bash scripts/vast_run.sh rsync
bash scripts/vast_run.sh bootstrap
```

## 1. Dataset — RSICD

Either grab the HuggingFace mirror or use the original ZIP. The legacy
trainer expects a `dataset_rsicd.csv` next to a `RSICD_images/` directory.

```bash
# from inside the repo
python scripts/build_rsicd_csv.py            # builds data/RSICD_optimal/{dataset_rsicd.csv,RSICD_images/}
```

Expected layout:

```
data/RSICD_optimal/
├── dataset_rsicd.csv     # filename, sent1..sent5, split
└── RSICD_images/
    ├── airport_1.jpg
    └── ...               # 10 921 JPEGs
```

`split` column values: `train` / `val` / `test` (1093 test images).

## 2. Train the LR base — 1000 ep

Runs `ddpm/models/Imagen_text_pytorch.py` end-to-end with milestone
snapshotting every 100 epochs.

```bash
# on box, inside tmux
bash scripts/vast_run.sh run 1000 full_lr_gdm     # nohup-detached, survives SSH drop
bash scripts/vast_run.sh logs full_lr_gdm          # follow logfile.log
```

Wall ~50 hr / ~$36 on a 4090. Outputs land in
`ddpm/logs/full_lr_gdm/{checkpoint.pt, milestones/*.pt, logfile.log}`.

Periodically pull milestones to local for safekeeping:

```bash
bash scripts/vast_run.sh pull-milestones full_lr_gdm
```

## 3. Train the SR unet — Path B, 1000 ep

Freezes the chosen LR base seed (we used `ep700` = LR FID winner) and
trains the SR unet only on GT-lowres targets.

```bash
# point LR_CKPT at the chosen base milestone (ep700 = step 95900)
LR_CKPT=ddpm/logs/full_lr_gdm/milestones/ckpt_step95900.pt \
  bash scripts/vast_run.sh run-sr 1000 full_sr_gdm
bash scripts/vast_run.sh logs full_sr_gdm
```

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
bash scripts/vast_run.sh snapshot full_sr_gdm
```

## 5. SR FID sweep — 18 milestones

The post-training FID sweep that produced the SR FID curve. Idempotent: if
the TSV already contains a row for a milestone, it is skipped.

```bash
bash scripts/sr_fid_sweep.sh           # symlinks data/ to ddpm/RSICD_optimal
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
tmux new -d -s postsweep  'bash scripts/sr_post_sweep.sh 2>&1 | tee postsweep.log'
tmux new -d -s afterpost  'bash scripts/sr_after_postsweep.sh 2>&1 | tee afterpost.log'
```

`sr_after_postsweep.sh` reads the SR TSV, picks the FID winner (ep650),
fires the CFG sweep + final 1093 + CLIP automatically.

## 6. CFG `cond_scale` sweep

Cheap-N picker (N=64) on the SR winner to choose `cond_scale` for the
headline. Reads `WINNER_STEP` from env or the watcher derives it.

```bash
WINNER_STEP=89050 bash scripts/sr_cfg_sweep.sh      # sweeps cs ∈ {1,2,3,4,5,6,8}
```

Wall ~10 hr / ~$7. Output: `outputs/fid_cfg_full_sr_gdm_step89050.tsv`.

## 7. Final 1093 + FID + CLIP

```bash
# pick winner_step + winner_cs from the two sweeps
WINNER_STEP=89050 CFG_SCALE=5 bash scripts/sr_final_1093.sh
GEN_DIR=ddpm/logs/full_sr_gdm/generated_images/final_test_step89050_cs5 \
  bash scripts/sr_clip_score.sh
```

Wall ~24 hr (sampling) + a couple of minutes (FID + CLIP). Cost ~$17.

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
bash scripts/sr_post_sweep_pull.sh         &       # waits on POST_SWEEP_DONE
bash scripts/sr_after_postsweep_pull.sh    &       # waits on POST_FINAL_DONE
```

Both rsync to `outputs/vast/...` with `--inplace --append --partial` so
they tolerate vast.ai's flaky SSH.

## 9. Plots

Once the pulls land, regenerate every figure in `docs/figures/` from the
small artifacts in `results/`:

```bash
source .venv/bin/activate
python scripts/make_plots.py
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
  past SR ep650 is visible in `docs/figures/sr_fid_curve.png`; v0 will
  ship mitigations.
- **`column -t` quirk.** Original `fid_sweep.sh` / `sr_cfg_sweep.sh` had a
  trailing `column -t` that fails on minimal vast.ai images and aborts
  the calling watcher under `set -e`. Patched to plain `cat` — if you
  fork an older revision, drop the `column` line.
- **In-memory image accumulation in `sample_grid.py`.** The original
  collected all PILs in RAM and wrote only at the end — a host migration
  mid-run wiped ~16 hr of work. The patch in this repo flushes PNGs per
  batch and resumes from existing files; recommended for any > 1 hr
  sampling job.
