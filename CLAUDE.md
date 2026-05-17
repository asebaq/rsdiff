# rsdiff — Claude Code notes

## What this repo is

Open-source diffusion models for remote sensing imagery. Clean rewrite of the
2024 master's thesis ("RSDiff", Nile University) on top of HuggingFace
`diffusers` + `accelerate`.

Status: **v0 scaffold**. Training loop is `NotImplementedError`. M1 scope is
pending a literature review pass (see "Open work" below).

## Layout

```
src/rsdiff/
  datasets/      # RSICD adapter implemented; others stubbed
  models/        # diffusers UNet2DConditionModel + T5 builders
  pipelines/     # cascaded LR+SR, conditional, control-guided (stubs)
  training/      # config (OmegaConf+dataclass), trainer (stub)
  eval/          # zeroshot_oa.py (functional); fid/clip_score TBD
  conditioning/  # text/class/mask/lowres (stubs)
  cli.py         # rsdiff {train,sample,eval}
configs/         # YAML run configs
scripts/         # vast_setup.sh, train.sh
docker/          # CUDA 12.4 + torch 2.5 image
tests/           # smoke tests, no GPU/network
docs/roadmap.md  # v0 / v0.x / v1 / v1.x / v2 milestones
```

## Where the legacy code + data live

- **Unmodified thesis code**: `~/dev/ms/code/Generative-Models/`. Treat as a
  read-only historical artifact. Do not edit.
- **Patched reproduction baseline**: `legacy/` in this repo. Same imagen-pytorch
  scripts plus repo-relative paths, device auto-detect, fixed eval bugs,
  newer `requirements_dgx.txt`, and a `REPRODUCE.md` runbook.
- **Thesis docs**: `~/dev/ms/docs/nu_msc_thesis/` (LaTeX sources + `Thesis.pdf`).
- **RSICD dataset**: `~/dev/ms/code/Generative-Models/RSICD_optimal/`
  (10 921 images, `dataset_rsicd.csv`). Symlink or copy into `data/` here:
  `ln -s ~/dev/ms/code/Generative-Models/RSICD_optimal data/RSICD_optimal`.

## Conventions

- Python 3.10+. Type hints required for public APIs.
- `ruff` for lint, no formatter — `ruff format` only if explicitly enabled.
- Tests live alongside `tests/`, run with `pytest -q`.
- Do not commit checkpoints, datasets, generated images, or `outputs/`.
- Match diffusers' naming where possible (`UNet2DConditionModel`,
  `DDPMScheduler`, etc.) to keep the mental model small.
- Configs are OmegaConf-merged on top of `RunConfig` dataclass defaults —
  add new fields to `src/rsdiff/training/config.py` first, then to the YAML.

## Workflow

```bash
pip install -e ".[dev,eval]"
pytest -q                                   # smoke tests
rsdiff train --config configs/rsicd_text_128.yaml   # NotImplementedError until M1
```

Cloud GPU runbook is `scripts/vast_setup.sh` (vast.ai-friendly, also works on
RunPod). Container image is `docker/Dockerfile` (CUDA 12.4 + torch 2.5).

## Open work

- **M1 scope blocked on lit review.** Need to confirm: keep imagen-style
  cascade vs latent diffusion (LDM); RSFID vs Inception-FID; which
  conditioning extensions to ship in v1. See `docs/roadmap.md` open questions.
- **Trainer not implemented.** `src/rsdiff/training/trainer.py` raises.
  Acceptance bar: ≤ FID 70 on RSICD test split (thesis got 66.49).
- **FID + CLIP-score metrics not implemented.** Only `zeroshot_oa` works.
- **No checkpoints released.** Will go on HF Hub under `asebaq/rsdiff-*`.

## Behaviour preferences for Claude

- Caveman mode is on at the user level — keep responses terse, fragments OK.
  Code, commit messages, PRs, and security warnings stay in normal English.
- Prefer editing existing files over creating new ones.
- Don't add docstrings/comments unless the *why* is non-obvious.
- For UI/notebook changes: actually run them before reporting done.
- Don't auto-commit. The user runs commits manually unless asked.
- When the thesis is referenced, the authoritative artifact is
  `~/dev/ms/docs/nu_msc_thesis/Thesis.pdf` and `Abstract.tex`.
