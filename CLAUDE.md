# rsdiff repository guidance

## Current state

RSDiff has a published reference release. The canonical baseline is `rsdiff1.5`: a
T5-conditioned Imagen-style 128→256 cascade with 119.9 M parameters (standalone UNets;
132.2 M as loaded in the cascade, where the SR UNet gains low-res conditioning layers).
The released merged SR epoch-650 checkpoint reaches FID-2048 65.70 and CLIP-score 0.278
on the complete 1,093-image RSICD test split at `cond_scale=5`.

The installable package `src/rsdiff/` (published to PyPI as `rsdiff`) owns sampling and
evaluation: `rsdiff sample` loads the released checkpoint through imagen-pytorch 2.1.0 and
`rsdiff eval fid|clip` reproduces the published metrics. Training lives in `ddpm/`, the
patched thesis-era training code; keep it until the package can train. Packaged
sampling and evaluation match `ddpm/` (PLAN B2 complete, see `docs/reproducibility.md`).

## Canonical documents

- `README.md`: public release overview and sampling quickstart.
- `docs/reproducibility.md`: complete training and evaluation runbook.
- `docs/REPORT.md`: methodology and published results.
- `docs/PLAN.md`: active research plan.
- `docs/LICENSE_AUDIT.md`: commercial-rights inventory and blockers.
- `CHANGELOG.md`: release notes; update with every user-visible change.

Private historical research notes under `.notes/` are not authoritative when they
conflict with tracked documentation.

## Conventions

- Python 3.10+. Type hints required for public APIs.
- `uv sync` for the dev environment; `uv.lock` is committed and CI runs `uv sync --locked`.
- CI enforces `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest -q`.
- Do not commit checkpoints, datasets, generated images, logs, or `outputs/`.
- Preserve public metric protocol: RSICD test N=1,093, cascade 256², torchmetrics
  `FrechetInceptionDistance` with Inception feature 2048 (`rsdiff.eval.fid`, not
  clean-fid), `cond_scale=5`, real images from `scripts/build_rsicd_csv.py --split-csv`
  (original RSICD JPEGs give FID 70.15, not 65.70); CLIP ViT-B/32 with the seed-17
  shuffled-caption baseline.
- Keep generated imagery clearly identified as synthetic; never present it as observed
  or predicted geography. `rsdiff.cascade.generate` writes this notice into every PNG.
- Config fields belong in `src/rsdiff/config.py` before YAML use; presets live in
  `src/rsdiff/configs/`.
- Do not use customer imagery for training unless explicit rights and consent exist.
- Releases: bump `rsdiff.__version__` (the single version source), add a CHANGELOG
  entry, tag `vX.Y.Z`; `.github/workflows/publish.yml` publishes via PyPI trusted
  publishing.

## Commercial work

Commercial monitoring lives in the separate `geo-monitor-core` repository. RSDiff and
commercial monitoring stay independent unless synthetic augmentation improves
performance on an untouched real-data test set.

## Local context

- Historical thesis source outside this repository is read-only.
- RSICD data belongs under `data/RSICD_optimal/` and is never committed.
- User-owned uncommitted changes must be preserved.
- Do not auto-commit.
