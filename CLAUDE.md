# rsdiff repository guidance

## Current state

RSDiff has a published reference release. The canonical baseline is `rsdiff1.5`: a
T5-conditioned Imagen-style 128→256 cascade with 119.9 M parameters. The released merged
SR epoch-650 checkpoint reaches FID-2048 65.70 and CLIP-score 0.278 on the complete
1,093-image RSICD test split at `cond_scale=5`.

The working sampler, trainer, and evaluation path is `ddpm/`. The `src/rsdiff/`
diffusers-native package remains an incomplete scaffold: training and sampling raise
`NotImplementedError`. Do not describe the scaffold as the released implementation and
do not delete `ddpm/` until replacement metric parity is demonstrated.

## Canonical documents

- `README.md`: public release overview and sampling quickstart.
- `docs/reproducibility.md`: complete training and evaluation runbook.
- `docs/REPORT.md`: methodology and published results.
- `docs/PLAN.md`: active research and commercial-validation plan.
- `docs/LICENSE_AUDIT.md`: commercial-rights inventory and blockers.

Private historical research notes under `.notes/` are not authoritative when they
conflict with tracked documentation.

## Conventions

- Python 3.10+. Type hints required for public APIs.
- Use `ruff` for lint; do not enable formatting implicitly.
- Run `pytest -q` for smoke tests.
- Do not commit checkpoints, datasets, generated images, logs, or `outputs/`.
- Preserve public metric protocol: RSICD test N=1,093, cascade 256², clean-FID
  Inception feature 2048, `cond_scale=5`.
- Keep generated imagery clearly identified as synthetic; never present it as observed
  or predicted geography.
- Config fields belong in `src/rsdiff/training/config.py` before YAML use.
- Do not use customer imagery for training unless explicit rights and consent exist.

## Commercial work

Do not start with a QGIS plugin. Follow `docs/PLAN.md`: validate demand, run manual
pilots, and build application-neutral analysis behavior before adding a thin QGIS UI.
RSDiff and commercial monitoring stay independent unless synthetic augmentation improves
performance on an untouched real-data test set.

## Local context

- Historical thesis source outside this repository is read-only.
- RSICD data belongs under `data/RSICD_optimal/` and is never committed.
- User-owned uncommitted changes must be preserved.
- Do not auto-commit.
