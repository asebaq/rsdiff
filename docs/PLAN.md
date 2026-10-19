# RSDiff research plan

This is the canonical execution plan for RSDiff. Commercial monitoring validation
(formerly Track A) lives in the separate `geo-monitor-core` repository; neither project
depends on the other until evidence justifies integration.

## Invariants

- `rsdiff1.5` is the released reference baseline: a 119.9 M-parameter Imagen-style
  128→256 cascade with FID 65.70 and CLIP-score 0.278 on 1,093 RSICD test images.
- `ddpm/` is the training implementation; the `rsdiff` package covers sampling and
  evaluation. `ddpm/` stays until the package can train.
- Generated imagery is never represented as observation or prediction.
- Commercial work lives in `geo-monitor-core`. RSDiff enters a commercial workflow only
  after an untouched real-data evaluation shows downstream benefit.

## Track A — commercial validation

Moved to the separate `geo-monitor-core` repository, including its plan, evidence,
experiments, and kill conditions.

## Track B — RSDiff

### B1. Baseline and governance — complete 2026-08-28

- Preserve and name the released cascade as the reference baseline.
- Reconcile repository guidance with the published release.
- Verify exact sampling and evaluation commands.
- Record code, model, dependency, and dataset licensing separately.
- Keep `ddpm/` as the training implementation.

Exit criterion: a new contributor can identify the canonical checkpoint, metrics,
commands, limitations, and commercial-license blockers from tracked documentation.

### B2. Independent reproduction — complete 2026-10-18

- Published checkpoint downloads from a pinned revision and passes its SHA-256 check.
- `rsdiff sample` reproduces `ddpm/sample_grid.py` pixel for pixel (seed 17).
- `rsdiff eval` reproduces FID 65.70 (65.688) and CLIP 0.278 (0.27771) on the published
  1,093-image bundle.

Evidence, hardware and deltas: `docs/reproducibility.md`, "Reproduction status".

### B3. Commercial-relevance experiment

- Generate urban/construction prompts with the current checkpoint.
- Audit realism, diversity, semantic alignment, and memorization.
- Measure how a pretrained building detector behaves on generated images.
- Do not add spatial conditioning unless this cheap screen is promising.

If promising, train mask/layout conditioning to create images paired with known labels.
Compare a detector trained on real data against the same detector trained on real plus
synthetic data, using an untouched real test set.

Exit criterion: synthetic augmentation improves a predeclared real-world metric without
degrading important subgroups. FID improvement alone does not qualify.

## Current order

1. Run B3.

## Kill conditions

- Imagery or dataset rights prohibit the intended training or redistribution.
- Synthetic data does not improve evaluation on untouched real imagery.
