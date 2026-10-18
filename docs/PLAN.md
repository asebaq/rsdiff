# RSDiff research plan

This is the canonical execution plan for RSDiff. Commercial monitoring validation
(formerly Track A) lives in the separate `geo-monitor-core` repository; neither project
depends on the other until evidence justifies integration.

## Invariants

- `rsdiff1.5` is the released reference baseline: a 119.9 M-parameter Imagen-style
  128→256 cascade with FID 65.70 and CLIP-score 0.278 on 1,093 RSICD test images.
- `ddpm/` remains the reference implementation until another implementation reproduces
  its sampling behavior and evaluation metrics.
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
- Keep `ddpm/` until parity is demonstrated.

Exit criterion: a new contributor can identify the canonical checkpoint, metrics,
commands, limitations, and commercial-license blockers from tracked documentation.

### B2. Independent reproduction

- Create a clean environment and download the published checkpoint.
- Generate deterministic samples with seed 17.
- Re-run FID-2048 and CLIP score on the complete test split.
- Record dependency versions, hardware, runtime, checkpoint digest, and metric deltas.

Exit criterion: sampling works from the published artifact and metrics match the
release within a documented tolerance.

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

1. Finish B1.
2. Run B2.
3. Run B3 only after B2 is stable.

## Kill conditions

- Imagery or dataset rights prohibit the intended training or redistribution.
- Synthetic data does not improve evaluation on untouched real imagery.
