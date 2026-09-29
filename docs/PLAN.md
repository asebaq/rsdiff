# RSDiff research and commercial validation plan

This is the canonical execution plan. The released model and a possible commercial
geospatial product are separate tracks. Neither track depends on the other until
evidence justifies integration.

## Invariants

- `rsdiff1.5` is the released reference baseline: a 119.9 M-parameter Imagen-style
  128→256 cascade with FID 65.70 and CLIP-score 0.278 on 1,093 RSICD test images.
- `ddpm/` remains the reference implementation until another implementation reproduces
  its sampling behavior and evaluation metrics.
- Operational monitoring uses observed imagery and deterministic analysis. Generated
  imagery is never represented as observation or prediction.
- No QGIS plugin is built until two manual pilots expose the same repeated interaction.
- RSDiff enters a commercial workflow only after an untouched real-data evaluation
  shows downstream benefit.

## Track A — commercial validation

### A1. Evidence and access — desk artifacts complete 2026-08-29

Deliverables:

- 30 independent, dated demand signals: public questions, issues, jobs, tenders, case
  studies, competitors, and price anchors;
- 40 reachable prospective users or buyers;
- interview guide centered on the respondent's last completed monitoring job;
- one example deliverable containing before/after evidence, editable polygons,
  confidence, and an Arabic/English report.

Online validation comes first: compare candidate jobs, select one narrow claim, and
produce a public-data manual demo. The current comparison and demo protocol are in
[`commercial/a1/online-candidate-review.md`](commercial/a1/online-candidate-review.md).

Interviews are postponed until the online demo passes its acceptance test. Commercial
validation still requires 5 interviews describing the same recurring job, 3
quantifying its cost, 3 sharing a sanitized example, 2 identifying the buyer, and 1
accepting a manual or paid pilot. Desk research and a demo cannot establish willingness
to pay or access to internal workflows.

The initial desk artifacts are tracked under [`commercial/a1/`](commercial/a1/).
The phase remains open until the interview criteria above are met; organizations in the
prospect list have not been contacted or endorsed the concept.

### A2. Concierge pilot

Test construction-progress or unauthorized-building monitoring without a plugin:

```text
before/after GeoTIFF
  → registration
  → narrow change inference
  → polygon extraction
  → analyst correction
  → GeoPackage/GeoJSON + bilingual report
```

Measure analyst time, false positives, missed changes, correction time, and whether the
output satisfies the customer's acceptance process. Continue if two pilots show a
repeatable workflow and at least 50% analyst-time reduction.

### A3. Product shell

Build algorithms as an application-neutral `geo-monitor-core`. Add a thin QGIS plugin
only for layer selection, AOI selection, job status, polygon review, and export. Keep
CLI, web, ArcGIS, and Blender adapters possible without moving core behavior into UI
code.

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
2. Complete the online candidate review and one-week public-data validation demo; do
   not implement a plugin.
3. Run B2.
4. If the online demo passes, begin interviews and seek one A2 manual pilot.
5. Run B3 only after B2 is stable.
6. Decide independently whether each track continues.

## Kill conditions

- Customers buy imagery and analysis only as an inseparable incumbent service.
- Imagery or dataset rights prohibit the intended processing or redistribution.
- Affordable imagery cannot support the required accuracy.
- Buyers value the result but will not fund it separately.
- Required security, procurement, or support exceeds the project's capacity.
- Synthetic data does not improve evaluation on untouched real imagery.
