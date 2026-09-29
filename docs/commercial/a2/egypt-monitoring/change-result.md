# Deterministic change result

Run 2026-09-02 on `new_capital_core` using the frozen method in
[`change-method.md`](change-method.md). Parameters were not tuned after viewing the
result.

## Output summary

- 448,337 valid 10 m pixels;
- 9,687 pixels exceeded the fixed score threshold before area filtering;
- 2,858 pixels remained after the fixed 6,000 m² component filter;
- 27 editable review candidates covering 285,800 m²;
- component area range: 6,000–39,300 m², median 9,500 m²;
- 25 mechanically high-confidence and 2 medium-confidence candidates.

Spatial inspection confirms that the score and vector mask align with the after image
and include clear large construction-related changes. The mask also includes plausible
false positives from landscaping, disturbed sand, and other surface activity. All
features therefore remain `unclassified_spectral_change` with `needs_review` status.
No feature is yet accepted as a milestone.

## Manual review result

All 27 candidates were reviewed against the frozen rubric on 2026-09-03. Decisions
are recorded in [`review-decisions.csv`](review-decisions.csv).

- accepted: 5 (3 large structural footprints and 2 road-network formations);
- rejected: 22;
- usefulness: **18.52%**, below the 80% acceptance gate.

The current baseline therefore fails. The dominant causes are responses on existing
road surfaces and landscaping, plus fragmentation of visually related construction
into components below the class-specific minimum area. Mechanical confidence was not
reliable: 20 of the 25 “high-confidence” candidates were rejected. This run must not
be rescued by changing its frozen threshold after review.

This result rejects `robust_multiband_rms_v1` as the deliverable-producing method. It
does not yet reject construction monitoring itself because the observed before/after
pair contains useful visible changes. Any second method must be declared as a new run,
aggregate related structural fragments, and suppress stable road/landscape surfaces
without using these 27 review decisions as its final evaluation set.

Local ignored outputs:

```text
data/commercial/egypt-monitoring/change/change-score.tif
data/commercial/egypt-monitoring/change/change-mask.tif
data/commercial/egypt-monitoring/change/change-candidates.geojson
data/commercial/egypt-monitoring/change/change-metrics.json
data/commercial/egypt-monitoring/change/change-preview.png
```

Next, review each polygon against both observed dates and assign one of the frozen
rubric classes or `rejected`, with a reason. That review—not the mechanical confidence
field—determines usefulness and false-positive rate.
