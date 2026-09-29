# Three-date persistence v3 result

Run and reviewed 2026-09-03 on the previously unseen
`new_capital_east_holdout` AOI using 2024-08-15, 2025-08-15, and 2026-08-10
Sentinel-2 L2A observations. The method was frozen in
[`change-method-v3.md`](change-method-v3.md) before inspection.

## Result

- v2 proposals entering persistence filter: 11;
- proposals retained by v3: 8;
- accepted after three-date visual review: 5;
- rejected: 3;
- usefulness: **62.5%**, below the fixed 80% gate;
- accepted: 1 clearing/grading area, 3 road formations, and 1 structural footprint;
- registration: 2025→2024 = -0.25 row/+0.25 column; 2026→2025 = 0/0.

Tracked decisions are in [`review-decisions-v3-holdout.csv`](review-decisions-v3-holdout.csv).

## Finding

Persistence did remove three v2 candidates, but persistence is not specificity. Long-
lived disturbed sand and an existing linear surface also persisted. Two candidates at
the AOI boundary could not be classified reliably. V3 therefore performs worse than
v2's 76.47% usefulness and fails the acceptance gate.

Do not tune the persistence fraction on this holdout. The next improvement should not
be another spectral threshold. It needs explicit object context: exclude an inward AOI
edge buffer, compare proposed roads with before-date linear features, and use higher-
resolution confirmation for building-level claims. Sentinel-2 remains useful for
screening large development zones, not autonomous milestone classification.
