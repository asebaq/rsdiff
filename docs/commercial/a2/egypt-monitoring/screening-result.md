# Imagery screening result

Screened 2026-09-02 using Sentinel-2 L2A observations from 2024-08-15 and
2025-08-15. The images share tile 36RUU, CRS EPSG:32636, acquisition time within two
seconds, and exactly aligned output grids.

## Quality

| AOI | 2024 cloud | 2024 shadow | 2025 cloud | 2025 shadow |
|---|---:|---:|---:|---:|
| Mostakbal / eastern New Cairo | 0% | 0% | 0% | 0% |
| New Capital core | 0% | 0% | 0% | 0% |

Percentages are calculated over valid AOI pixels from the Sentinel scene
classification layer. They do not measure haze, registration error, or classification
accuracy.

## Selection

**Advance `new_capital_core`; retain `new_cairo_mostakbal` as backup.**

Both AOIs contain visible year-over-year change. The New Capital crop provides the
stronger first validation case because the changes are larger and more spatially
coherent at 10 m: expanding structure clusters, connected road/graded corridors, and
large prepared surfaces are visible. Mostakbal also shows development, but more of its
signal is distributed among smaller residential footprints near Sentinel-2's useful
limit.

This is a visual screening decision, not yet a scored change-detection result. The next
stage must generate candidate changes using a declared deterministic method and review
them against the frozen rubric.

## Reproduction

The scripts, inputs, commands, display scale, and limitations are documented in
[`README.md`](README.md). Local raster crops, SHA-256 crop manifest, SCL metrics, and
PNG quicklooks are stored under `data/commercial/egypt-monitoring/` and intentionally
excluded from Git.
