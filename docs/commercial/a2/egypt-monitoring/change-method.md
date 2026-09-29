# Deterministic change baseline

Frozen 2026-09-02 before generating candidate polygons. This is a screening baseline,
not a construction classifier.

## Inputs

- selected AOI: `new_capital_core`;
- observed Sentinel-2 L2A bands B02, B03, B04, and B08 at 10 m;
- SCL validity masks from both dates, nearest-neighbour aligned to the 10 m grid;
- before: `S2A_36RUU_20240815_0_L2A`;
- after: `S2B_36RUU_20250815_0_L2A`.

## Fixed algorithm and parameters

1. Require exactly aligned before/after reflectance grids.
2. Exclude pixels marked no-data, saturated/defective, cloud shadow, medium/high cloud,
   cirrus, or snow/ice in either date.
3. For each reflectance band, calculate `after - before` over valid pixels. Remove its
   AOI-wide median and divide by `1.4826 × MAD`. A zero MAD is an error.
4. Combine the four normalized differences with root-mean-square magnitude.
5. Mark pixels with score **at least 5.0**.
6. Join eight-connected pixels and retain components of at least **6,000 m²**. This is
   the smallest frozen rubric footprint: a 300 m × 20 m road segment. Smaller objects
   are outside this experiment.
7. Export retained components as EPSG:4326 GeoJSON for review. Area is measured in the
   source projected CRS, EPSG:32636.

Confidence is mechanical: `high` when a component's 90th-percentile score is at least
8.0, otherwise `medium`. It does not mean the change is construction. Every feature
starts as `needs_review` and must be classified or rejected against the observable-
milestone rubric.

## Outputs and limitations

The method writes a continuous score GeoTIFF, filtered binary mask GeoTIFF, metrics
JSON, and editable candidate GeoJSON. Atmospheric difference, sand disturbance,
vehicle movement, landscaping, sensor differences, and residual misregistration can
all produce candidates. The method cannot determine schedule, completion, ownership,
cost, legality, or causation.
