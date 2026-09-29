# Egypt construction-monitoring validation

Started 2026-09-02. This experiment screens two public-data areas, then takes the
better-observed area through the full manual-demo workflow described in
[`../../a1/online-candidate-review.md`](../../a1/online-candidate-review.md).

## Question

Can repeat public imagery support a traceable, reviewable report of macro construction
milestones in eastern Cairo?

This is not a claim that a project is on schedule, complete, compliant, or properly
funded. Only changes visible in observed imagery may be reported.

## Candidate AOIs

The polygons in [`aoi.geojson`](aoi.geojson) are deliberately small screening areas,
not administrative boundaries.

| ID | Working name | Approximate extent | Reason for inclusion |
|---|---|---:|---|
| `new_cairo_mostakbal` | Mostakbal / eastern New Cairo growth area | 46 km² | Large residential and road development with desert contrast and continuing build-out |
| `new_capital_core` | New Capital CBD–government corridor | 47 km² | Large structures, roads, and earthworks; NASA has already demonstrated multi-year Landsat observation of the capital |
| `new_capital_east_holdout` | New Capital eastern holdout | 47 km² | Adjacent unseen area reserved for three-date v3 evaluation |

The Mostakbal location is anchored by the public Google Maps place result at
approximately 30.0496° N, 31.6140° E. The New Capital location is cross-checked against
[NASA Earth Observatory](https://science.nasa.gov/earth/earth-observatory/constructing-egypts-new-capital-153237/),
which compares Landsat observations from August 2017 and August 2024. These anchors do
not certify project boundaries or developer ownership.

## Frozen observable-milestone rubric

The rubric is fixed before date-pair selection to reduce hindsight bias.

### Include

1. **Major clearing or grading** — contiguous surface change at least 0.05 km² and
   visible across multiple adjacent 10 m pixels.
2. **Large structural footprint** — a new or materially expanded high-contrast roof,
   slab, or structure cluster at least 0.01 km².
3. **Road-network formation** — a new connected graded or paved segment at least 300 m
   long and approximately 20 m wide.
4. **Large laydown or stockpile change** — contiguous appearance, disappearance, or
   material relocation at least 0.02 km².

### Exclude

- individual villas, small buildings, narrow local roads, façades, floors, interiors,
  utilities, landscaping detail, occupancy, and construction quality;
- percentage-complete, schedule, cost, ownership, permit, or legality claims;
- one-pixel changes, cloud or shadow edges, seasonal vegetation, atmospheric haze,
  sand brightness changes, and map-label changes;
- details visible only in generated or super-resolved imagery.

### Confidence

- `high`: clear spatially coherent change, visible in true color and at least one
  independent band/index view, away from cloud/shadow.
- `medium`: coherent likely change with one ambiguity such as thin haze or mixed land
  cover.
- `low`: plausible but not sufficient for an evidence claim; retain only as rejected
  or needs-review.

## Imagery screen

Source of truth: Copernicus Sentinel-2 Level-2A surface reflectance. Landsat Collection
2 Level-2 may be used as an independent historical cross-check, not silently mixed
into the detection baseline.

For each AOI, screen the same seasonal window in two years before choosing dates:

- target months: June–September;
- target separation: 12–24 months;
- AOI cloud/shadow: preferably below 5%, reject above 10%;
- no material clipping or missing bands;
- use product identifiers and acquisition timestamps, not screenshot dates;
- retain B02, B03, B04, B08, B11, scene classification, and metadata;
- resample only onto a declared reference grid; never imply that 20 m B11 is observed
  at 10 m detail.

The winning AOI is the one with the best valid date pair and at least three clear rubric
events. If neither qualifies, stop or obtain licensed higher-resolution imagery; do
not compensate with generative reconstruction.

### Provisional date pair

The public Earth Search STAC catalogue returned both AOIs in tile `36RUU`. The initial
season-matched pair is:

- before: `S2A_36RUU_20240815_0_L2A`, 2024-08-15 08:41:53 UTC;
- after: `S2B_36RUU_20250815_0_L2A`, 2025-08-15 08:41:54 UTC.

The acquisition dates and local times are almost identical. Catalogue scene-cloud
values are 0.000259% and 0.004117%, with 0% scene shadow reported for both. These are
tile-wide metadata values, not yet AOI measurements. The pair remains provisional
until the scene-classification masks and pixels are inspected over each AOI. Full
identifiers and source links are recorded in [`source-manifest.csv`](source-manifest.csv).

## Required outputs

```text
source-manifest.csv        product IDs, timestamps, terms URL, CRS, hashes
observed/                  untouched source assets or download references
derived/                   registered composites, indices, and change raster
review/change.geojson      proposed geometry and review fields
report/                    English/Arabic evidence brief
metrics.json               usefulness, false positives, omissions, time
```

Every reviewed feature must contain `aoi_id`, `before_id`, `after_id`, `change_type`,
`method`, `confidence`, `review_state`, `review_note`, and `geometry`. Derived files
must never overwrite source assets.

## Next action

Pull and crop the six tracked observed bands with:

```bash
python -m pip install -e '.[geo]'
python scripts/commercial/pull_crop_sentinel.py \
  --manifest docs/commercial/a2/egypt-monitoring/source-manifest.csv \
  --aoi docs/commercial/a2/egypt-monitoring/aoi.geojson \
  --output-dir data/commercial/egypt-monitoring/derived
```

Use `--dry-run` to resolve and inspect source URLs without reading raster pixels. The
script window-reads public Cloud Optimized GeoTIFFs, writes AOI crops, embeds the source
URL, and creates `crop-manifest.csv` with SHA-256 hashes. Next, calculate AOI
cloud/shadow coverage and produce true-color quicklooks. No plugin code starts here.

Calculate SCL quality metrics and fixed-scale RGB comparisons with:

```bash
python scripts/commercial/analyze_sentinel_crops.py \
  --crop-manifest data/commercial/egypt-monitoring/derived/crop-manifest.csv \
  --source-manifest docs/commercial/a2/egypt-monitoring/source-manifest.csv \
  --output-dir data/commercial/egypt-monitoring/analysis
```

RGB displays use the same physical reflectance range (0.0–0.4) and gamma (2.2) for
both dates. They are labeled visualization products; the single-band GeoTIFF crops
remain the evidence source.

Generate the frozen deterministic change baseline with:

```bash
python scripts/commercial/detect_sentinel_change.py \
  --crop-manifest data/commercial/egypt-monitoring/derived/crop-manifest.csv \
  --aoi-id new_capital_core \
  --after-quicklook data/commercial/egypt-monitoring/analysis/new_capital_core_after_rgb.png \
  --output-dir data/commercial/egypt-monitoring/change
```

Parameters and limitations are fixed in [`change-method.md`](change-method.md).
Candidate polygons are algorithmic proposals and always begin as `needs_review`.

Version 2 is fixed in [`change-method-v2.md`](change-method-v2.md) and evaluated on
held-out Mostakbal. Its tracked decisions are in
[`review-decisions-v2-mostakbal.csv`](review-decisions-v2-mostakbal.csv).

Version 3 tests three-date persistence on a new eastern New Capital holdout. See
[`change-method-v3.md`](change-method-v3.md),
[`change-result-v3.md`](change-result-v3.md), and the tracked
[`review-decisions-v3-holdout.csv`](review-decisions-v3-holdout.csv).

Prepare numbered review cards and the controlled review CSV with:

```bash
python scripts/commercial/review_change_candidates.py \
  --candidates data/commercial/egypt-monitoring/change/change-candidates.geojson \
  --reference-raster data/commercial/egypt-monitoring/change/change-mask.tif \
  --before-quicklook data/commercial/egypt-monitoring/analysis/new_capital_core_before_rgb.png \
  --after-quicklook data/commercial/egypt-monitoring/analysis/new_capital_core_after_rgb.png \
  --review-csv docs/commercial/a2/egypt-monitoring/review-decisions.csv \
  --output-dir data/commercial/egypt-monitoring/review
```

Edit only the tracked `review-decisions.csv`, using the frozen change types or
`rejected` with a reason. Re-run the command with `--finalize` after all rows are
decided; it refuses an incomplete or inconsistent review and writes a separate
reviewed GeoJSON and metrics.
