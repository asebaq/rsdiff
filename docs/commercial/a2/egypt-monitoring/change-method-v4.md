# Sentinel-1 backscatter corroboration v4

Frozen 2026-09-29, before any Sentinel-2 or Sentinel-1 pixels of the evaluation AOI are
inspected. This file must not be changed after the blind review below is recorded.

## Motivation

The v2 and v3 false positives were broad disturbed sand, ambiguous bare surfaces,
existing linear surfaces, and AOI-edge truncation. All of them are
reflectance-only evidence. Sentinel-1 C-band backscatter responds to surface roughness
and geometry instead of colour, so it is an independent sensor view in the sense of the
frozen rubric. Construction adds new dihedral and roughness structure; sand that was
disturbed before the baseline date and was not changed afterwards should keep its
backscatter.

Interferometric coherence would separate active work from static ground more directly,
but it needs SLC processing and Copernicus/Earthdata accounts that this experiment does
not have. v4 uses terrain-corrected backscatter only. Coherence remains a separate
future method.

## Evaluation AOI

`new_capital_south_holdout`, 31.72–31.79° E × 29.91–29.97° N (about 47 km²). It was
selected by a fixed rule, not by looking at its imagery: it is the same size as the
earlier AOIs, directly south of `new_capital_core`, and does not overlap any AOI used
before. Before this method was frozen, the only quantity read over the AOI was one
connectivity-test scalar: mean linear VV for a single 2024-08-17 scene, 0.110. No
image, histogram, or change product was viewed.

Sentinel-2 dates are the unchanged v2 pair: `S2A_36RUU_20240815_0_L2A` and
`S2B_36RUU_20250815_0_L2A`.

## Fixed method

1. Generate proposals with the unchanged `construction_aware_v2` detector.
2. Search the Microsoft Planetary Computer `sentinel-1-rtc` collection (Catalyst RTC
   gamma-nought, 10 m, CC BY 4.0) for IW scenes from relative orbit 58, ascending,
   acquired within ±30 days of each Sentinel-2 timestamp. Abort if either window has
   fewer than 3 scenes. Record every item ID and each crop's SHA-256.
3. Reproject VV and VH onto the Sentinel-2 before-date B04 grid with bilinear
   resampling. A value is valid when it is finite and positive. The per-pixel window
   composite is the median of the valid linear values; it requires at least 3 valid
   scenes. Convert the composite to dB.
4. `Δ = after_dB − before_dB` for each polarization. The background is every pixel that
   is valid in both Sentinel-2 dates and both SAR composites and lies more than 50 m
   from every v2 aggregate. Subtract the background median of `Δ` to remove
   sensor/calibration offsets between windows (S1A only in 2024; S1A+S1C in 2025).
   Report the background robust sigma, `1.4826 × MAD`.
5. For each v2 proposal, reconstruct its original v2 evidence pixels as v3 does. Use the
   median calibrated `Δ` over evidence pixels that are SAR-valid.
6. Apply these rules in order:
   - drop `aoi_edge` if any aggregate pixel is within 100 m (10 pixels) of the crop edge;
   - drop `sar_insufficient` if fewer than 30 evidence pixels are SAR-valid;
   - `large_structural_footprint`: retain only if median `ΔVV ≥ +1.0 dB`;
   - every other v2 class: retain only if `max(|median ΔVV|, |median ΔVH|) ≥ 1.0 dB`;
   - otherwise drop `sar_unchanged`.
7. Retained proposals keep their v2 class and geometry. They are exported as
   `needs_review` with SAR fields, scene counts, and item IDs attached. SAR values
   corroborate observed optical change; they are never used to draw new geometry.

## Evaluation

1. Review every v2 proposal on the evaluation AOI with the existing Sentinel-2
   before/after review cards, using the frozen rubric. The reviewer must not see SAR
   outputs. Record the decisions before the v4 filter runs.
2. Run v4 once and join its retain/drop status to those decisions.
3. Metrics:
   - v4 usefulness: accepted ÷ retained;
   - v4 recall: accepted and retained ÷ all accepted v2 proposals;
   - v2 usefulness on the same AOI, as the baseline.
4. **Pass** requires v4 usefulness ≥ 80%, recall ≥ 60%, and at least 3 accepted retained
   proposals. If fewer than 5 proposals are retained, the result is **inconclusive**,
   not a pass. The recall gate prevents a filter that drops almost everything from
   passing on precision alone.
5. Secondary, not gated: apply the same frozen v4 to the existing labelled proposals —
   Mostakbal v2 (17 decisions) and the east-holdout proposals that have v3 decisions.
   Those labels were created before v4 existed, but the designer has already seen those
   AOIs, so they cannot replace the fresh holdout.

No threshold, window, orbit, or rule may change after the blind review or after v4
outputs are seen. A failure is reported as a failure.
