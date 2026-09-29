# Sentinel-1 backscatter corroboration v4 result

Run 2026-09-29 with the method frozen in [`change-method-v4.md`](change-method-v4.md).
The 23 v2 proposals on `new_capital_south_holdout` were reviewed on Sentinel-2
before/after imagery alone. Those decisions were committed in
[`review-decisions-v4-south.csv`](review-decisions-v4-south.csv) before the SAR filter
ran for the first time. No parameter was changed afterwards.

## Primary result — `new_capital_south_holdout`

| Metric | Value |
|---|---:|
| v2 proposals | 23 |
| v2 accepted by blind review | 10 (**43.48%** usefulness) |
| Sentinel-1 scenes, 2024 / 2025 window | 5 (S1A) / 10 (S1A + S1C) |
| Background calibration offset, ΔVV / ΔVH | −0.16 / −0.65 dB |
| Per-pixel background robust sigma, ΔVV / ΔVH | 1.36 / 1.34 dB |
| Dropped `aoi_edge` / `sar_unchanged` / `sar_insufficient` | 8 / 9 / 0 |
| v4 retained | 6 |
| v4 accepted | 1 |
| **v4 usefulness** | **16.67%** (gate 80%) |
| **v4 recall** | **10.0%** (gate 60%) |
| **Verdict** | **fail** |

## Secondary checks (not gated)

| AOI / labels | v2 usefulness | v4 retained | v4 usefulness | v4 recall | Frozen verdict |
|---|---:|---:|---:|---:|---|
| Mostakbal, 17 v2 decisions | 76.47% | 8 | 87.50% | 53.85% | fail (recall) |
| East holdout, 8 of 11 v2 proposals with v3 decisions | 62.50% | 2 | 100.00% | 40.00% | inconclusive |

On the already-seen AOIs, backscatter raised precision but dropped roughly half of the
useful proposals. On the fresh AOI it did neither: the accepted and rejected proposals
have overlapping median SAR changes.

## Findings

1. **Backscatter did not separate useful from non-useful proposals on the fresh AOI.**
   - Among non-edge proposals, accepted ones had `max(|ΔVV|, |ΔVH|)` from 0.58 to
     3.19 dB.
   - Rejected ones had 0.16 to 2.51 dB.
   - Tonal changes along road verges (0009, 0014) and faint desert patches (0016, 0019)
     crossed 1 dB.
   - Visibly new roof rows (0007, 0010) and a works-yard pad (0005) did not.
2. **The class-specific rule inherits v2's class errors.** v2 labelled 15 of 23
   proposals `large_structural_footprint`, but the reviewer accepted only 4 as
   structural. The new graded road 0020 darkened (ΔVV −1.80 dB), which is physically
   consistent with a smoother surface. It was dropped because v2 had called it
   structural and the structural rule requires a VV increase.
3. **The edge rule was the largest single loss.** It dropped 8 of 23 proposals,
   including accepted 0002, 0003, 0008, and 0021. v2 aggregates are sprawling, so
   "any pixel within 100 m of the crop edge" removes large valid areas together with
   true truncation artefacts.
4. **The no-data edge was not covered by the edge rule.** The frozen edge rule measures
   distance from the crop edge, not the Sentinel-2 tile's no-data boundary. The three
   no-data-edge artefacts (0013, 0022, 0023) were dropped as `aoi_edge` only because
   they also reached the crop edge. A no-data boundary inside the crop would not be
   caught.
5. **The v2 detector did not generalize.** Its usefulness fell from 76.47% on
   Mostakbal to 62.5% on the east holdout and 43.48% here. Established neighbourhoods
   produce sprawling tonal aggregates (0001, 0006) that no per-proposal
   corroboration rule can rescue.

## Decision

v4 fails. Do not tune the edge distance, dB threshold, or class rules on this holdout.

The Sentinel-only track has now been evaluated on three fresh AOIs with four frozen
methods, and none reached the 80% gate. The common failure is proposal geometry and
class, not a missing extra signal. Following [`../../../PLAN.md`](../../../PLAN.md)
step 7, the next choice is not v5 thresholding. It is one of:

- **change the target resolution:** use licensed sub-metre archival imagery for one
  site, keeping Sentinel-2 only as a coarse screen;
- **reframe the claim:** offer analyst-triage screening, where recall and review time
  are the gated metrics instead of per-polygon usefulness, and test it with users;
- **stop** the public-data milestone-verification hypothesis.

Interferometric coherence remains untested. It is a different measurement from
backscatter and could still distinguish active works from static ground, but it would
not repair v2's geometry or class errors.

## Limitations

- One reviewer designed the method and reviewed all decisions.
- Review used 10 m true-colour quicklooks, and several decisions were close calls.
- The 2024 window has half as many looks as the 2025 window, so the before composite
  has more speckle.
- All three AOIs sit in one Sentinel-2 tile and one Sentinel-1 relative orbit.
