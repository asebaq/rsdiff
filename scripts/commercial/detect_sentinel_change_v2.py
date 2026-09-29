#!/usr/bin/env python3
"""Construction-aware deterministic Sentinel-2 change baseline v2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.warp import reproject, transform_geom
from scipy import ndimage

BANDS = ("B02", "B03", "B04", "B08")
INVALID_SCL = (0, 1, 3, 8, 9, 10, 11)
METHOD = "construction_aware_v2"


def robust_score(before: np.ndarray, after: np.ndarray, valid: np.ndarray) -> np.ndarray:
    normalized = []
    for index in range(before.shape[0]):
        delta = after[index] - before[index]
        median = np.median(delta[valid])
        mad = np.median(np.abs(delta[valid] - median))
        if mad <= 0:
            raise ValueError(f"band {index} has zero MAD")
        normalized.append((delta - median) / (1.4826 * mad))
    score = np.sqrt(np.mean(np.square(np.stack(normalized)), axis=0)).astype(np.float32)
    score[~valid] = np.nan
    return score


def estimate_shift(before_red: np.ndarray, after_red: np.ndarray, valid: np.ndarray, limit: float = 1.5, step: float = 0.25) -> tuple[float, float, float]:
    before_gradient = np.hypot(ndimage.sobel(before_red, 0), ndimage.sobel(before_red, 1))
    stable = valid & (before_gradient > np.percentile(before_gradient[valid], 80))
    if stable.sum() < 32:
        raise ValueError("registration has too few high-gradient samples")
    sample = stable
    candidates = np.arange(-limit, limit + step / 2, step)
    best: tuple[float, float, float] | None = None
    for dy in candidates:
        for dx in candidates:
            shifted = ndimage.shift(after_red, (dy, dx), order=1, mode="constant", cval=np.nan, prefilter=False)
            residual = shifted[sample] - before_red[sample]
            residual = residual[np.isfinite(residual)]
            offset = np.median(residual)
            objective = float(np.mean(np.abs(residual - offset)))
            candidate = (objective, float(dy), float(dx))
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("registration has no valid samples")
    objective, dy, dx = best
    if abs(dy) == limit or abs(dx) == limit:
        raise ValueError(f"registration optimum lies on search boundary: dy={dy}, dx={dx}")
    return dy, dx, objective


def polygon_area(geometry: dict[str, Any]) -> float:
    def ring(r: list[list[float]]) -> float:
        return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(r, r[1:], strict=False)) / 2)

    rings = geometry["coordinates"]
    return ring(rings[0]) - sum(ring(item) for item in rings[1:])


def classify(area: float, width: float, height: float, before_brightness: float, brightness_delta: float) -> str | None:
    elongation = max(width, height) / max(min(width, height), 1)
    fill = area / max(width * height, 1)
    if area >= 6000 and elongation >= 3 and before_brightness >= 0.22 and brightness_delta <= -0.02:
        return "road_network_formation"
    if area >= 10000 and fill >= 0.15:
        return "large_structural_footprint"
    if area >= 50000:
        return "major_clearing_or_grading"
    if area >= 20000:
        return "large_laydown_or_stockpile_change"
    return None


def load_inputs(manifest: Path, aoi_id: str) -> tuple[dict[tuple[str, str], Path], list[dict[str, str]]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["aoi_id"] == aoi_id]
    indexed = {(row["role"], row["band"]): Path(row["output_path"]) for row in rows}
    required = {(role, band) for role in ("before", "after") for band in (*BANDS, "SCL")}
    if not required.issubset(indexed):
        raise ValueError(f"missing inputs: {sorted(required - set(indexed))}")
    return indexed, rows


def aligned_scl(path: Path, reference: rasterio.DatasetReader) -> np.ndarray:
    result = np.zeros(reference.shape, dtype=np.uint8)
    with rasterio.open(path) as source:
        reproject(source.read(1), result, src_transform=source.transform, src_crs=source.crs, dst_transform=reference.transform, dst_crs=reference.crs, resampling=Resampling.nearest)
    return result


def write_raster(path: Path, values: np.ndarray, reference: rasterio.DatasetReader, dtype: str, nodata: float | int) -> None:
    profile = reference.profile.copy()
    profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate", predictor=2, tiled=True)
    with rasterio.open(path, "w", **profile) as target:
        target.write(values.astype(dtype), 1)
        target.update_tags(METHOD=METHOD, PROCESSING="algorithmic proposal; not observed geography")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--aoi-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / name for name in ("change-score.tif", "change-mask.tif", "change-candidates.geojson", "change-metrics.json")]
    if not args.force and any(path.exists() for path in outputs):
        raise FileExistsError("output exists; pass --force")
    indexed, rows = load_inputs(args.crop_manifest, args.aoi_id)
    with rasterio.open(indexed[("before", "B04")]) as reference:
        before, after = [], []
        signature = (reference.shape, reference.transform, reference.crs)
        for band in BANDS:
            with rasterio.open(indexed[("before", band)]) as b, rasterio.open(indexed[("after", band)]) as a:
                if (b.shape, b.transform, b.crs) != signature or (a.shape, a.transform, a.crs) != signature:
                    raise ValueError(f"unaligned input: {band}")
                before.append(b.read(1).astype(np.float32) * 0.0001 - 0.1)
                after.append(a.read(1).astype(np.float32) * 0.0001 - 0.1)
        before_stack, after_stack = np.stack(before), np.stack(after)
        before_scl, after_scl = aligned_scl(indexed[("before", "SCL")], reference), aligned_scl(indexed[("after", "SCL")], reference)
        valid = np.all(before_stack > -0.1, axis=0) & np.all(after_stack > -0.1, axis=0)
        valid &= ~np.isin(before_scl, INVALID_SCL) & ~np.isin(after_scl, INVALID_SCL)
        dy, dx, registration_mad = estimate_shift(before_stack[2], after_stack[2], valid)
        shifted_after = np.stack([ndimage.shift(band, (dy, dx), order=1, mode="constant", cval=np.nan, prefilter=False) for band in after_stack])
        shifted_scl = ndimage.shift(after_scl, (dy, dx), order=0, mode="constant", cval=0, prefilter=False)
        valid &= np.all(np.isfinite(shifted_after), axis=0) & ~np.isin(shifted_scl, INVALID_SCL)
        score = robust_score(before_stack, shifted_after, valid)
        before_ndvi = (before_stack[3] - before_stack[2]) / np.maximum(before_stack[3] + before_stack[2], 0.01)
        after_ndvi = (shifted_after[3] - shifted_after[2]) / np.maximum(shifted_after[3] + shifted_after[2], 0.01)
        before_brightness = before_stack[:3].mean(axis=0)
        after_brightness = shifted_after[:3].mean(axis=0)
        brightness_delta = after_brightness - before_brightness
        raw = valid & (score >= 4.5) & (np.abs(brightness_delta) >= 0.03) & (before_ndvi <= 0.25) & (after_ndvi <= 0.25)
        grouped = ndimage.binary_closing(raw, structure=np.ones((7, 7), dtype=bool))
        labels, count = ndimage.label(grouped, structure=np.ones((3, 3), dtype=np.uint8))
        retained = np.zeros(reference.shape, dtype=np.uint8)
        proposals = []
        pixel_area = abs(reference.transform.a * reference.transform.e)
        for label_id in range(1, count + 1):
            group = labels == label_id
            evidence = group & raw
            if not evidence.any():
                continue
            rows_px, cols_px = np.nonzero(group)
            width = (cols_px.max() - cols_px.min() + 1) * abs(reference.transform.a)
            height = (rows_px.max() - rows_px.min() + 1) * abs(reference.transform.e)
            area = float(evidence.sum() * pixel_area)
            change_type = classify(area, width, height, float(np.median(before_brightness[evidence])), float(np.median(brightness_delta[evidence])))
            if change_type is None:
                continue
            retained[group] = 1
            component_geometries = [geometry for geometry, value in shapes(group.astype(np.uint8), mask=group, connectivity=8, transform=reference.transform) if value == 1]
            if len(component_geometries) != 1:
                raise ValueError(f"aggregate {label_id} produced multiple geometries")
            proposals.append((component_geometries[0], evidence, area, change_type))
        before_id = next(row["item_id"] for row in rows if row["role"] == "before")
        after_id = next(row["item_id"] for row in rows if row["role"] == "after")
        features = []
        for index, (geometry, evidence, area, change_type) in enumerate(proposals, 1):
            p90 = float(np.nanpercentile(score[evidence], 90))
            features.append({"type": "Feature", "id": f"{args.aoi_id}-v2-{index:04d}", "properties": {
                "aoi_id": args.aoi_id, "before_id": before_id, "after_id": after_id,
                "change_type": change_type, "method": METHOD, "confidence": "high" if p90 >= 8 else "medium",
                "review_state": "needs_review", "review_note": "", "evidence_area_m2": area,
                "aggregate_area_m2": polygon_area(geometry), "score_p90": round(p90, 3),
            }, "geometry": transform_geom(reference.crs, "EPSG:4326", geometry, precision=7)})
        write_raster(outputs[0], np.nan_to_num(score, nan=-9999), reference, "float32", -9999)
        write_raster(outputs[1], retained, reference, "uint8", 0)
    collection = {"type": "FeatureCollection", "name": f"{args.aoi_id}_v2_candidates", "features": features}
    metrics = {"aoi_id": args.aoi_id, "method": METHOD, "registration_shift_pixels": {"dy": dy, "dx": dx}, "registration_objective_mae": registration_mad, "valid_pixels": int(valid.sum()), "raw_pixels": int(raw.sum()), "retained_components": len(features), "retained_pixels": int(retained.sum())}
    outputs[2].write_text(json.dumps(collection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs[3].write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(features)} v2 candidates for {args.aoi_id}; shift dy={dy}, dx={dx}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
