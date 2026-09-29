#!/usr/bin/env python3
"""Generate deterministic spectral-change candidates from aligned Sentinel-2 crops."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.warp import reproject, transform_geom

BANDS = ("B02", "B03", "B04", "B08")
INVALID_SCL = (0, 1, 3, 8, 9, 10, 11)
METHOD = "robust_multiband_rms_v1"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def robust_score(before: np.ndarray, after: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("before and after arrays must be matching band stacks")
    if valid.shape != before.shape[1:] or not valid.any():
        raise ValueError("valid mask must match the raster and contain valid pixels")
    normalized = []
    stats = []
    for index in range(before.shape[0]):
        delta = after[index].astype(np.float32) - before[index].astype(np.float32)
        median = float(np.median(delta[valid]))
        mad = float(np.median(np.abs(delta[valid] - median)))
        scale = 1.4826 * mad
        if scale <= 0:
            raise ValueError(f"band {index} has zero robust difference scale")
        normalized.append((delta - median) / scale)
        stats.append({"median_delta_dn": median, "mad_delta_dn": mad, "robust_scale_dn": scale})
    score = np.sqrt(np.mean(np.square(np.stack(normalized)), axis=0)).astype(np.float32)
    score[~valid] = np.nan
    return score, stats


def polygon_area(geometry: dict[str, Any]) -> float:
    def ring_area(ring: list[list[float]]) -> float:
        return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(ring, ring[1:], strict=False)) / 2)

    if geometry["type"] == "Polygon":
        rings = geometry["coordinates"]
        return ring_area(rings[0]) - sum(ring_area(ring) for ring in rings[1:])
    return sum(polygon_area({"type": "Polygon", "coordinates": polygon}) for polygon in geometry["coordinates"])


def align_scl(path: Path, reference: rasterio.DatasetReader) -> np.ndarray:
    result = np.zeros(reference.shape, dtype=np.uint8)
    with rasterio.open(path) as source:
        reproject(
            source=source.read(1), destination=result,
            src_transform=source.transform, src_crs=source.crs,
            dst_transform=reference.transform, dst_crs=reference.crs,
            resampling=Resampling.nearest,
        )
    return result


def write_raster(path: Path, values: np.ndarray, reference: rasterio.DatasetReader, dtype: str, nodata: float | int, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    profile = reference.profile.copy()
    profile.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata, compress="deflate", predictor=2, tiled=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tif", delete=False) as tmp:
        temp_name = tmp.name
    try:
        with rasterio.open(temp_name, "w", **profile) as target:
            target.write(values.astype(dtype), 1)
            target.update_tags(METHOD=METHOD, PROCESSING="deterministic change candidate; not observed geography")
        os.replace(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)


def atomic_json(path: Path, value: object, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as tmp:
        json.dump(value, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        temp_name = tmp.name
    os.replace(temp_name, path)


def write_preview(path: Path, quicklook: Path, mask: np.ndarray, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    base = Image.open(quicklook).convert("RGBA")
    if base.size != (mask.shape[1], mask.shape[0]):
        raise ValueError("after quicklook and change mask dimensions do not match")
    red = Image.new("RGBA", base.size, (255, 0, 0, 125))
    base.alpha_composite(Image.composite(red, Image.new("RGBA", base.size), Image.fromarray(mask * 255)))
    header = 42
    preview = Image.new("RGB", (base.width, base.height + header), "white")
    preview.paste(base.convert("RGB"), (0, header))
    ImageDraw.Draw(preview).text((12, 13), "ALGORITHM CANDIDATES - NEEDS REVIEW", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(path, optimize=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--aoi-id", default="new_capital_core")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--minimum-area-m2", type=float, default=6000.0)
    parser.add_argument("--high-confidence-score", type=float, default=8.0)
    parser.add_argument("--after-quicklook", type=Path, help="Optional aligned RGB PNG for a red candidate overlay")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threshold <= 0 or args.minimum_area_m2 <= 0 or args.high_confidence_score < args.threshold:
        raise ValueError("invalid threshold, area, or confidence parameters")
    rows = [row for row in read_manifest(args.crop_manifest) if row["aoi_id"] == args.aoi_id]
    indexed = {(row["role"], row["band"]): Path(row["output_path"]) for row in rows}
    required = {(role, band) for role in ("before", "after") for band in (*BANDS, "SCL")}
    if not required.issubset(indexed):
        raise ValueError(f"crop manifest lacks inputs: {sorted(required - set(indexed))}")

    with rasterio.open(indexed[("before", BANDS[0])]) as reference:
        before, after = [], []
        for band in BANDS:
            with rasterio.open(indexed[("before", band)]) as source_before, rasterio.open(indexed[("after", band)]) as source_after:
                signature = (reference.shape, reference.transform, reference.crs)
                if (source_before.shape, source_before.transform, source_before.crs) != signature or (source_after.shape, source_after.transform, source_after.crs) != signature:
                    raise ValueError(f"unaligned grid for {band}")
                before.append(source_before.read(1))
                after.append(source_after.read(1))
        before_stack, after_stack = np.stack(before), np.stack(after)
        before_scl = align_scl(indexed[("before", "SCL")], reference)
        after_scl = align_scl(indexed[("after", "SCL")], reference)
        valid = np.all(before_stack > 0, axis=0) & np.all(after_stack > 0, axis=0)
        valid &= ~np.isin(before_scl, INVALID_SCL) & ~np.isin(after_scl, INVALID_SCL)
        score, band_stats = robust_score(before_stack, after_stack, valid)
        raw_mask = valid & (score >= args.threshold)

        retained_projected = []
        for geometry, value in shapes(raw_mask.astype(np.uint8), mask=raw_mask, connectivity=8, transform=reference.transform):
            if value == 1 and polygon_area(geometry) >= args.minimum_area_m2:
                retained_projected.append(geometry)
        filtered = rasterize(((geometry, 1) for geometry in retained_projected), out_shape=reference.shape, transform=reference.transform, fill=0, dtype=np.uint8)

        features = []
        before_id = next(row["item_id"] for row in rows if row["role"] == "before")
        after_id = next(row["item_id"] for row in rows if row["role"] == "after")
        for index, geometry in enumerate(retained_projected, 1):
            component = rasterize([(geometry, 1)], out_shape=reference.shape, transform=reference.transform, fill=0, dtype=np.uint8).astype(bool)
            component_scores = score[component]
            p90 = float(np.nanpercentile(component_scores, 90))
            features.append({
                "type": "Feature",
                "id": f"{args.aoi_id}-{index:04d}",
                "properties": {
                    "aoi_id": args.aoi_id, "before_id": before_id, "after_id": after_id,
                    "change_type": "unclassified_spectral_change", "method": METHOD,
                    "confidence": "high" if p90 >= args.high_confidence_score else "medium",
                    "review_state": "needs_review", "review_note": "",
                    "area_m2": round(polygon_area(geometry), 1),
                    "score_p50": round(float(np.nanmedian(component_scores)), 3),
                    "score_p90": round(p90, 3),
                    "score_max": round(float(np.nanmax(component_scores)), 3),
                },
                "geometry": transform_geom(reference.crs, "EPSG:4326", geometry, precision=7),
            })

        score_output = np.nan_to_num(score, nan=-9999.0)
        write_raster(args.output_dir / "change-score.tif", score_output, reference, "float32", -9999.0, args.force)
        write_raster(args.output_dir / "change-mask.tif", filtered, reference, "uint8", 0, args.force)

    collection = {"type": "FeatureCollection", "name": f"{args.aoi_id}_change_candidates", "features": features}
    atomic_json(args.output_dir / "change-candidates.geojson", collection, args.force)
    if args.after_quicklook:
        write_preview(args.output_dir / "change-preview.png", args.after_quicklook, filtered, args.force)
    metrics = {
        "aoi_id": args.aoi_id, "method": METHOD, "threshold": args.threshold,
        "minimum_area_m2": args.minimum_area_m2, "high_confidence_score": args.high_confidence_score,
        "valid_pixels": int(valid.sum()), "raw_candidate_pixels": int(raw_mask.sum()),
        "retained_candidate_pixels": int(filtered.sum()), "retained_components": len(features),
        "retained_area_m2": float(filtered.sum() * abs(reference.transform.a * reference.transform.e)),
        "bands": dict(zip(BANDS, band_stats, strict=True)),
    }
    atomic_json(args.output_dir / "change-metrics.json", metrics, args.force)
    print(f"wrote {len(features)} review candidates to {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
