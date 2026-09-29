#!/usr/bin/env python3
"""Filter v2 Sentinel candidates using an independent third-date persistence check."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from detect_sentinel_change_v2 import BANDS, INVALID_SCL, estimate_shift, robust_score
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject, transform_geom
from scipy import ndimage

METHOD = "three_date_persistence_v3"


def load_manifest(path: Path, aoi_id: str) -> tuple[dict[tuple[str, str], Path], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["aoi_id"] == aoi_id]
    indexed = {(row["role"], row["band"]): Path(row["output_path"]) for row in rows}
    required = {(role, band) for role in ("before", "after", "followup") for band in (*BANDS, "SCL")}
    if not required.issubset(indexed):
        raise ValueError(f"missing three-date inputs: {sorted(required - set(indexed))}")
    return indexed, rows


def align_scl(path: Path, reference: rasterio.DatasetReader) -> np.ndarray:
    output = np.zeros(reference.shape, dtype=np.uint8)
    with rasterio.open(path) as source:
        reproject(source.read(1), output, src_transform=source.transform, src_crs=source.crs, dst_transform=reference.transform, dst_crs=reference.crs, resampling=Resampling.nearest)
    return output


def persistence_mask(before: np.ndarray, after: np.ndarray, followup: np.ndarray, evidence: np.ndarray) -> tuple[np.ndarray, float]:
    distance_12 = np.linalg.norm(after - before, axis=0)
    distance_13 = np.linalg.norm(followup - before, axis=0)
    distance_23 = np.linalg.norm(followup - after, axis=0)
    usable = evidence & np.isfinite(distance_13) & (distance_12 > 0)
    persistent = usable & (distance_13 >= 0.7 * distance_12) & (distance_23 <= distance_13)
    fraction = float(persistent.sum() / max(usable.sum(), 1))
    return persistent, fraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--v2-candidates", type=Path, required=True)
    parser.add_argument("--aoi-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = json.loads(args.v2_candidates.read_text(encoding="utf-8"))
    indexed, manifest_rows = load_manifest(args.crop_manifest, args.aoi_id)
    with rasterio.open(indexed[("before", "B04")]) as reference:
        stacks = {}
        signature = (reference.shape, reference.transform, reference.crs)
        for role in ("before", "after", "followup"):
            arrays = []
            for band in BANDS:
                with rasterio.open(indexed[(role, band)]) as source:
                    if (source.shape, source.transform, source.crs) != signature:
                        raise ValueError(f"unaligned grid: {role}/{band}")
                    arrays.append(source.read(1).astype(np.float32) * 0.0001 - 0.1)
            stacks[role] = np.stack(arrays)
        scls = {role: align_scl(indexed[(role, "SCL")], reference) for role in ("before", "after", "followup")}
        valid_12 = np.all(stacks["before"] > -0.1, axis=0) & np.all(stacks["after"] > -0.1, axis=0)
        valid_12 &= ~np.isin(scls["before"], INVALID_SCL) & ~np.isin(scls["after"], INVALID_SCL)
        dy12, dx12, _ = estimate_shift(stacks["before"][2], stacks["after"][2], valid_12)
        after = np.stack([ndimage.shift(band, (dy12, dx12), order=1, mode="constant", cval=np.nan, prefilter=False) for band in stacks["after"]])
        valid_23 = np.all(stacks["after"] > -0.1, axis=0) & np.all(stacks["followup"] > -0.1, axis=0)
        valid_23 &= ~np.isin(scls["after"], INVALID_SCL) & ~np.isin(scls["followup"], INVALID_SCL)
        dy23, dx23, _ = estimate_shift(stacks["after"][2], stacks["followup"][2], valid_23)
        followup = np.stack([ndimage.shift(band, (dy12 + dy23, dx12 + dx23), order=1, mode="constant", cval=np.nan, prefilter=False) for band in stacks["followup"]])
        valid = valid_12 & np.all(np.isfinite(after), axis=0) & np.all(np.isfinite(followup), axis=0)
        score = robust_score(stacks["before"], after, valid)
        before_ndvi = (stacks["before"][3] - stacks["before"][2]) / np.maximum(stacks["before"][3] + stacks["before"][2], 0.01)
        after_ndvi = (after[3] - after[2]) / np.maximum(after[3] + after[2], 0.01)
        brightness_delta = after[:3].mean(axis=0) - stacks["before"][:3].mean(axis=0)
        evidence_all = valid & (score >= 4.5) & (np.abs(brightness_delta) >= 0.03) & (before_ndvi <= 0.25) & (after_ndvi <= 0.25)
        retained_features = []
        retained_mask = np.zeros(reference.shape, dtype=np.uint8)
        for feature in candidates.get("features", []):
            projected = transform_geom("EPSG:4326", reference.crs, feature["geometry"], precision=3)
            aggregate = rasterize([(projected, 1)], out_shape=reference.shape, transform=reference.transform, fill=0, dtype=np.uint8).astype(bool)
            evidence = aggregate & evidence_all
            persistent, fraction = persistence_mask(stacks["before"], after, followup, evidence)
            if fraction < 0.65:
                continue
            kept = json.loads(json.dumps(feature))
            kept["id"] = feature["id"].replace("-v2-", "-v3-")
            kept["properties"]["method"] = METHOD
            kept["properties"]["review_state"] = "needs_review"
            kept["properties"]["review_note"] = ""
            kept["properties"]["followup_id"] = next(row["item_id"] for row in manifest_rows if row["role"] == "followup")
            kept["properties"]["persistence_fraction"] = round(fraction, 4)
            kept["properties"]["persistent_pixels"] = int(persistent.sum())
            kept["properties"]["persistence_valid_pixels"] = int(evidence.sum())
            retained_features.append(kept)
            retained_mask[aggregate] = 1
        profile = reference.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0, compress="deflate", tiled=True)
        mask_path = args.output_dir / "persistent-mask.tif"
        if mask_path.exists() and not args.force:
            raise FileExistsError("output exists; pass --force")
        with rasterio.open(mask_path, "w", **profile) as target:
            target.write(retained_mask, 1)
            target.update_tags(METHOD=METHOD, PROCESSING="algorithmic proposal; not observed geography")
    output = {"type": "FeatureCollection", "name": f"{args.aoi_id}_v3_candidates", "features": retained_features}
    metrics = {"aoi_id": args.aoi_id, "method": METHOD, "input_v2_candidates": len(candidates.get("features", [])), "retained_candidates": len(retained_features), "minimum_persistence_fraction": 0.65, "registration_shifts_pixels": {"after_to_before": {"dy": dy12, "dx": dx12}, "followup_to_after": {"dy": dy23, "dx": dx23}}}
    (args.output_dir / "change-candidates.geojson").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "persistence-metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"retained {len(retained_features)} of {len(candidates.get('features', []))} v2 candidates")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
