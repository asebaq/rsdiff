#!/usr/bin/env python3
"""Corroborate v2 Sentinel-2 proposals with Sentinel-1 RTC backscatter (method v4)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from detect_sentinel_change_v2 import BANDS, INVALID_SCL, estimate_shift, robust_score
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds, transform_geom
from scipy import ndimage

METHOD = "sar_corroboration_v4"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
COLLECTION = "sentinel-1-rtc"
RELATIVE_ORBIT = 58
ORBIT_STATE = "ascending"
WINDOW_DAYS = 30
MIN_SCENES = 3
POLARIZATIONS = ("vv", "vh")
EDGE_PIXELS = 10
BACKGROUND_BUFFER_PIXELS = 5
MIN_SAR_PIXELS = 30
THRESHOLD_DB = 1.0


def decide(change_type: str, median_vv: float, median_vh: float, near_edge: bool, sar_pixels: int) -> str:
    """Return ``retained`` or the frozen drop reason for one proposal."""
    if near_edge:
        return "aoi_edge"
    if sar_pixels < MIN_SAR_PIXELS:
        return "sar_insufficient"
    if change_type == "large_structural_footprint":
        return "retained" if median_vv >= THRESHOLD_DB else "sar_unchanged"
    return "retained" if max(abs(median_vv), abs(median_vh)) >= THRESHOLD_DB else "sar_unchanged"


def composite_db(stack: np.ndarray) -> np.ndarray:
    """Median of valid linear backscatter per pixel, in dB; NaN below MIN_SCENES valid looks."""
    valid = np.isfinite(stack) & (stack > 0)
    values = np.where(valid, stack, np.nan)
    count = valid.sum(axis=0)
    median = np.full(stack.shape[1:], np.nan, dtype=np.float32)
    enough = count >= MIN_SCENES
    median[enough] = np.nanmedian(values[:, enough], axis=0)
    return (10 * np.log10(median)).astype(np.float32)


def touches_edge(mask: np.ndarray, pixels: int = EDGE_PIXELS) -> bool:
    rows, cols = np.nonzero(mask)
    height, width = mask.shape
    return bool(rows.min() < pixels or cols.min() < pixels or rows.max() >= height - pixels or cols.max() >= width - pixels)


def evaluate(status: dict[str, str], reviews: dict[str, str]) -> dict[str, Any]:
    """Score v4 retain/drop status against blind review states using the frozen gates."""
    missing = sorted(set(status) - set(reviews))
    if missing:
        raise ValueError(f"review lacks decisions for: {missing}")
    retained = [cid for cid, state in status.items() if state == "retained"]
    accepted = [cid for cid in status if reviews[cid] == "accepted"]
    accepted_retained = [cid for cid in retained if reviews[cid] == "accepted"]
    usefulness = 100 * len(accepted_retained) / len(retained) if retained else 0.0
    recall = 100 * len(accepted_retained) / len(accepted) if accepted else 0.0
    if len(retained) < 5:
        verdict = "inconclusive"
    elif usefulness >= 80 and recall >= 60 and len(accepted_retained) >= 3:
        verdict = "pass"
    else:
        verdict = "fail"
    return {
        "v2_proposals": len(status), "v2_accepted": len(accepted),
        "v2_usefulness_pct": round(100 * len(accepted) / max(len(status), 1), 2),
        "v4_retained": len(retained), "v4_accepted": len(accepted_retained),
        "v4_usefulness_pct": round(usefulness, 2), "v4_recall_pct": round(recall, 2),
        "verdict": verdict,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sas_token() -> str:
    request = urllib.request.Request(TOKEN_URL, headers={"User-Agent": "rsdiff-commercial-validation/1"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.load(response)["token"]


def search_window(client: Client, bbox: tuple[float, float, float, float], center: datetime) -> list[Any]:
    start, end = center - timedelta(days=WINDOW_DAYS), center + timedelta(days=WINDOW_DAYS)
    items = client.search(collections=[COLLECTION], bbox=bbox, datetime=f"{start:%Y-%m-%dT%H:%M:%SZ}/{end:%Y-%m-%dT%H:%M:%SZ}").items()
    selected = [
        item for item in items
        if item.properties.get("sat:relative_orbit") == RELATIVE_ORBIT
        and item.properties.get("sat:orbit_state") == ORBIT_STATE
        and item.properties.get("sar:instrument_mode") == "IW"
        and all(pol in item.assets for pol in POLARIZATIONS)
    ]
    if len(selected) < MIN_SCENES:
        raise ValueError(f"only {len(selected)} Sentinel-1 scenes within ±{WINDOW_DAYS} days of {center.isoformat()}")
    return sorted(selected, key=lambda item: item.datetime)


def read_on_grid(href: str, reference: rasterio.DatasetReader, output: Path, force: bool) -> np.ndarray:
    if output.exists() and not force:
        with rasterio.open(output) as cached:
            return cached.read(1)
    with rasterio.open(href) as source, WarpedVRT(
        source, crs=reference.crs, transform=reference.transform, width=reference.width, height=reference.height,
        resampling=Resampling.bilinear, src_nodata=source.nodata, nodata=np.nan, dtype="float32",
    ) as vrt:
        values = vrt.read(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    profile = reference.profile.copy()
    profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate", predictor=3, tiled=True)
    with rasterio.open(output, "w", **profile) as target:
        target.write(values, 1)
        target.update_tags(PROCESSING="observed Sentinel-1 RTC gamma0, bilinear onto Sentinel-2 grid")
    return values


def load_s2(crop_manifest: Path, source_manifest: Path, aoi_id: str) -> tuple[dict[tuple[str, str], Path], dict[str, dict[str, str]]]:
    with crop_manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["aoi_id"] == aoi_id]
    indexed = {(row["role"], row["band"]): Path(row["output_path"]) for row in rows}
    required = {(role, band) for role in ("before", "after") for band in (*BANDS, "SCL")}
    if not required.issubset(indexed):
        raise ValueError(f"missing inputs: {sorted(required - set(indexed))}")
    with source_manifest.open(newline="", encoding="utf-8") as handle:
        sources = {row["role"]: row for row in csv.DictReader(handle) if row["aoi_id"] == aoi_id and row["role"] in ("before", "after")}
    if set(sources) != {"before", "after"}:
        raise ValueError(f"source manifest lacks before/after rows for {aoi_id}")
    return indexed, sources


def v2_evidence(indexed: dict[tuple[str, str], Path], reference: rasterio.DatasetReader) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct frozen v2 evidence pixels and S2 validity on the before grid."""
    stacks, scl = {}, {}
    for role in ("before", "after"):
        arrays = []
        for band in BANDS:
            with rasterio.open(indexed[(role, band)]) as source:
                if (source.shape, source.transform, source.crs) != (reference.shape, reference.transform, reference.crs):
                    raise ValueError(f"unaligned grid: {role}/{band}")
                arrays.append(source.read(1).astype(np.float32) * 0.0001 - 0.1)
        stacks[role] = np.stack(arrays)
    for role in ("before", "after"):
        scl[role] = np.zeros(reference.shape, dtype=np.uint8)
        with rasterio.open(indexed[(role, "SCL")]) as source:
            reproject(source.read(1), scl[role], src_transform=source.transform, src_crs=source.crs, dst_transform=reference.transform, dst_crs=reference.crs, resampling=Resampling.nearest)
    valid = np.all(stacks["before"] > -0.1, axis=0) & np.all(stacks["after"] > -0.1, axis=0)
    valid &= ~np.isin(scl["before"], INVALID_SCL) & ~np.isin(scl["after"], INVALID_SCL)
    dy, dx, _ = estimate_shift(stacks["before"][2], stacks["after"][2], valid)
    after = np.stack([ndimage.shift(band, (dy, dx), order=1, mode="constant", cval=np.nan, prefilter=False) for band in stacks["after"]])
    shifted_scl = ndimage.shift(scl["after"], (dy, dx), order=0, mode="constant", cval=0, prefilter=False)
    valid &= np.all(np.isfinite(after), axis=0) & ~np.isin(shifted_scl, INVALID_SCL)
    before = stacks["before"]
    score = robust_score(before, after, valid)
    before_ndvi = (before[3] - before[2]) / np.maximum(before[3] + before[2], 0.01)
    after_ndvi = (after[3] - after[2]) / np.maximum(after[3] + after[2], 0.01)
    brightness_delta = after[:3].mean(axis=0) - before[:3].mean(axis=0)
    evidence = valid & (score >= 4.5) & (np.abs(brightness_delta) >= 0.03) & (before_ndvi <= 0.25) & (after_ndvi <= 0.25)
    return evidence, valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--v2-candidates", type=Path, required=True)
    parser.add_argument("--aoi-id", required=True)
    parser.add_argument("--sar-dir", type=Path, required=True, help="Cache for Sentinel-1 crops on the Sentinel-2 grid")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, help="Blind v2 review decisions to score against")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.force and (args.output_dir / "v4-metrics.json").exists():
        raise FileExistsError("output exists; pass --force")
    candidates = json.loads(args.v2_candidates.read_text(encoding="utf-8"))["features"]
    indexed, sources = load_s2(args.crop_manifest, args.source_manifest, args.aoi_id)
    client = Client.open(STAC_URL)
    token = sas_token()
    with rasterio.open(indexed[("before", "B04")]) as reference:
        bbox = transform_bounds(reference.crs, "EPSG:4326", *reference.bounds)
        evidence_all, s2_valid = v2_evidence(indexed, reference)
        composites: dict[tuple[str, str], np.ndarray] = {}
        sar_rows = []
        for role in ("before", "after"):
            center = datetime.fromisoformat(sources[role]["acquired_utc"].replace("Z", "+00:00"))
            items = search_window(client, bbox, center)
            for pol in POLARIZATIONS:
                layers = []
                for item in items:
                    href = item.assets[pol].href
                    output = args.sar_dir / args.aoi_id / role / f"{item.id}_{pol}.tif"
                    layers.append(read_on_grid(f"{href}?{token}", reference, output, args.force))
                    sar_rows.append({"aoi_id": args.aoi_id, "role": role, "item_id": item.id, "acquired_utc": item.datetime.isoformat(), "polarization": pol, "source_href": href, "output_path": str(output), "sha256": sha256(output)})
                composites[(role, pol)] = composite_db(np.stack(layers))
        aggregates = []
        for feature in candidates:
            projected = transform_geom("EPSG:4326", reference.crs, feature["geometry"], precision=3)
            aggregates.append(rasterize([(projected, 1)], out_shape=reference.shape, transform=reference.transform, fill=0, dtype=np.uint8).astype(bool))
        union = np.any(aggregates, axis=0) if aggregates else np.zeros(reference.shape, dtype=bool)
        near_union = ndimage.binary_dilation(union, iterations=BACKGROUND_BUFFER_PIXELS)
        sar_valid = np.all([np.isfinite(values) for values in composites.values()], axis=0)
        background = s2_valid & sar_valid & ~near_union
        deltas, calibration = {}, {}
        for pol in POLARIZATIONS:
            raw = composites[("after", pol)] - composites[("before", pol)]
            offset = float(np.median(raw[background]))
            sigma = float(1.4826 * np.median(np.abs(raw[background] - offset)))
            deltas[pol] = raw - offset
            calibration[pol] = {"background_median_db": round(offset, 4), "background_robust_sigma_db": round(sigma, 4)}
    retained, decisions = [], []
    for feature, aggregate in zip(candidates, aggregates, strict=True):
        pixels = aggregate & evidence_all & sar_valid
        n = int(pixels.sum())
        m_vv = float(np.median(deltas["vv"][pixels])) if n else float("nan")
        m_vh = float(np.median(deltas["vh"][pixels])) if n else float("nan")
        status = decide(feature["properties"]["change_type"], m_vv, m_vh, touches_edge(aggregate), n)
        decisions.append({"candidate_id": feature["id"], "v2_change_type": feature["properties"]["change_type"], "sar_pixels": n, "median_dvv_db": round(m_vv, 3), "median_dvh_db": round(m_vh, 3), "status": status})
        if status == "retained":
            kept = json.loads(json.dumps(feature))
            kept["id"] = feature["id"].replace("-v2-", "-v4-")
            kept["properties"].update({"method": METHOD, "review_state": "needs_review", "review_note": "", "sar_pixels": n, "median_dvv_db": round(m_vv, 3), "median_dvh_db": round(m_vh, 3)})
            retained.append(kept)
    scene_counts = {role: len({row["item_id"] for row in sar_rows if row["role"] == role}) for role in ("before", "after")}
    metrics: dict[str, Any] = {"aoi_id": args.aoi_id, "method": METHOD, "scene_counts": scene_counts, "calibration": calibration, "background_pixels": int(background.sum()), "status_counts": {}}
    for row in decisions:
        metrics["status_counts"][row["status"]] = metrics["status_counts"].get(row["status"], 0) + 1
    if args.review_csv:
        with args.review_csv.open(newline="", encoding="utf-8") as handle:
            reviews = {row["candidate_id"]: row["review_state"] for row in csv.DictReader(handle)}
        metrics["evaluation"] = evaluate({row["candidate_id"]: row["status"] for row in decisions}, reviews)
    for name, rows in (("sar-manifest.csv", sar_rows), ("sar-decisions.csv", decisions)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    collection = {"type": "FeatureCollection", "name": f"{args.aoi_id}_v4_candidates", "features": retained}
    (args.output_dir / "change-candidates.geojson").write_text(json.dumps(collection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "v4-metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
