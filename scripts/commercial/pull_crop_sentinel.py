#!/usr/bin/env python3
"""Resolve Sentinel-2 STAC assets and crop observed bands to tracked AOIs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

BAND_ASSETS = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir", "B11": "swir16", "SCL": "scl"}
DEFAULT_BANDS = tuple(BAND_ASSETS)
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class Task:
    aoi_id: str
    role: str
    item_id: str
    stac_url: str
    geometry: dict[str, Any]


def _safe_id(value: str, field: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"unsafe {field}: {value!r}")
    return value


def _https_url(value: str, field: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{field} must be an HTTPS URL: {value!r}")
    return value


def load_aois(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") != "FeatureCollection":
        raise ValueError("AOI file must be a GeoJSON FeatureCollection")
    aois: dict[str, dict[str, Any]] = {}
    for feature in data.get("features", []):
        aoi_id = _safe_id(str(feature.get("properties", {}).get("aoi_id", "")), "aoi_id")
        geometry = feature.get("geometry")
        if not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise ValueError(f"AOI {aoi_id!r} must have Polygon or MultiPolygon geometry")
        if aoi_id in aois:
            raise ValueError(f"duplicate aoi_id: {aoi_id}")
        aois[aoi_id] = geometry
    if not aois:
        raise ValueError("AOI file contains no features")
    return aois


def load_tasks(manifest: Path, aois: dict[str, dict[str, Any]]) -> list[Task]:
    tasks: list[Task] = []
    seen: set[tuple[str, str]] = set()
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"aoi_id", "role", "item_id", "stac_url", "selection_status"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"manifest missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            if row["selection_status"] not in {"provisional", "selected", "backup"}:
                continue
            aoi_id = _safe_id(row["aoi_id"], "aoi_id")
            role = _safe_id(row["role"], "role")
            item_id = _safe_id(row["item_id"], "item_id")
            if aoi_id not in aois:
                raise ValueError(f"manifest references unknown AOI: {aoi_id}")
            key = (aoi_id, role)
            if key in seen:
                raise ValueError(f"duplicate task for AOI/role: {aoi_id}/{role}")
            seen.add(key)
            tasks.append(Task(aoi_id, role, item_id, _https_url(row["stac_url"], "stac_url"), aois[aoi_id]))
    if not tasks:
        raise ValueError("manifest contains no provisional, selected, or backup tasks")
    return tasks


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "rsdiff-commercial-validation/1"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        if response.headers.get_content_type() not in {"application/json", "application/geo+json"}:
            raise ValueError(f"unexpected STAC content type: {response.headers.get_content_type()}")
        return json.load(response)


def resolve_assets(task: Task, bands: tuple[str, ...]) -> list[tuple[str, str]]:
    item = fetch_json(_https_url(task.stac_url, "stac_url"))
    if item.get("id") != task.item_id:
        raise ValueError(f"STAC ID mismatch: expected {task.item_id}, received {item.get('id')}")
    assets = item.get("assets", {})
    resolved = []
    for band in bands:
        key = BAND_ASSETS[band]
        asset = assets.get(key)
        if not asset or "href" not in asset:
            raise ValueError(f"STAC item {task.item_id} has no {band}/{key} asset")
        resolved.append((band, _https_url(asset["href"], f"{band} asset")))
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crop_asset(href: str, geometry: dict[str, Any], output: Path, force: bool) -> dict[str, Any]:
    try:
        import rasterio
        from rasterio.mask import mask
        from rasterio.warp import transform_geom
    except ImportError as exc:
        raise RuntimeError("rasterio is required; install with: pip install -e '.[geo]'") from exc
    if output.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MULTIRANGE="YES"), rasterio.open(href) as source:
        projected = transform_geom("EPSG:4326", source.crs, geometry, precision=3)
        pixels, transform = mask(source, [projected], crop=True, filled=True)
        profile = source.profile.copy()
        profile.update(driver="GTiff", height=pixels.shape[1], width=pixels.shape[2], transform=transform, compress="deflate", predictor=2, tiled=pixels.shape[1] >= 256 and pixels.shape[2] >= 256)
        source_crs, source_nodata, source_dtype = str(source.crs), source.nodata, source.dtypes[0]
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".tif", delete=False) as tmp:
            temp_name = tmp.name
        with rasterio.open(temp_name, "w", **profile) as target:
            target.write(pixels)
            target.update_tags(SOURCE_HREF=href, PROCESSING="AOI crop only; observed pixels")
        os.replace(temp_name, output)
        temp_name = None
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)
    return {"source_href": href, "output_path": str(output), "sha256": sha256(output), "crs": source_crs, "dtype": source_dtype, "nodata": "" if source_nodata is None else source_nodata, "width": pixels.shape[2], "height": pixels.shape[1]}


def _write_output_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["aoi_id", "role", "item_id", "band", "source_href", "output_path", "sha256", "crs", "dtype", "nodata", "width", "height"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, newline="", encoding="utf-8", delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temp_name = tmp.name
    os.replace(temp_name, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--aoi", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bands", nargs="+", choices=sorted(BAND_ASSETS), default=list(DEFAULT_BANDS))
    parser.add_argument("--aoi-id", action="append", help="Only process this AOI; repeatable")
    parser.add_argument("--role", action="append", help="Only process this temporal role; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print assets without reading pixels")
    parser.add_argument("--force", action="store_true", help="Replace existing crop files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    aois = load_aois(args.aoi)
    tasks = load_tasks(args.manifest, aois)
    if args.aoi_id:
        tasks = [task for task in tasks if task.aoi_id in args.aoi_id]
    if args.role:
        tasks = [task for task in tasks if task.role in args.role]
    if not tasks:
        raise ValueError("task filters selected no manifest rows")
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for band, href in resolve_assets(task, tuple(args.bands)):
            output = args.output_dir / task.aoi_id / task.role / f"{task.item_id}_{band}.tif"
            print(f"{task.aoi_id}/{task.role} {band}: {href}")
            if not args.dry_run:
                rows.append({"aoi_id": task.aoi_id, "role": task.role, "item_id": task.item_id, "band": band, **crop_asset(href, task.geometry, output, args.force)})
    if not args.dry_run:
        _write_output_manifest(args.output_dir / "crop-manifest.csv", rows)
        print(f"wrote {len(rows)} crops and {args.output_dir / 'crop-manifest.csv'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileExistsError, RuntimeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
