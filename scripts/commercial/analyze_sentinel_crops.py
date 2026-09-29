#!/usr/bin/env python3
"""Measure SCL quality and render fixed-scale Sentinel-2 RGB quicklooks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw

SCL_NAMES = {
    0: "no_data", 1: "saturated_or_defective", 2: "dark_area", 3: "cloud_shadow",
    4: "vegetation", 5: "not_vegetated", 6: "water", 7: "unclassified",
    8: "cloud_medium_probability", 9: "cloud_high_probability", 10: "thin_cirrus",
    11: "snow_or_ice",
}
CLOUD_CLASSES = {8, 9, 10}
SHADOW_CLASS = 3


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scl_metrics(values: np.ndarray) -> dict[str, object]:
    counts = Counter(int(value) for value in values.ravel())
    valid = values.size - counts[0]
    clouds = sum(counts[value] for value in CLOUD_CLASSES)
    shadows = counts[SHADOW_CLASS]
    denominator = max(valid, 1)
    return {
        "total_pixels": int(values.size),
        "valid_pixels": int(valid),
        "cloud_pixels": int(clouds),
        "shadow_pixels": int(shadows),
        "cloud_pct_valid": round(100 * clouds / denominator, 6),
        "shadow_pct_valid": round(100 * shadows / denominator, 6),
        "class_counts": {SCL_NAMES.get(key, f"class_{key}"): value for key, value in sorted(counts.items())},
    }


def fixed_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Convert baseline-05.xx Sentinel L2A DN to comparable display RGB."""
    stacked = np.stack([red, green, blue]).astype(np.float32)
    reflectance = stacked * 0.0001 - 0.1
    display = np.clip(reflectance / 0.4, 0, 1) ** (1 / 2.2)
    return np.moveaxis(np.round(display * 255).astype(np.uint8), 0, -1)


def read_band(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    with rasterio.open(path) as source:
        return source.read(1), {"shape": source.shape, "transform": source.transform, "crs": source.crs}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as tmp:
        json.dump(value, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        temp_name = tmp.name
    os.replace(temp_name, path)


def labeled_panel(image: Image.Image, label: str) -> Image.Image:
    header = 42
    panel = Image.new("RGB", (image.width, image.height + header), "white")
    panel.paste(image, (0, header))
    ImageDraw.Draw(panel).text((12, 13), label, fill="black")
    return panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    crops = read_csv(args.crop_manifest)
    sources = read_csv(args.source_manifest)
    dates = {(row["aoi_id"], row["role"]): row["acquired_utc"] for row in sources}
    indexed = {(row["aoi_id"], row["role"], row["band"]): Path(row["output_path"]) for row in crops}
    aois = sorted({row["aoi_id"] for row in crops})
    results: dict[str, object] = {"display": {"reflectance_min": 0.0, "reflectance_max": 0.4, "gamma": 2.2}, "aois": {}}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for aoi_id in aois:
        aoi_result: dict[str, object] = {}
        panels: list[Image.Image] = []
        available_roles = {row["role"] for row in crops if row["aoi_id"] == aoi_id}
        roles = [role for role in ("before", "after", "followup") if role in available_roles]
        for role in roles:
            arrays = {}
            metadata = {}
            for band in ("B04", "B03", "B02", "SCL"):
                arrays[band], metadata[band] = read_band(indexed[(aoi_id, role, band)])
            reference = metadata["B04"]
            if any(metadata[band] != reference for band in ("B03", "B02")):
                raise ValueError(f"RGB grids do not align for {aoi_id}/{role}")
            rgb = fixed_rgb(arrays["B04"], arrays["B03"], arrays["B02"])
            image = Image.fromarray(rgb, "RGB")
            quicklook = args.output_dir / f"{aoi_id}_{role}_rgb.png"
            image.save(quicklook, optimize=True)
            label = f"{role.upper()}  {dates[(aoi_id, role)]}  Sentinel-2 L2A"
            panels.append(labeled_panel(image, label))
            aoi_result[role] = {"acquired_utc": dates[(aoi_id, role)], "scl": scl_metrics(arrays["SCL"]), "quicklook": str(quicklook)}

        comparison = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height), "white")
        left = 0
        for panel in panels:
            comparison.paste(panel, (left, 0))
            left += panel.width
        comparison_path = args.output_dir / f"{aoi_id}_timeline.png"
        comparison.save(comparison_path, optimize=True)
        aoi_result["comparison"] = str(comparison_path)
        results["aois"][aoi_id] = aoi_result

    atomic_json(args.output_dir / "quality-metrics.json", results)
    print(f"wrote metrics and quicklook/timeline images for {len(aois)} AOIs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
