#!/usr/bin/env python3
"""Prepare visual review cards and finalize reviewed change candidates."""

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
from rasterio.features import rasterize
from rasterio.warp import transform_geom

CHANGE_TYPES = {
    "major_clearing_or_grading",
    "large_structural_footprint",
    "road_network_formation",
    "large_laydown_or_stockpile_change",
    "rejected",
    "needs_review",
}
REVIEW_STATES = {"accepted", "rejected", "needs_review"}
CSV_FIELDS = [
    "candidate_id", "algorithm_confidence", "area_m2", "score_p90",
    "review_state", "change_type", "review_note",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_review(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, newline="", encoding="utf-8", delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        temporary = tmp.name
    os.replace(temporary, path)


def validate_review(rows: list[dict[str, str]], candidate_ids: set[str], require_complete: bool) -> None:
    ids = [row.get("candidate_id", "") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("review CSV contains duplicate candidate IDs")
    if set(ids) != candidate_ids:
        raise ValueError("review CSV candidate IDs do not match candidate GeoJSON")
    for row in rows:
        state, change_type = row.get("review_state"), row.get("change_type")
        if state not in REVIEW_STATES or change_type not in CHANGE_TYPES:
            raise ValueError(f"invalid decision for {row.get('candidate_id')}: {state}/{change_type}")
        if state == "accepted" and change_type in {"rejected", "needs_review"}:
            raise ValueError(f"accepted candidate lacks accepted change type: {row['candidate_id']}")
        if state == "rejected" and change_type != "rejected":
            raise ValueError(f"rejected candidate must use rejected change type: {row['candidate_id']}")
        if state == "needs_review" and change_type != "needs_review":
            raise ValueError(f"pending candidate must use needs_review change type: {row['candidate_id']}")
        if state == "rejected" and not row.get("review_note", "").strip():
            raise ValueError(f"rejected candidate needs a reason: {row['candidate_id']}")
        if require_complete and state == "needs_review":
            raise ValueError(f"review is incomplete: {row['candidate_id']}")


def boundary(mask: np.ndarray) -> np.ndarray:
    interior = mask.copy()
    interior[1:, :] &= mask[:-1, :]
    interior[:-1, :] &= mask[1:, :]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    return mask & ~interior


def overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgba = image.convert("RGBA")
    fill = Image.new("RGBA", rgba.size, (255, 0, 0, 65))
    rgba.alpha_composite(Image.composite(fill, Image.new("RGBA", rgba.size), Image.fromarray(mask * 255)))
    edge = Image.fromarray(boundary(mask).astype(np.uint8) * 255)
    rgba.alpha_composite(Image.composite(Image.new("RGBA", rgba.size, (255, 0, 0, 255)), Image.new("RGBA", rgba.size), edge))
    return rgba.convert("RGB")


def render_cards(features: list[dict[str, Any]], reference_path: Path, before_path: Path, after_path: Path, output_dir: Path, followup_path: Path | None = None) -> None:
    images = [Image.open(before_path).convert("RGB"), Image.open(after_path).convert("RGB")]
    if followup_path:
        images.append(Image.open(followup_path).convert("RGB"))
    with rasterio.open(reference_path) as reference:
        if any(image.size != (reference.width, reference.height) for image in images):
            raise ValueError("quicklooks and reference raster dimensions do not match")
        full_shape = reference.shape
        transform, crs = reference.transform, reference.crs

    cards = []
    cards_dir = output_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    for feature in features:
        projected = transform_geom("EPSG:4326", crs, feature["geometry"], precision=3)
        mask = rasterize([(projected, 1)], out_shape=full_shape, transform=transform, fill=0, dtype=np.uint8)
        rows, cols = np.nonzero(mask)
        if not len(rows):
            raise ValueError(f"candidate does not overlap reference: {feature['id']}")
        pad = 24
        top, bottom = max(0, int(rows.min()) - pad), min(full_shape[0], int(rows.max()) + pad + 1)
        left, right = max(0, int(cols.min()) - pad), min(full_shape[1], int(cols.max()) + pad + 1)
        crop_mask = mask[top:bottom, left:right]
        crops = [overlay(image.crop((left, top, right, bottom)), crop_mask) for image in images]
        pair = Image.new("RGB", (220 * len(crops), 250), "white")
        for image_index, crop in enumerate(crops):
            pair.paste(crop.resize((220, 220), Image.Resampling.BILINEAR), (220 * image_index, 30))
        props = feature["properties"]
        area = props.get("area_m2", props.get("evidence_area_m2"))
        if area is None:
            raise ValueError(f"candidate lacks area: {feature['id']}")
        sequence = "BEFORE / AFTER / FOLLOWUP" if followup_path else "BEFORE / AFTER"
        ImageDraw.Draw(pair).text((8, 8), f"{feature['id']} | {area:.0f} m2 | p90 {props['score_p90']:.2f} | {sequence}", fill="black")
        card_path = cards_dir / f"{feature['id']}.png"
        pair.save(card_path, optimize=True)
        cards.append(pair)

    for page_index in range(0, len(cards), 6):
        page_cards = cards[page_index:page_index + 6]
        card_width = cards[0].width
        sheet = Image.new("RGB", (card_width * 2, 750), "white")
        for position, card in enumerate(page_cards):
            sheet.paste(card, ((position % 2) * card_width, (position // 2) * 250))
        sheet.save(output_dir / f"review-sheet-{page_index // 6 + 1:02d}.png", optimize=True)


def finalize(candidates: dict[str, Any], rows: list[dict[str, str]], output_dir: Path) -> None:
    validate_review(rows, {feature["id"] for feature in candidates["features"]}, require_complete=True)
    decisions = {row["candidate_id"]: row for row in rows}
    reviewed = json.loads(json.dumps(candidates))
    counts = {state: 0 for state in ("accepted", "rejected")}
    type_counts: dict[str, int] = {}
    for feature in reviewed["features"]:
        decision = decisions[feature["id"]]
        feature["properties"].update({key: decision[key] for key in ("review_state", "change_type", "review_note")})
        counts[decision["review_state"]] += 1
        type_counts[decision["change_type"]] = type_counts.get(decision["change_type"], 0) + 1
    total = len(rows)
    metrics = {"total": total, **counts, "usefulness_pct": round(100 * counts["accepted"] / max(total, 1), 2), "change_types": type_counts}
    for name, value in (("reviewed-change.geojson", reviewed), ("review-metrics.json", metrics)):
        path = output_dir / name
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--reference-raster", type=Path, required=True)
    parser.add_argument("--before-quicklook", type=Path, required=True)
    parser.add_argument("--after-quicklook", type=Path, required=True)
    parser.add_argument("--followup-quicklook", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path, help="Review decisions; defaults to OUTPUT_DIR/review.csv")
    parser.add_argument("--finalize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = read_json(args.candidates)
    features = candidates.get("features", [])
    review_path = args.review_csv or args.output_dir / "review.csv"
    candidate_ids = {feature["id"] for feature in features}
    if not review_path.exists():
        rows = [{
            "candidate_id": feature["id"], "algorithm_confidence": feature["properties"]["confidence"],
            "area_m2": feature["properties"].get("area_m2", feature["properties"].get("evidence_area_m2")), "score_p90": feature["properties"]["score_p90"],
            "review_state": "needs_review", "change_type": "needs_review", "review_note": "",
        } for feature in features]
        write_csv_atomic(review_path, rows)
    rows = read_review(review_path)
    validate_review(rows, candidate_ids, require_complete=False)
    render_cards(features, args.reference_raster, args.before_quicklook, args.after_quicklook, args.output_dir, args.followup_quicklook)
    if args.finalize:
        finalize(candidates, rows, args.output_dir)
    print(f"prepared {len(features)} candidates in {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
