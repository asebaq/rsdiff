#!/usr/bin/env python3
"""Materialize the arampacha/rsicd HF dataset into the thesis layout.

Output:
    <out>/RSICD_images/<basename>.jpg
    <out>/dataset_rsicd.csv   columns: filename,split,imgid,sent1..sent5,label

The HF mirror uses Parquet with columns `filename,captions,image` and splits
`train|test|valid`. We strip the `rsicd_images/` prefix from filenames, map
`valid` -> `val` to match the local thesis CSV, and infer `label` from the
filename prefix (e.g. `airport_12.jpg` -> `airport`).

Split-count delta vs local CSV: HF mirror has train/test/valid = 8730/1090/1090
while the local thesis CSV has 8734/1093/1094 (same 10921 images, ~11 rows
reshuffled across splits). Set --split-csv to override splits from a local CSV.
"""
from __future__ import annotations

import argparse
import csv
import pathlib

from datasets import load_dataset

SPLIT_MAP = {"train": "train", "test": "test", "valid": "val", "validation": "val"}
COLUMNS = ["filename", "split", "imgid", "sent1", "sent2", "sent3", "sent4", "sent5", "label"]


def label_from_filename(name: str) -> str:
    stem = name.rsplit(".", 1)[0]
    return stem.rsplit("_", 1)[0]


def normalize_caption(s: str) -> str:
    """Match the thesis CSV's lowercase, space-before-period style.

    The HF mirror's captions are the cleaned/recapitalized variant; the model
    was trained on the lowercase originals. Lowercase + insert a space before
    the trailing period when missing, so T5 sees inputs the same shape as
    during the 2024 thesis runs.
    """
    s = s.strip().lower()
    if s.endswith(".") and not s.endswith(" ."):
        s = s[:-1].rstrip() + " ."
    return s


def load_split_override(path: pathlib.Path) -> dict[str, str]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return {row["filename"]: row["split"] for row in reader}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/RSICD_optimal"))
    p.add_argument("--repo", default="arampacha/rsicd")
    p.add_argument("--split-csv", type=pathlib.Path, default=None,
                   help="Optional local CSV; overrides HF splits using its filename->split map.")
    p.add_argument("--quality", type=int, default=95, help="JPEG quality for re-encoded images.")
    args = p.parse_args()

    img_dir = args.out / "RSICD_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "dataset_rsicd.csv"

    split_override = load_split_override(args.split_csv) if args.split_csv else None
    if split_override is not None:
        print(f"split-csv override: {len(split_override)} rows from {args.split_csv}")

    ds = load_dataset(args.repo)
    rows: list[dict[str, object]] = []
    imgid = 0
    for split_name, split in ds.items():
        split_local = SPLIT_MAP.get(split_name, split_name)
        for ex in split:
            fname = pathlib.Path(ex["filename"]).name
            captions = [normalize_caption(c) for c in ex["captions"]]
            if len(captions) < 5:
                captions += [captions[-1]] * (5 - len(captions))
            captions = captions[:5]

            img = ex["image"]
            img.convert("RGB").save(img_dir / fname, format="JPEG", quality=args.quality)

            row_split = split_override.get(fname, split_local) if split_override else split_local
            rows.append({
                "filename": fname,
                "split": row_split,
                "imgid": imgid,
                "sent1": captions[0],
                "sent2": captions[1],
                "sent3": captions[2],
                "sent4": captions[3],
                "sent5": captions[4],
                "label": label_from_filename(fname),
            })
            imgid += 1

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    by_split: dict[str, int] = {}
    for r in rows:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
    print(f"wrote {len(rows)} rows -> {csv_path}")
    print(f"images -> {img_dir}")
    print(f"splits: {by_split}")


if __name__ == "__main__":
    main()
