from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from PIL import Image

from rsdiff.datasets.base import RSDataset, Sample


class RSICD(RSDataset):
    """Remote Sensing Image Captioning Dataset.

    Expects the standard layout::

        <root>/RSICD_images/*.jpg
        <root>/dataset_rsicd.csv

    The CSV has columns ``filename, split, imgid, sent1..sent5, label``.
    A random one of the five captions is returned per ``__getitem__`` call
    by default to act as caption-level augmentation; set ``caption_idx``
    to a value in ``0..4`` to pin a specific column.
    """

    name = "rsicd"
    splits = ("train", "val", "test")

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform=None,
        caption_idx: int | None = None,
        images_subdir: str = "RSICD_images",
        csv_name: str = "dataset_rsicd.csv",
    ) -> None:
        super().__init__(root, split=split, transform=transform)
        self.caption_idx = caption_idx
        self.images_dir = self.root / images_subdir
        csv_path = self.root / csv_name
        if not csv_path.is_file():
            raise FileNotFoundError(f"RSICD CSV not found at {csv_path}")
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self._caption_cols = [f"sent{i}" for i in range(1, 6)]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Sample:
        row = self.df.iloc[idx]
        img = Image.open(self.images_dir / row["filename"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.caption_idx is None:
            caption = row[random.choice(self._caption_cols)]
        else:
            caption = row[self._caption_cols[self.caption_idx]]

        return Sample(
            image=img,
            caption=str(caption),
            label=str(row.get("label", "")) or None,
            meta={"filename": row["filename"], "imgid": int(row["imgid"])},
        )

    @property
    def classes(self) -> list[str]:
        return sorted(self.df["label"].dropna().unique().tolist())


def sample_captions(
    root: str | Path,
    split: str = "test",
    n: int | None = None,
    seed: int = 17,
    csv_name: str = "dataset_rsicd.csv",
) -> list[str]:
    """Pick ``n`` first captions (``sent1``) from ``split``, shuffled with ``seed``.

    ``n=None`` returns every row of the split. With ``seed=17``, the full test split and a
    CSV built by ``scripts/build_rsicd_csv.py --split-csv``, this returns exactly the
    captions of the published 1,093-image evaluation bundle.
    """
    df = pd.read_csv(Path(root) / csv_name)
    df = df[df["split"] == split].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"no rows for split={split!r} in {Path(root) / csv_name}")
    k = len(df) if n is None else min(n, len(df))
    return df.sample(n=k, random_state=seed)["sent1"].astype(str).tolist()
