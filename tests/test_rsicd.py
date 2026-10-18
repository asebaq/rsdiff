from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from rsdiff.datasets import build_dataset
from rsdiff.datasets.rsicd import RSICD, sample_captions


@pytest.fixture
def rsicd_root(tmp_path: Path) -> Path:
    img_dir = tmp_path / "RSICD_images"
    img_dir.mkdir()
    rows = []
    for i in range(12):
        name = f"img_{i}.jpg"
        Image.new("RGB", (32, 32)).save(img_dir / name)
        rows.append(
            {
                "filename": name,
                "split": "train" if i < 2 else "test",
                "imgid": i,
                **{f"sent{k}": f"caption {i} {k}" for k in range(1, 6)},
                "label": "airport",
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "dataset_rsicd.csv", index=False)
    return tmp_path


def test_dataset_filters_split_and_pins_caption(rsicd_root: Path):
    ds = RSICD(root=rsicd_root, split="train", caption_idx=0)
    assert len(ds) == 2
    s = ds[0]
    assert s.image.size == (32, 32)
    assert s.caption == "caption 0 1"
    assert s.meta == {"filename": "img_0.jpg", "imgid": 0}
    assert ds.classes == ["airport"]


def test_unknown_dataset_is_rejected():
    with pytest.raises(KeyError):
        build_dataset("not-a-dataset", root=".")


def test_sample_captions_is_seeded_and_uses_first_caption(rsicd_root: Path):
    full = sample_captions(rsicd_root, split="test", seed=17)
    assert sorted(full) == sorted(f"caption {i} 1" for i in range(2, 12))
    assert full == sample_captions(rsicd_root, split="test", seed=17)
    three = sample_captions(rsicd_root, split="test", n=3, seed=17)
    assert len(three) == 3 and set(three) <= set(full)
    assert len(sample_captions(rsicd_root, split="test", n=999)) == 10


def test_sample_captions_rejects_empty_split(rsicd_root: Path):
    with pytest.raises(ValueError, match="split='val'"):
        sample_captions(rsicd_root, split="val")
