"""Smoke tests — no GPU, no network. Run with `pytest`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_version():
    import rsdiff

    assert rsdiff.__version__


def test_config_load(tmp_path: Path):
    from rsdiff.training.config import load_config

    cfg_yaml = """
data:
  name: rsicd
  image_size: 64
train:
  batch_size: 4
"""
    p = tmp_path / "c.yaml"
    p.write_text(cfg_yaml)
    cfg = load_config(p)
    assert cfg.data.image_size == 64
    assert cfg.train.batch_size == 4


def test_rsicd_dataset(tmp_path: Path):
    from PIL import Image

    from rsdiff.datasets.rsicd import RSICD

    img_dir = tmp_path / "RSICD_images"
    img_dir.mkdir()
    fnames = []
    for i in range(3):
        f = f"img_{i}.jpg"
        Image.new("RGB", (32, 32)).save(img_dir / f)
        fnames.append(f)

    rows = []
    for i, f in enumerate(fnames):
        rows.append({
            "filename": f,
            "split": "train" if i < 2 else "test",
            "imgid": i,
            "sent1": f"caption {i} a",
            "sent2": f"caption {i} b",
            "sent3": f"caption {i} c",
            "sent4": f"caption {i} d",
            "sent5": f"caption {i} e",
            "label": "airport",
        })
    pd.DataFrame(rows).to_csv(tmp_path / "dataset_rsicd.csv", index=False)

    ds = RSICD(root=tmp_path, split="train", caption_idx=0)
    assert len(ds) == 2
    s = ds[0]
    assert s.image.size == (32, 32)
    assert s.caption == "caption 0 a"
    assert s.label == "airport"
    assert s.meta["imgid"] == 0
    assert ds.classes == ["airport"]

    ds_test = RSICD(root=tmp_path, split="test")
    assert len(ds_test) == 1


def test_dataset_registry():
    from rsdiff.datasets import build_dataset

    with pytest.raises(KeyError):
        build_dataset("not-a-dataset", root=".")


def test_cli_parser():
    from rsdiff.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["train", "-c", "x.yaml"])
    assert args.config == Path("x.yaml")
