"""Fréchet Inception Distance for generated RS imagery.

torchmetrics' ``FrechetInceptionDistance`` (InceptionV3 backbone). Stage-agnostic:
point ``gen_dir`` at whatever the model produced — LR 128² grids, SR 256² grids,
or full-cascade 256² — and ``real_*`` at the matching RSICD split. Both sides are
resized to ``image_size`` before the metric (Inception rescales to 299 internally),
so keep ``image_size`` equal to the stage you are scoring (128 for LR, 256 for SR /
cascade) and compare only against numbers computed the same way.

The thesis FID snippet (commented out in the legacy non-cascade DDPM script) used
``feature=64``; standard, paper-comparable FID is ``feature=2048``. Both are exposed
so we can reproduce the thesis number *and* report a comparable one.

Note: FID is biased for small samples. Use the full test split (~1093 generations),
not a 16-image grid, for a number worth quoting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class FIDResult:
    fid: float
    feature: int
    image_size: int
    n_real: int
    n_gen: int


def _load_uint8(path: Path, size: int) -> torch.Tensor:
    """Load one image as a CxHxW uint8 tensor at (size, size)."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    t = torch.frombuffer(img.tobytes(), dtype=torch.uint8)
    return t.view(size, size, 3).permute(2, 0, 1).contiguous()


def _real_paths(real_csv: Path | None, real_dir: Path, split: str) -> list[Path]:
    if real_csv is not None:
        df = pd.read_csv(real_csv)
        df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows for split={split!r} in {real_csv}")
        return [real_dir / fn for fn in df["filename"]]
    return [p for p in sorted(real_dir.iterdir()) if p.suffix.lower() in _IMG_EXTS]


def _gen_paths(gen_dir: Path) -> list[Path]:
    return sorted(p for p in gen_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS)


def fid(
    gen_dir: str | Path,
    real_csv: str | Path | None = None,
    real_dir: str | Path | None = None,
    split: str = "test",
    feature: int = 2048,
    image_size: int = 256,
    batch_size: int = 32,
    max_n: int | None = None,
    device: str | None = None,
) -> FIDResult:
    from torchmetrics.image.fid import FrechetInceptionDistance

    device = device or _auto_device()
    gen_dir = Path(gen_dir)
    if real_dir is None and real_csv is None:
        raise ValueError("provide real_csv (+ real_dir) or real_dir")
    real_dir = Path(real_dir) if real_dir is not None else None
    real_csv = Path(real_csv) if real_csv is not None else None
    if real_csv is not None and real_dir is None:
        raise ValueError("real_csv needs real_dir (image root for its filenames)")

    real = _real_paths(real_csv, real_dir, split)
    gen = _gen_paths(gen_dir)
    if max_n is not None:
        real, gen = real[:max_n], gen[:max_n]
    if not real or not gen:
        raise ValueError(f"empty set (real={len(real)}, gen={len(gen)})")

    metric = FrechetInceptionDistance(feature=feature, normalize=False).to(device)

    def _feed(paths: list[Path], real_flag: bool, desc: str) -> int:
        buf: list[torch.Tensor] = []
        n = 0
        for p in tqdm(paths, desc=desc):
            if not p.exists():
                continue
            try:
                buf.append(_load_uint8(p, image_size))
            except Exception:
                continue
            if len(buf) >= batch_size:
                metric.update(torch.stack(buf).to(device), real=real_flag)
                n += len(buf)
                buf.clear()
        if buf:
            metric.update(torch.stack(buf).to(device), real=real_flag)
            n += len(buf)
        return n

    n_real = _feed(real, True, "fid/real")
    n_gen = _feed(gen, False, "fid/gen")

    return FIDResult(
        fid=float(metric.compute().item()),
        feature=feature,
        image_size=image_size,
        n_real=n_real,
        n_gen=n_gen,
    )
