from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


@dataclass
class Sample:
    image: Any
    caption: str | None = None
    label: str | None = None
    meta: dict[str, Any] | None = None


class RSDataset(Dataset, ABC):
    """Common interface for all remote-sensing datasets in rsdiff.

    Implementations should return a :class:`Sample` from ``__getitem__``.
    A torchvision-style ``transform`` may be applied to ``image`` before return.
    """

    name: str = "base"
    splits: tuple[str, ...] = ("train", "val", "test")

    def __init__(self, root: str | Path, split: str = "train", transform=None) -> None:
        if split not in self.splits:
            raise ValueError(f"split={split!r} not in {self.splits}")
        self.root = Path(root)
        self.split = split
        self.transform = transform

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Sample: ...

    @property
    def classes(self) -> list[str]:
        return []
