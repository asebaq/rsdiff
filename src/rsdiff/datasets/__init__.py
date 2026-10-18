from rsdiff.datasets.base import RSDataset
from rsdiff.datasets.rsicd import RSICD

__all__ = ["RSDataset", "RSICD", "build_dataset"]


_REGISTRY = {
    "rsicd": RSICD,
}


def build_dataset(name: str, **kwargs) -> RSDataset:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)
