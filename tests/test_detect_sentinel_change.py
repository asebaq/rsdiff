from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "commercial" / "detect_sentinel_change.py"
SPEC = importlib.util.spec_from_file_location("detect_sentinel_change", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_robust_score_centers_unchanged_pixels_and_finds_outlier() -> None:
    before = np.zeros((2, 3, 3), dtype=np.uint16)
    after = np.array([
        [[1, 2, 1], [2, 30, 2], [1, 2, 1]],
        [[2, 4, 2], [4, 40, 4], [2, 4, 2]],
    ], dtype=np.uint16)
    score, stats = MODULE.robust_score(before, after, np.ones((3, 3), dtype=bool))
    assert score[1, 1] > 5
    assert score[0, 0] < 1
    assert len(stats) == 2


def test_robust_score_rejects_zero_mad() -> None:
    values = np.ones((1, 2, 2), dtype=np.uint16)
    with pytest.raises(ValueError, match="zero robust difference scale"):
        MODULE.robust_score(values, values + 1, np.ones((2, 2), dtype=bool))


def test_polygon_area_subtracts_hole() -> None:
    geometry = {"type": "Polygon", "coordinates": [
        [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
        [[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]],
    ]}
    assert MODULE.polygon_area(geometry) == 96
