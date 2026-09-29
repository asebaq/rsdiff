from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "commercial" / "detect_sentinel_change_v2.py"
SPEC = importlib.util.spec_from_file_location("detect_sentinel_change_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_estimate_shift_recovers_subpixel_translation() -> None:
    before = np.zeros((64, 64), dtype=np.float32)
    before[15:45, 20:40] = 1
    after = MODULE.ndimage.shift(before, (0.5, -0.75), order=1, mode="constant", cval=0, prefilter=False)
    dy, dx, _ = MODULE.estimate_shift(before, after, np.ones_like(before, dtype=bool))
    assert abs(dy - (-0.5)) <= 0.5
    assert abs(dx - 0.75) <= 0.25


def test_classify_applies_frozen_gates() -> None:
    assert MODULE.classify(6500, 400, 20, 0.3, -0.04) == "road_network_formation"
    assert MODULE.classify(12000, 120, 120, 0.2, 0.04) == "large_structural_footprint"
    assert MODULE.classify(5500, 100, 100, 0.3, -0.04) is None
