from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "scripts" / "commercial" / "analyze_sentinel_crops.py"
SPEC = importlib.util.spec_from_file_location("analyze_sentinel_crops", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_scl_metrics_excludes_nodata_from_percentages() -> None:
    result = MODULE.scl_metrics(np.array([[0, 3], [8, 5]], dtype=np.uint8))
    assert result["valid_pixels"] == 3
    assert result["cloud_pct_valid"] == 33.333333
    assert result["shadow_pct_valid"] == 33.333333


def test_fixed_rgb_uses_physical_fixed_scale() -> None:
    values = np.array([[1000, 5000, 6000]], dtype=np.uint16)
    result = MODULE.fixed_rgb(values, values, values)
    assert result.shape == (1, 3, 3)
    assert result[0, 0, 0] == 0
    assert result[0, 1, 0] == 255
    assert result[0, 2, 0] == 255
