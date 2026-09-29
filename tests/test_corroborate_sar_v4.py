from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts" / "commercial"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("corroborate_sar_v4", SCRIPTS / "corroborate_sar_v4.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_structural_footprint_requires_vv_increase_not_decrease() -> None:
    assert MODULE.decide("large_structural_footprint", 1.0, 0.0, False, 30) == "retained"
    assert MODULE.decide("large_structural_footprint", -3.0, -3.0, False, 30) == "sar_unchanged"
    assert MODULE.decide("large_structural_footprint", 0.99, 5.0, False, 30) == "sar_unchanged"


def test_other_classes_accept_change_in_either_direction_or_polarization() -> None:
    assert MODULE.decide("road_network_formation", -1.2, 0.1, False, 30) == "retained"
    assert MODULE.decide("major_clearing_or_grading", 0.2, 1.0, False, 30) == "retained"
    assert MODULE.decide("large_laydown_or_stockpile_change", 0.9, -0.9, False, 30) == "sar_unchanged"


def test_edge_and_insufficient_pixels_override_strong_signal() -> None:
    assert MODULE.decide("road_network_formation", 5.0, 5.0, True, 500) == "aoi_edge"
    assert MODULE.decide("road_network_formation", 5.0, 5.0, False, 29) == "sar_insufficient"


def test_composite_needs_three_valid_looks_and_ignores_nodata() -> None:
    stack = np.array([[[0.1, 0.1]], [[0.1, np.nan]], [[0.1, 0.0]], [[1000.0, 0.1]]], dtype=np.float32)
    result = MODULE.composite_db(stack)
    assert result[0, 0] == pytest.approx(-10.0)
    assert np.isnan(result[0, 1])


def test_touches_edge_uses_ten_pixel_border() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:20, 10:20] = True
    assert not MODULE.touches_edge(mask)
    mask[29, 15] = True
    assert not MODULE.touches_edge(mask)
    mask[30, 15] = True
    assert MODULE.touches_edge(mask)


def test_evaluate_gates_on_usefulness_recall_and_sample_size() -> None:
    reviews = {f"c{i}": "accepted" if i < 6 else "rejected" for i in range(10)}
    status = {f"c{i}": "retained" if i < 5 else "sar_unchanged" for i in range(10)}
    result = MODULE.evaluate(status, reviews)
    assert (result["v4_usefulness_pct"], result["v4_recall_pct"], result["verdict"]) == (100.0, 83.33, "pass")
    status = {f"c{i}": "retained" if i in (0, 1, 2, 6) else "sar_unchanged" for i in range(10)}
    assert MODULE.evaluate(status, reviews)["verdict"] == "inconclusive"
    status = {f"c{i}": "retained" if i in (0, 1, 2, 3, 6, 7) else "sar_unchanged" for i in range(10)}
    assert MODULE.evaluate(status, reviews)["verdict"] == "fail"


def test_evaluate_rejects_unreviewed_candidates() -> None:
    with pytest.raises(ValueError, match="c1"):
        MODULE.evaluate({"c0": "retained", "c1": "retained"}, {"c0": "accepted"})
