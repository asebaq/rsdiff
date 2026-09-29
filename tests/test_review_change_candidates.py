from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "commercial" / "review_change_candidates.py"
SPEC = importlib.util.spec_from_file_location("review_change_candidates", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_boundary_returns_only_perimeter() -> None:
    mask = np.ones((3, 3), dtype=np.uint8)
    result = MODULE.boundary(mask)
    assert result.sum() == 8
    assert not result[1, 1]


def test_validate_review_requires_rejection_reason() -> None:
    rows = [{"candidate_id": "one", "review_state": "rejected", "change_type": "rejected", "review_note": ""}]
    with pytest.raises(ValueError, match="needs a reason"):
        MODULE.validate_review(rows, {"one"}, require_complete=True)


def test_validate_review_accepts_complete_decision() -> None:
    rows = [{"candidate_id": "one", "review_state": "accepted", "change_type": "road_network_formation", "review_note": "visible linear road extension"}]
    MODULE.validate_review(rows, {"one"}, require_complete=True)
