from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parents[1] / "scripts" / "commercial"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("filter_change_persistence_v3", SCRIPTS / "filter_change_persistence_v3.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_persistence_keeps_state_close_to_after() -> None:
    before = np.zeros((2, 2, 2), dtype=np.float32)
    after = np.ones_like(before)
    followup = np.ones_like(before) * 1.1
    mask, fraction = MODULE.persistence_mask(before, after, followup, np.ones((2, 2), dtype=bool))
    assert mask.all()
    assert fraction == 1


def test_persistence_rejects_reversal() -> None:
    before = np.zeros((1, 2, 2), dtype=np.float32)
    after = np.ones_like(before)
    followup = np.zeros_like(before)
    mask, fraction = MODULE.persistence_mask(before, after, followup, np.ones((2, 2), dtype=bool))
    assert not mask.any()
    assert fraction == 0
