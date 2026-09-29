from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "commercial" / "pull_crop_sentinel.py"
SPEC = importlib.util.spec_from_file_location("pull_crop_sentinel", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_aoi(path: Path) -> None:
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"aoi_id": "site"}, "geometry": {"type": "Polygon", "coordinates": [[[31, 30], [32, 30], [32, 31], [31, 30]]]}}]}))


def test_load_tasks_joins_manifest_to_aoi(tmp_path: Path) -> None:
    aoi_path, manifest_path = tmp_path / "aoi.geojson", tmp_path / "manifest.csv"
    _write_aoi(aoi_path)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["aoi_id", "role", "item_id", "stac_url", "selection_status"])
        writer.writeheader()
        writer.writerow({"aoi_id": "site", "role": "before", "item_id": "S2_item", "stac_url": "https://example.test/item.json", "selection_status": "selected"})
    tasks = MODULE.load_tasks(manifest_path, MODULE.load_aois(aoi_path))
    assert [(task.aoi_id, task.role, task.item_id) for task in tasks] == [("site", "before", "S2_item")]


def test_load_tasks_rejects_path_traversal(tmp_path: Path) -> None:
    aoi_path, manifest_path = tmp_path / "aoi.geojson", tmp_path / "manifest.csv"
    _write_aoi(aoi_path)
    manifest_path.write_text("aoi_id,role,item_id,stac_url,selection_status\nsite,../before,S2_item,https://example.test/item.json,selected\n")
    with pytest.raises(ValueError, match="unsafe role"):
        MODULE.load_tasks(manifest_path, MODULE.load_aois(aoi_path))


def test_resolve_assets_requires_matching_item(monkeypatch: pytest.MonkeyPatch) -> None:
    task = MODULE.Task("site", "before", "expected", "https://example.test/item", {"type": "Polygon", "coordinates": []})
    monkeypatch.setattr(MODULE, "fetch_json", lambda _: {"id": "other", "assets": {}})
    with pytest.raises(ValueError, match="STAC ID mismatch"):
        MODULE.resolve_assets(task, ("B02",))
