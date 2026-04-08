"""Tests for strict-baseline golden artifact manifest."""

from __future__ import annotations

import json
from pathlib import Path

MANIFEST_PATH = (
    Path(__file__).resolve().parents[2] / "golden" / "strict_baseline" / "manifest.json"
)


def test_manifest_contains_15_strategies() -> None:
    """Manifest should pin exactly 15 baseline strategies."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["strategy_count"] == 15
    assert len(manifest["task_id_mapping"]) == 15
    assert len(set(manifest["strategy_names"])) == 15


def test_manifest_strategy_files_exist() -> None:
    """Every manifest row should point to a readable strategy file."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    strategy_names: set[str] = set()

    for entry in manifest["task_id_mapping"]:
        strategy_name = entry["strategy_name"]
        strategy_names.add(strategy_name)
        strategy_path = MANIFEST_PATH.parent / entry["file"]
        strategy_document = json.loads(strategy_path.read_text(encoding="utf-8"))
        assert strategy_document["strategy_name"] == strategy_name
        assert strategy_document["source_task"]["task_id"] == entry["task_id"]

    assert strategy_names == set(manifest["strategy_names"])
