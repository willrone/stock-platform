"""`check_paper_nav_coverage.py` 的门禁测试 —— 证明这道门禁**能变红**。

（只会变绿的门禁等于没有门禁。这里用临时目录造四种破坏场景。）
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

BACKEND = Path(__file__).resolve().parents[3]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

SCRIPT = BACKEND / "scripts" / "check_paper_nav_coverage.py"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gate():
    return _load(SCRIPT, "check_paper_nav_coverage_test")


def _write(dirpath: Path, name: str, payload: Dict[str, Any]) -> None:
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, name), "w") as fh:
        json.dump(payload, fh)


BASE_LINES: Dict[str, Any] = {
    "22f": {"holdings": {"last_trade": "2026-11-02", "cash": 1.0,
                         "positions": {}},
            "nav": {"points": [{"date": "2026-11-03", "value": 990000.0}]}},
    "multi_asset_final": {
        "holdings": {"last_trade": "2026-11-02", "cash": 1.0,
                     "positions": {}, "legs": {"a": {}, "sleeve": {}}},
        "nav": {"points": [{"date": "2026-11-03", "value": 1000000.0,
                            "legs": {"a": 500000.0, "sleeve": 500000.0}}]}},
    "multi_asset_final_a": {"holdings": {"last_trade": "2026-11-02"},
                            "nav": {"points": [{"date": "2026-11-03",
                                                "value": 500000.0}]}},
    "multi_asset_final_sleeve": {"holdings": {"last_trade": "2026-11-02"},
                                 "nav": {"points": [{"date": "2026-11-03",
                                                     "value": 500000.0}]}},
}


@pytest.fixture
def wired(gate, tmp_path, monkeypatch, capsys):
    def build(mutate=None):
        d = tmp_path / "paper"
        if d.exists():
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
        data = {k: {"holdings": dict(v["holdings"]),
                    "nav": json.loads(json.dumps(v["nav"]))}
                for k, v in BASE_LINES.items()}
        if mutate:
            mutate(data)
        for line, blob in data.items():
            _write(d, f"{line}_holdings.json", blob["holdings"])
            if blob["nav"] is not None:
                _write(d, f"{line}_nav_daily.json", blob["nav"])
        monkeypatch.setattr(gate, "PAPER_DIR", str(d))
        monkeypatch.setattr(gate.pp, "acquire_run_lock",
                            lambda timeout_s=None: 99)
        capsys.readouterr()
        return gate.main()

    return build


# ── 绿：一切齐 ────────────────────────────────────────────────────────
def test_green_when_everything_present(wired):
    assert wired() == 0


# ── 红 1：有成交却没有日频产物（当初的覆盖缺口）──────────────────────
def test_red_when_line_has_trades_but_no_nav(wired):
    def mutate(data):
        data["22f"]["nav"] = None
    assert wired(mutate) == 1


# ── 红 2：母线有、子腿缺一条 ──────────────────────────────────────────
def test_red_when_a_subleg_is_missing(wired):
    def mutate(data):
        data["multi_asset_final_sleeve"]["nav"] = None
    assert wired(mutate) == 1


# ── 红 3：母线末点 ≠ 两腿之和（合并口径漂移）─────────────────────────
def test_red_when_parent_value_is_not_sum_of_legs(wired):
    def mutate(data):
        data["multi_asset_final"]["nav"]["points"][0]["value"] = 900000.0
    assert wired(mutate) == 1


# ── 红 4：两条腿末点日期不同（不同日不能相加）────────────────────────
def test_red_when_leg_dates_mismatch(wired):
    def mutate(data):
        data["multi_asset_final_sleeve"]["nav"]["points"][0]["date"] = "2026-11-02"
    assert wired(mutate) == 1


# ── 红 5：成交日之后曲线停住（末点早于 last_trade）───────────────────
def test_red_when_nav_last_date_is_older_than_last_trade(wired):
    def mutate(data):
        data["22f"]["nav"]["points"][0]["date"] = "2026-10-30"
    assert wired(mutate) == 1


# ── 子腿不算母线（否则母线数会虚增、判据会重复计）────────────────────
def test_sublegs_are_not_counted_as_main_lines(gate, wired, monkeypatch):
    wired()  # 建立 tmp 目录并接线
    mains, sublegs = gate.list_lines()
    assert "multi_asset_final" in mains
    assert "multi_asset_final_a" not in mains
    assert set(sublegs) == {"multi_asset_final_a", "multi_asset_final_sleeve"}
