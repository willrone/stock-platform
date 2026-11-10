"""日频盯市接入 `/trading/signal`（§12.41 方案 A）—— 2026-11-03。

背景
----
§12.41 记录了一个"建好了但没人用"的东西：**日频盯市净值**（`data/paper/
<strategy>_nav_daily.json`）已经能算出来，但接口不暴露、前端不读、也没有定时任务。

方案 A 的边界（有意做小）：
  - ✅ 只**增**一个 `nav_daily` 字段（数据层接入），不删不改任何现有字段
  - ❌ 不动周频 `nav_basis` 口径（那是另一套口径，混在一起会读错）
  - ❌ 不动前端（口径切换是方案 B，要用户点头）
  - ❌ 不加定时任务（那是方案 C）

三条诚实披露原则（对应下面三个测试）
------------------------------------
1. **没有日频数据就说没有**：不能返回空数组让人以为"没波动"
2. **陈旧价格要透出**：`stale_price` 非空 = 该点用的是非当日收盘价估的，
   吞掉它等于伪造精度
3. **新鲜度要透出**：日频产物目前**没有定时任务**在跑，`updated_at`
   就是数据有多旧的唯一证据，必须给到调用方
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.trading import signal_service as ss


def _write_product(tmp_path: Path, strategy: str, payload: dict) -> Path:
    (tmp_path / f"{strategy}_nav_daily.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


# ── 1. 没有产物 → 明确不可用（不许用空数组冒充"没有波动"）──────────────────


def test_missing_product_reports_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(ss, "PAPER_DIR", str(tmp_path))
    out = ss.load_nav_daily("22f")
    assert out["available"] is False
    assert out["points"] == []
    assert out["n_points"] == 0
    assert out["note"], "不可用时必须给出原因说明"


def test_product_without_usable_points_reports_unavailable(tmp_path, monkeypatch):
    _write_product(tmp_path, "22f", {"points": []})
    monkeypatch.setattr(ss, "PAPER_DIR", str(tmp_path))
    out = ss.load_nav_daily("22f")
    assert out["available"] is False
    assert out["points"] == []


def test_unparseable_value_points_are_dropped_not_fatal(tmp_path, monkeypatch):
    """坏点跳过，好点保留 —— 不能因为一个 None/字符串就整份产物报废。"""
    _write_product(
        tmp_path,
        "22f",
        {
            "points": [
                {"date": "2026-10-16", "value": None},
                {"date": "2026-10-17", "value": "abc"},
                {"date": "2026-10-18", "value": 1000123.456},
            ]
        },
    )
    monkeypatch.setattr(ss, "PAPER_DIR", str(tmp_path))
    out = ss.load_nav_daily("22f")
    assert out["available"] is True
    assert out["n_points"] == 1
    assert out["points"][0]["date"] == "2026-10-18"
    assert out["points"][0]["value"] == 1000123.46  # 保留两位


# ── 2. 陈旧价格标记必须透出（不许吞）────────────────────────────────────────


def test_stale_price_flag_is_preserved_and_counted(tmp_path, monkeypatch):
    _write_product(
        tmp_path,
        "22f",
        {
            "basis": "total_return@trade_day (paper_portfolio.total_return_portfolio_value)",
            "initial_capital": 1000000.0,
            "updated_at": "2026-11-02T20:39:00",
            "points": [
                {
                    "date": "2026-10-16",
                    "value": 998795.88,
                    "return_pct": -0.1204,
                    "n_positions": 30,
                    "stale_price": None,
                },
                {
                    "date": "2026-10-17",
                    "value": 999000.0,
                    "return_pct": -0.1,
                    "n_positions": 30,
                    "stale_price": "2026-10-16",
                },
            ],
        },
    )
    monkeypatch.setattr(ss, "PAPER_DIR", str(tmp_path))
    out = ss.load_nav_daily("22f")
    assert out["available"] is True
    assert out["n_points"] == 2
    assert out["n_stale_points"] == 1, "陈旧价点必须计数，不能悄悄混进净值曲线"
    assert out["points"][0]["stale_price"] is None
    assert out["points"][1]["stale_price"] == "2026-10-16"
    assert out["basis"]
    assert out["updated_at"] == "2026-11-02T20:39:00"
    assert out["initial_capital"] == 1000000.0


# ── 3. 接口层：只增不改 ────────────────────────────────────────────────────


def test_daily_signal_exposes_nav_daily_without_breaking_existing_fields():
    """新字段必须挂上，且 daily_signal 原有字段一个都不能少。"""
    svc = ss.TradingSignalService()
    out = svc.daily_signal("22f")
    if "尚无信号数据" in str(out.get("message", "")):
        pytest.skip("没有 22f 信号产物，daily_signal 走的是提示分支（字段集本来就不完整）——字段契约必须在本机跑")
    assert "nav_daily" in out
    for key in ("buys", "sells", "holds", "cost_model", "fill_assumption", "as_of"):
        assert key in out, f"原有字段 {key} 不能因为这次改动消失"
    nav_daily = out["nav_daily"]
    assert isinstance(nav_daily, dict)
    assert "available" in nav_daily
    assert "updated_at" in nav_daily
    assert "points" in nav_daily


def test_performance_exposes_nav_daily_and_keeps_weekly_nav_fields():
    """周频 nav_* 在 performance() 里 —— 日频是新增，不许覆盖/改名。"""
    svc = ss.TradingSignalService()
    out = svc.performance("22f")
    for key in ("nav_basis", "nav_points", "nav_points_required", "nav_insufficient"):
        assert key in out, f"周频字段 {key} 不能因为这次改动消失"
    assert "nav_daily" in out
    assert isinstance(out["nav_daily"]["points"], list)


# ── 4. 真实产物（有就跑，没有就 skip，不许编）───────────────────────────────


def test_real_product_is_readable_and_dates_are_ordered():
    out = ss.load_nav_daily("22f")
    if not out["available"]:
        pytest.skip("本机没有日频盯市产物（未跑 paper_daily_nav.py）")
    assert out["n_points"] > 0
    assert out["basis"], "必须带口径说明"
    assert out["updated_at"], "必须带数据新鲜度（cron 产出，产物可能滞后）"
    dates = [p["date"] for p in out["points"]]
    assert dates == sorted(dates), "净值点必须按日期有序"
    assert all(p["value"] > 0 for p in out["points"]), "净值不能为负或零"
