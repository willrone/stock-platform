"""纸面盘 **ledger nav 列** 门禁测试（§12.43）。

## 背景

2026-11-02 17:35 的周频跑发生在**进程互斥锁落地之前**（`00ada47d` 18:51 才提交；
台账 §12.34 记的就是"三档排期并发写账本"）→ `data/paper/<策略>_ledger.csv`
里 11-02 的 6 行 `portfolio_value` 被写成**单笔盈亏**（-54.89 / -56.45）。

后端 `signal_service` 的净值曲线直接读这一列 → 22f 末日点变成 -56.45
（图表上从 990,864 掉到 -56，像清零）；且 5 个月后攒够 20 个调仓点时，
这条脏点会把最大回撤算成 -100%。

同时 `holdings.nav_history` 只剩 1 个点 —— 某次跑 holdings 文件缺失，
`load_holdings()` 返回空壳 → 只追加当前点 → 历史被抹掉
（§12.36 的缺失守卫 `d5a37982` 就是为这个加的，**已经发生过了**）。

## 本文件守什么

1. `extract_nav_from_trades` 的**防御逻辑**（合成数据）：非正/空/非数字一律跳过
2. **产物门禁**（真实 data/paper）：
   - ledger 每行 `portfolio_value` 非空且 > 0
   - `nav_history` 点数 == ledger 成交日数（防止历史再被抹）
   - `nav_history` 末日 vs `portfolio.json` value 残差 <= 0.05
   - **生产函数**提取出的净值末日 == `portfolio.json` value
     （直接调 `signal_service.extract_nav_from_trades`，不复刻逻辑，避免测试漂移）
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.services.trading.signal_service import extract_nav_from_trades

BACKEND_ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = BACKEND_ROOT.parent / "data" / "paper"

VALUE_TOL = 0.05  # 与 portfolio.json 的残差阈值（元）


def _load_portfolio() -> Dict[str, Any]:
    p = PAPER_DIR / "portfolio.json"
    if not p.exists():
        pytest.skip("portfolio.json 不存在（纸面盘还没跑过）")
    with open(p) as fh:
        return json.load(fh)


def _ledger_rows(strategy: str) -> List[Dict[str, str]]:
    p = PAPER_DIR / f"{strategy}_ledger.csv"
    if not p.exists():
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def _holdings(strategy: str) -> Dict[str, Any]:
    p = PAPER_DIR / f"{strategy}_holdings.json"
    if not p.exists():
        return {}
    with open(p) as fh:
        return json.load(fh)


# ══════════════════════════════════════════════════════════════
# 1. extract_nav_from_trades 的防御逻辑（合成数据）
# ══════════════════════════════════════════════════════════════

class TestExtractNavDefensive:
    def test_skips_non_positive_values(self):
        """-56.45 这种脏值必须被跳过，不能进净值曲线。"""
        trades = [
            {"date": "2026-10-26", "portfolio_value": "990864.02"},
            {"date": "2026-11-02", "portfolio_value": "-56.45"},
            {"date": "2026-11-03", "portfolio_value": "978982.94"},
        ]
        nav, dates = extract_nav_from_trades(trades)
        assert nav == [990864.02, 978982.94]
        assert dates == ["2026-10-26", "2026-11-03"]

    def test_skips_zero_blank_and_garbage(self):
        trades = [
            {"date": "2026-11-02", "portfolio_value": "0"},
            {"date": "2026-11-02", "portfolio_value": ""},
            {"date": "2026-11-02", "portfolio_value": None},
            {"date": "2026-11-02", "portfolio_value": "abc"},
            {"date": "2026-11-02", "portfolio_value": "978982.94"},
        ]
        nav, _ = extract_nav_from_trades(trades)
        assert nav == [978982.94]

    def test_same_day_takes_last(self):
        trades = [
            {"date": "2026-10-23", "portfolio_value": "980115.02"},
            {"date": "2026-10-23", "portfolio_value": "991972.81"},
        ]
        nav, dates = extract_nav_from_trades(trades)
        assert nav == [991972.81]
        assert dates == ["2026-10-23"]

    def test_empty_input(self):
        assert extract_nav_from_trades([]) == ([], [])
        assert extract_nav_from_trades(None) == ([], [])


# ══════════════════════════════════════════════════════════════
# 2. 产物门禁（真实 data/paper）
# ══════════════════════════════════════════════════════════════

def _lines() -> List[str]:
    pf = _load_portfolio()
    return sorted((pf.get("lines") or {}).keys())


class TestLedgerNavColumns:
    def test_ledger_portfolio_value_all_positive(self):
        """ledger 每一行的 portfolio_value 都必须存在且 > 0。"""
        bad = []
        for name in _lines():
            for i, r in enumerate(_ledger_rows(name)):
                pv = (r.get("portfolio_value") or "").strip()
                if not pv:
                    bad.append(f"{name} 第{i + 1}行 缺 portfolio_value")
                    continue
                try:
                    v = float(pv)
                except ValueError:
                    bad.append(f"{name} 第{i + 1}行 portfolio_value 非数字: {pv!r}")
                    continue
                if v <= 0:
                    bad.append(f"{name} 第{i + 1}行 portfolio_value 非正: {v}")
        assert not bad, "ledger 的 portfolio_value 有脏值：\n  " + "\n  ".join(bad[:20])

    def test_nav_history_covers_all_trade_days(self):
        """nav_history 必须覆盖 ledger 的每个成交日（防止历史再被抹掉）。"""
        bad = []
        for name in _lines():
            rows = _ledger_rows(name)
            if not rows:
                continue
            ledger_days = {str(r.get("date") or "").strip() for r in rows if r.get("date")}
            nav = _holdings(name).get("nav_history") or []
            nav_days = {str(p.get("date") or "").strip() for p in nav}
            if ledger_days - nav_days:
                bad.append(f"{name} nav_history 缺成交日 {sorted(ledger_days - nav_days)}")
        assert not bad, "\n  ".join(bad)

    def test_nav_history_last_matches_portfolio_json(self):
        """nav_history 末日值 == portfolio.json 的 value（残差 <= 0.05）。"""
        pf = _load_portfolio()
        bad = []
        for name, ln in (pf.get("lines") or {}).items():
            nav = _holdings(name).get("nav_history") or []
            if not nav:
                continue
            last = float(nav[-1].get("value") or 0)
            pfv = float(ln.get("value") or 0)
            if abs(last - pfv) > VALUE_TOL:
                bad.append(f"{name} nav_history 末日 {last} vs portfolio {pfv}（残差 {last - pfv}）")
        assert not bad, "\n  ".join(bad)

    def test_backend_nav_last_matches_portfolio_json(self):
        """**生产函数**提取出的净值末日 == portfolio.json 的 value。

        这条是本文件的核心：后端净值曲线读的就是这一列。
        """
        pf = _load_portfolio()
        bad = []
        for name, ln in (pf.get("lines") or {}).items():
            rows = _ledger_rows(name)
            if not rows:
                continue
            nav, _ = extract_nav_from_trades(rows)
            if not nav:
                bad.append(f"{name} 提取不到任何净值点")
                continue
            pfv = float(ln.get("value") or 0)
            if abs(nav[-1] - pfv) > VALUE_TOL:
                bad.append(f"{name} 后端 nav 末日 {nav[-1]} vs portfolio {pfv}")
            if any(v <= 0 for v in nav):
                bad.append(f"{name} 后端 nav 里有非正值")
        assert not bad, "\n  ".join(bad)

    def test_backend_nav_has_no_drop_to_zero(self):
        """回归防线：净值曲线不允许出现断崖式归零（>90% 单点跌幅）。"""
        bad = []
        for name in _lines():
            nav, dates = extract_nav_from_trades(_ledger_rows(name))
            for i in range(1, len(nav)):
                if nav[i - 1] > 0 and nav[i] / nav[i - 1] < 0.10:
                    bad.append(f"{name} {dates[i]} 净值 {nav[i]:.2f} 相对 {dates[i-1]} 崩到 10% 以下")
        assert not bad, "\n  ".join(bad)
