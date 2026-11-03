"""`ledger_replay.py` 的门禁测试（方案 D：历史区间回溯）。

用**合成数据**（tmp_path 造 ledger + 手写后复权矩阵），**不依赖真实
`data/paper/` 数据** → 跑得快、可重复、能构造真实数据里罕见的边界情况。

## 本文件守什么

| # | 门禁 | 为什么 |
|---|---|---|
| a | **13 列 / 15 列两种列契约**的行都能解析 | §12.48 迁移前后的账本格式不同；且曾出现「13 列表头 + 15 列数据行」的错位（§12.53） |
| b | **`cash_after` 不可信**时，重算现金仍正确 | `cash_after` 混着三种语义（§12.53）→ 重建现金必须自己重算 |
| c | **全清仓语义**（与生产 `run_line` 一致） | 卖出 = 卖掉整个持仓并从 positions 删除；买入 = 整体覆盖 |
| d | **缺行情日被显式标注** | 严禁静默用前值填充（方案书 §2.6 风险 2） |
| e | **不读 `cash_after`** | 把 `cash_after` 改成毒值，重算结果必须不变 |

## 为什么不用真实数据

真实账本只有一种形态（当前 15 列 + 固定语义）→ 测不出「13 列」「错位」
「部分卖出」「缺价」这些**正是要防的**情况。
且真实回放要加载 5549 只 × 8756 天的后复权矩阵（2~4 分钟）→ 不适合做单元测试。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

BACKEND = Path(__file__).resolve().parents[3]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

SCRIPT = BACKEND / "scripts" / "ledger_replay.py"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # ⚠️ 必须先注册进 sys.modules 再 exec_module：
    # `LedgerRow` 是 @dataclass + `from __future__ import annotations`
    # → 注解是字符串 → dataclasses 装饰器要回 sys.modules 找模块解析注解。
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lr():
    return _load(SCRIPT, "ledger_replay_gate_test")


# ── 合成数据 ──────────────────────────────────────────────────────────

DATES = pd.to_datetime(["2026-10-16", "2026-10-19", "2026-10-20",
                        "2026-10-21", "2026-10-22"])


def make_adj() -> pd.DataFrame:
    """后复权价矩阵（行 = 交易日）。"""
    return pd.DataFrame({
        "600000.SH": [10.0, 10.5, 11.0, 12.0, 13.0],
        "600001.SH": [20.0, 20.0, 21.0, 21.0, 21.0],
        "600002.SH": [8.0, 8.0, 8.0, 8.0, 8.0],
    }, index=DATES)


def make_last_valid(adj: pd.DataFrame):
    """与 `paper_portfolio.last_valid_adj` 同口径的替身。

    ≤ as_of 的最后一个**有效**（非 NaN）后复权价 → (值, 日期字符串)。
    """
    def _fn(_adj: pd.DataFrame, code: str, as_of: Any):
        if _adj is None or code not in _adj.columns:
            return None, None
        s = _adj.loc[:pd.Timestamp(as_of), code].dropna()
        if not len(s):
            return None, None
        v = float(s.iloc[-1])
        if not np.isfinite(v) or v <= 0:
            return None, None
        return v, str(pd.Timestamp(s.index[-1]).date())
    return _fn


def write_ledger(path: Path, header: List[str], rows: List[List[Any]]) -> Path:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return path


# 15 列契约
H15 = ["date", "action", "code", "name", "price", "shares", "amount",
       "commission", "stamp_tax", "slippage", "score",
       "cost_amount", "realized_pnl", "cash_after", "portfolio_value"]
# 13 列旧契约（无 cost_amount / realized_pnl）
H13 = ["date", "action", "code", "name", "price", "shares", "amount",
       "commission", "stamp_tax", "slippage", "score",
       "cash_after", "portfolio_value"]


def row15(date, action, code, price, shares, amount,
          comm="0.0", stamp="0.0", slip="0.0", score="",
          cost_amt="", pnl="", cash_after="", pv=""):
    return [date, action, code, "名称", price, shares, amount,
            comm, stamp, slip, score, cost_amt, pnl, cash_after, pv]


def row13(date, action, code, price, shares, amount,
          comm="0.0", stamp="0.0", slip="0.0", score="",
          cash_after="", pv=""):
    return [date, action, code, "名称", price, shares, amount,
            comm, stamp, slip, score, cash_after, pv]


# ══════════════════════════════════════════════════════════════════
# a：两种列契约都能解析
# ══════════════════════════════════════════════════════════════════

class TestColumnContracts:

    def test_parse_15_col_contract(self, lr, tmp_path):
        p = write_ledger(tmp_path / "l15.csv", H15, [
            row15("2026-10-16", "BUY", "600000.SH", "10.0", "1000", "10000.0",
                  comm="5.0", stamp="0.0", slip="10.0", cash_after="989985.0",
                  pv="1000000.0"),
            row15("2026-10-20", "SELL", "600000.SH", "11.0", "1000", "11000.0",
                  comm="5.0", stamp="5.5", slip="11.0", cash_after="1000963.5",
                  pv="1000963.5"),
        ])
        rows, diag = lr.parse_ledger(str(p))
        assert len(rows) == 2
        assert diag["header_contract"] == "15"
        assert diag["row_contracts"] == {"15": 2}
        assert diag["contract_mismatch_rows"] == []
        assert rows[0].contract == "15"
        assert rows[0].amount == pytest.approx(10000.0)
        # 15 列下 cash_after 在**第 14 位**（索引 13）
        assert rows[0].cash_after == pytest.approx(989985.0)
        assert rows[0].portfolio_value == pytest.approx(1000000.0)
        assert rows[0].cost_amount is None      # 本行没填
        assert rows[0].realized_pnl is None

    def test_parse_13_col_contract(self, lr, tmp_path):
        p = write_ledger(tmp_path / "l13.csv", H13, [
            row13("2026-10-16", "BUY", "600000.SH", "10.0", "1000", "10000.0",
                  comm="5.0", stamp="0.0", slip="10.0", cash_after="989985.0",
                  pv="1000000.0"),
        ])
        rows, diag = lr.parse_ledger(str(p))
        assert len(rows) == 1
        assert diag["header_contract"] == "13"
        assert diag["row_contracts"] == {"13": 1}
        # 13 列下 cash_after 在**第 12 位**（索引 11）
        assert rows[0].cash_after == pytest.approx(989985.0)
        assert rows[0].portfolio_value == pytest.approx(1000000.0)

    def test_misaligned_header_15_rows_detected(self, lr, tmp_path):
        """**13 列表头 + 15 列数据行**（§12.53 的真实错位形态）。

        `csv.DictReader` 会把 `cash_after` 读成 `cost_amount`（伪值）。
        本模块**按行长度**判定契约 → 仍能读出真值，并**报告不一致**。
        """
        p = write_ledger(tmp_path / "misaligned.csv", H13, [
            # 15 个字段（真实写入的是 15 列），但表头只有 13 列
            row15("2026-11-02", "SELL", "600000.SH", "20.55", "1500", "30495.0",
                  comm="9.15", stamp="15.25", slip="30.49", score="-0.2521",
                  cost_amt="30495.0", pnl="-54.89", cash_after="78942.93",
                  pv="1001156.93"),
        ])
        rows, diag = lr.parse_ledger(str(p))
        assert len(rows) == 1
        # 数据行按 15 列解析 → cash_after 读出**真值** 78942.93
        assert rows[0].cash_after == pytest.approx(78942.93)
        assert rows[0].cost_amount == pytest.approx(30495.0)
        assert rows[0].realized_pnl == pytest.approx(-54.89)
        # 且**明确报告**表头与数据行契约不一致
        assert diag["header_contract"] == "13"
        assert diag["row_contracts"] == {"15": 1}
        assert diag["contract_mismatch_rows"] == [2]

    def test_unparseable_row_length_is_reported_not_guessed(self, lr, tmp_path):
        """长度既非 13 也非 15 的行 → **报告**，不猜、不静默丢弃。"""
        p = write_ledger(tmp_path / "weird.csv", H15, [
            row15("2026-10-16", "BUY", "600000.SH", "10.0", "1000", "10000.0"),
            ["2026-10-19", "BUY", "600001.SH", "x", "y", "z"],   # 6 列
        ])
        rows, diag = lr.parse_ledger(str(p))
        assert len(rows) == 1
        assert diag["unparsed_rows"] == [3]

    def test_missing_file_returns_empty_with_error(self, lr, tmp_path):
        rows, diag = lr.parse_ledger(str(tmp_path / "nope.csv"))
        assert rows == []
        assert "error" in diag


# ══════════════════════════════════════════════════════════════════
# b + e：`cash_after` 不可信 → 重算现金
# ══════════════════════════════════════════════════════════════════

class TestCashRecomputation:

    def _ledger(self, tmp_path, cash_after_values):
        """一笔买入 + 一笔卖出，`cash_after` 取给定值（用来投毒）。"""
        p = write_ledger(tmp_path / "cash.csv", H15, [
            row15("2026-10-16", "BUY", "600000.SH", "10.0", "1000", "10000.0",
                  comm="5.0", stamp="0.0", slip="10.0",
                  cash_after=cash_after_values[0], pv=""),
            row15("2026-10-20", "SELL", "600000.SH", "11.0", "1000", "11000.0",
                  comm="5.0", stamp="5.5", slip="11.0",
                  cash_after=cash_after_values[1], pv=""),
        ])
        return p

    def test_cash_is_recomputed_not_read_from_cash_after(self, lr, tmp_path):
        """`cash_after` 写成毒值 → 重算出的现金**不受影响**。"""
        adj = make_adj()
        lv = make_last_valid(adj)

        p1 = self._ledger(tmp_path, ["999999.0", "888888.0"])
        rows1, _ = lr.parse_ledger(str(p1))
        _s1, d1 = lr.replay_rows(rows1, 1_000_000.0, adj_close=adj,
                                 last_valid_fn=lv)

        p2 = self._ledger(tmp_path, ["123.0", "456.0"])
        rows2, _ = lr.parse_ledger(str(p2))
        _s2, d2 = lr.replay_rows(rows2, 1_000_000.0, adj_close=adj,
                                 last_valid_fn=lv)

        # 两份账本只有 cash_after 不同 → **重算现金必须完全相同**
        assert d1["final_cash"] == pytest.approx(d2["final_cash"])
        # 且等于生产口径的手算值：
        #   BUY : 10000 + max(5, 3) + 10 = 10015  → 现金 989985
        #   SELL: 11000 - max(5, 3.3) - 5.5 - 11 = 10978.5 → 现金 1000963.5
        assert d1["final_cash"] == pytest.approx(1_000_963.5)

    def test_cash_ignores_cash_after_entirely(self, lr, tmp_path):
        """把 cash_after 设成完全离谱的值，结果仍正确。"""
        adj = make_adj()
        lv = make_last_valid(adj)
        p = self._ledger(tmp_path, ["-1.0", "-2.0"])
        rows, _ = lr.parse_ledger(str(p))
        snaps, diag = lr.replay_rows(rows, 1_000_000.0, adj_close=adj,
                                     last_valid_fn=lv)
        assert diag["final_cash"] == pytest.approx(1_000_963.5)
        assert snaps[0]["cash"] == pytest.approx(989_985.0)

    def test_ledger_columns_model_uses_the_three_columns(self, lr, tmp_path):
        """`ledger_columns` 模型：直接用账本三列（对照口径，可审计）。"""
        adj = make_adj()
        lv = make_last_valid(adj)
        p = self._ledger(tmp_path, ["0", "0"])
        rows, _ = lr.parse_ledger(str(p))
        _s, diag = lr.replay_rows(rows, 1_000_000.0, adj_close=adj,
                                  last_valid_fn=lv, cash_model="ledger_columns")
        # BUY : 10000 + 5 + 10 = 10015 → 989985
        # SELL: 11000 - 5 - 5.5 - 11 = 10978.5 → 1000963.5
        assert diag["final_cash"] == pytest.approx(1_000_963.5)

    def test_unknown_cash_model_raises(self, lr, tmp_path):
        adj = make_adj()
        p = self._ledger(tmp_path, ["0", "0"])
        rows, _ = lr.parse_ledger(str(p))
        with pytest.raises(ValueError, match="cash_model"):
            lr.replay_rows(rows, 1_000_000.0, adj_close=adj,
                           last_valid_fn=make_last_valid(adj),
                           cash_model="bogus")


# ══════════════════════════════════════════════════════════════════
# c：全清仓语义（与生产 run_line 一致）
# ══════════════════════════════════════════════════════════════════

class TestFullLiquidationSemantics:

    def test_sell_removes_position_entirely(self, lr, tmp_path):
        adj = make_adj()
        lv = make_last_valid(adj)
        p = write_ledger(tmp_path / "liq.csv", H15, [
            row15("2026-10-16", "BUY", "600000.SH", "10.0", "1000", "10000.0"),
            row15("2026-10-20", "SELL", "600000.SH", "11.0", "1000", "11000.0"),
        ])
        rows, _ = lr.parse_ledger(str(p))
        snaps, _d = lr.replay_rows(rows, 1_000_000.0, adj_close=adj,
                                   last_valid_fn=lv, semantics="full_liquidation")
        assert "600000.SH" in snaps[0]["positions"]
        assert "600000.SH" not in snaps[1]["positions"]

    def test_rebuy_after_sell_overwrites_position(self, lr, tmp_path):
        """清仓后再买入同一只 → **整体覆盖**（不是累加股数）。"""
        adj = make_adj()
        lv = make_last_valid(adj)
        p = write_ledger(tmp_path / "rebuy.csv", H15, [
            row15("2026-10-16", "BUY", "600000.SH", "10.0", "1000", "10000.0"),
            row15("2026-10-20", "SELL", "600000.SH", "11.0", "1000", "11000.0"),
            row15("2026-10-21", "BUY", "600000.SH", "12.0", "500", "6000.0"),
        ])
        rows, _ = lr.parse_ledger(str(p))
        snaps, diag = lr.replay_rows(rows, 1_000_000.0, adj_close=adj,
                                     last_valid_fn=lv, semantics="full_liquidation")
        pos = snaps[2]["positions"]["600000.SH"]
        assert pos["shares"] == pytest.approx(500.0)     # 不是 1500
        assert pos["cost"] == pytest.approx(12.0)
        assert pos["buy_date"] == "2026-10-21"
        # 卖出已把该票从 positions 删除 → 这次买入**不是** re_buy（可区分）
        assert diag["re_buys"] == []

    def test_double_buy_without_sell_is_recorded_as_re_buy(self, lr, tmp_path):
        """**没卖过**就再买同一只 → 判定为 re_buy（审计用），
        且全清仓语义下**整体覆盖**（shares=500，不是 1500）。"""
        adj = make_adj()
        lv = make_last_valid(adj)
        p = write_ledger(tmp_path / "dbl.csv", H15, [
            row15("2026-10-16", "BUY", "600000.SH", "10.0", "1000", "10000.0"),
            row15("2026-10-21", "BUY", "600000.SH", "12.0", "500", "6000.0"),
        ])
        rows, _ = lr.parse_ledger(str(p))
        snaps, diag = lr.replay_rows(rows, 1_000_000.0, adj_close=adj,
                                     last_valid_fn=lv, semantics="full_liquidation")
        pos = snaps[1]["positions"]["600000.SH"]
        assert pos["shares"] == pytest.approx(500.0)
        assert pos["cost"] == pytest.approx(12.0)
        assert diag["re_buys"]
        assert diag["re_buys"][0]["code"] == "600000.SH"

    def test_moving_average_semantics_differs_on_double_buy(self, lr, tmp_path):
        """对照：`moving_average` 会**累加**股数并取加权平均成本。

        → 两种语义在「重复买入」上**确实不同**；
        现有真实账本里没有重复买入 → 这个选择无法被现有数据证伪
        → 选与生产 `run_line` 一致的**全清仓**。
        """
        adj = make_adj()
        lv = make_last_valid(adj)
        p = write_ledger(tmp_path / "dbl2.csv", H15, [
            row15("2026-10-16", "BUY", "600000.SH", "10.0", "1000", "10000.0"),
            row15("2026-10-21", "BUY", "600000.SH", "12.0", "500", "6000.0"),
        ])
        rows, _ = lr.parse_ledger(str(p))
        snaps_ma, _d = lr.replay_rows(rows, 1_000_000.0, adj_close=adj,
                                      last_valid_fn=lv, semantics="moving_average")
        pos_ma = snaps_ma[1]["positions"]["600000.SH"]
        assert pos_ma["shares"] == pytest.approx(1500.0)
        assert pos_ma["cost"] == pytest.approx(round((10.0 * 1000 + 12.0 * 500)
                                                     / 1500, 3))

        snaps_fl, _d2 = lr.replay_rows(rows, 1_000_000.0, adj_close=adj,
                                       last_valid_fn=lv,
                                       semantics="full_liquidation")
        pos_fl = snaps_fl[1]["positions"]["600000.SH"]
        assert pos_fl["shares"] == pytest.approx(500.0)
        assert pos_fl["cost"] == pytest.approx(12.0)
        assert pos_ma["shares"] != pos_fl["shares"]

    def test_partial_sell_is_flagged(self, lr, tmp_path):
        """部分卖出 → 记录进 `partial_sells`（现有真实账本里 0 次）。"""
        adj = make_adj()
        lv = make_last_valid(adj)
        p = write_ledger(tmp_path / "part.csv", H15, [
            row15("2026-10-16", "BUY", "600000.SH", "10.0", "1000", "10000.0"),
            row15("2026-10-20", "SELL", "600000.SH", "11.0", "400", "4400.0"),
        ])
        rows, _ = lr.parse_ledger(str(p))
        _snaps, diag = lr.replay_rows(rows, 1_000_000.0, adj_close=adj,
                                      last_valid_fn=lv)
        assert diag["partial_sells"]
        assert diag["partial_sells"][0]["code"] == "600000.SH"
        assert diag["partial_sells"][0]["reason"] == "部分卖出"

    def test_state_at_returns_state_on_or_before_as_of(self, lr, tmp_path):
        adj = make_adj()
        lv = make_last_valid(adj)
        p = write_ledger(tmp_path / "state.csv", H15, [
            row15("2026-10-16", "BUY", "600000.SH", "10.0", "1000", "10000.0"),
            row15("2026-10-20", "SELL", "600000.SH", "11.0", "1000", "11000.0"),
            row15("2026-10-21", "BUY", "600001.SH", "20.0", "500", "10000.0"),
        ])
        rows, _ = lr.parse_ledger(str(p))
        snaps, _d = lr.replay_rows(rows, 1_000_000.0, adj_close=adj,
                                   last_valid_fn=lv)
        # as_of = 卖出日之后、下一笔买入之前 → 空仓
        pos, cash, info = lr.state_at(snaps, "2026-10-20")
        assert pos == {}
        assert info["matched"] == "2026-10-20"
        # as_of = 最后一天 → 持有 600001.SH
        pos2, cash2, info2 = lr.state_at(snaps, "2026-10-22")
        assert set(pos2) == {"600001.SH"}
        assert info2["matched"] == "2026-10-21"

    def test_state_at_before_first_trade_returns_initial_capital(self, lr, tmp_path):
        """as_of 早于第一笔成交 → 空仓 + **初始现金**（不是 0）。"""
        adj = make_adj()
        lv = make_last_valid(adj)
        p = write_ledger(tmp_path / "early.csv", H15, [
            row15("2026-10-16", "BUY", "600000.SH", "10.0", "1000", "10000.0"),
        ])
        rows, _ = lr.parse_ledger(str(p))
        snaps, _d = lr.replay_rows(rows, 1_000_000.0, adj_close=adj,
                                   last_valid_fn=lv)
        pos, cash, info = lr.state_at(snaps, "2026-10-01")
        assert pos == {}
        assert cash == pytest.approx(1_000_000.0)
        assert info["matched"] is None


# ══════════════════════════════════════════════════════════════════
# d：缺行情日被显式标注（不静默用前值填充）
# ══════════════════════════════════════════════════════════════════

class TestMissingPriceAnnotation:

    def _setup(self, lr, tmp_path, adj):
        p = write_ledger(tmp_path / "miss.csv", H15, [
            row15("2026-10-16", "BUY", "600000.SH", "10.0", "1000", "10000.0"),
            row15("2026-10-16", "BUY", "600002.SH", "8.0", "1000", "8000.0"),
        ])
        rows, _ = lr.parse_ledger(str(p))
        return rows

    def test_stale_price_day_is_marked(self, lr, tmp_path):
        """某日某票没价（停牌）→ 用**更早**的有效价，但**必须标注 stale**。"""
        adj = make_adj()
        adj.loc[DATES[-1], "600000.SH"] = np.nan
        rows = self._setup(lr, tmp_path, adj)
        snaps, _d = lr.replay_rows(rows, 1_000_000.0, adj_close=adj,
                                   last_valid_fn=make_last_valid(adj))
        pts = lr.build_nav_points(
            snaps, adj_close=adj, trading_days=DATES, initial_capital=1_000_000.0,
            value_fn=_fake_value_fn, last_valid_fn=make_last_valid(adj))
        last = pts[-1]
        assert last["date"] == "2026-10-22"
        assert "600000.SH" in last["stale_codes"]
        assert last["stale_count"] >= 1
        # 该点的 stale_price（兼容字段）必须**非空** → 不能吞掉
        assert last["stale_price"]
        # 并记录了实际用的价格日期
        assert last["stale_price_dates"].get("600000.SH") == "2026-10-21"

    def test_missing_price_day_is_marked_as_missing(self, lr, tmp_path):
        """完全无价（该票从未有过后复权价）→ 标注 **missing**，不是 stale。"""
        adj = make_adj()
        adj["600002.SH"] = np.nan          # 整列无有效价
        rows = self._setup(lr, tmp_path, adj)
        snaps, _d = lr.replay_rows(rows, 1_000_000.0, adj_close=adj,
                                   last_valid_fn=make_last_valid(adj))
        pts = lr.build_nav_points(
            snaps, adj_close=adj, trading_days=DATES, initial_capital=1_000_000.0,
            value_fn=_fake_value_fn, last_valid_fn=make_last_valid(adj))
        for p in pts:
            assert "600002.SH" in p["missing_codes"]
            assert p["missing_count"] >= 1
            assert p["stale_price"]

    def test_no_silent_fill_missing_price(self, lr, tmp_path):
        """缺价日**不得**静默用前值填充。

        - **停牌**（有更早的有效价）→ 用更早的有效价估值，但**必须标注 stale**
          （方案书 §2.6 风险 2：用旧价冒充当日价会造出假的 0 收益点）
        - **完全无价**（该票从未有过有效后复权价）→ ratio=1.0，
          市值**冻结在买入额**，且标注 **missing**（不是"假装价格没动"）
        """
        adj = make_adj()
        lv = make_last_valid(adj)

        # ① 停牌：末日 600000.SH 无价，但 10-21 有价 12.0
        adj_stop = adj.copy()
        adj_stop.loc[DATES[-1], "600000.SH"] = np.nan
        rows = self._setup(lr, tmp_path, adj_stop)
        snaps, _d = lr.replay_rows(rows, 1_000_000.0, adj_close=adj_stop,
                                   last_valid_fn=lv)
        pts = lr.build_nav_points(
            snaps, adj_close=adj_stop, trading_days=DATES,
            initial_capital=1_000_000.0,
            value_fn=_fake_value_fn, last_valid_fn=lv)
        last = pts[-1]
        # 停牌日用 10-21 的 12.0 → ratio 1.2 → 市值 12000，**并标注 stale**
        _v, detail = _fake_value_fn(snaps[-1]["positions"], snaps[-1]["cash"],
                                     adj_stop, DATES[-1])
        assert detail["600000.SH"] == pytest.approx(12_000.0)
        assert "600000.SH" in last["stale_codes"]
        assert last["stale_price_dates"]["600000.SH"] == "2026-10-21"
        assert "600000.SH" not in last["missing_codes"]

        # ② 完全无价：600002.SH 整列 NaN → ratio=1.0 → 市值冻结在买入额 8000
        adj_none = adj.copy()
        adj_none["600002.SH"] = np.nan
        rows2 = self._setup(lr, tmp_path, adj_none)
        snaps2, _d2 = lr.replay_rows(rows2, 1_000_000.0, adj_close=adj_none,
                                     last_valid_fn=lv)
        _v2, detail2 = _fake_value_fn(snaps2[-1]["positions"], snaps2[-1]["cash"],
                                       adj_none, DATES[-1])
        assert detail2["600002.SH"] == pytest.approx(8_000.0)   # 冻结，不是涨跌
        pts2 = lr.build_nav_points(
            snaps2, adj_close=adj_none, trading_days=DATES,
            initial_capital=1_000_000.0,
            value_fn=_fake_value_fn, last_valid_fn=lv)
        assert "600002.SH" in pts2[-1]["missing_codes"]

    def test_every_trading_day_produces_a_point(self, lr, tmp_path):
        """每个交易日都要有净值点（首笔成交日起）—— 不能漏天。"""
        adj = make_adj()
        rows = self._setup(lr, tmp_path, adj)
        snaps, _d = lr.replay_rows(rows, 1_000_000.0, adj_close=adj,
                                   last_valid_fn=make_last_valid(adj))
        pts = lr.build_nav_points(
            snaps, adj_close=adj, trading_days=DATES, initial_capital=1_000_000.0,
            value_fn=_fake_value_fn, last_valid_fn=make_last_valid(adj))
        assert [p["date"] for p in pts] == [str(d.date()) for d in DATES]

    def test_range_filters_points(self, lr, tmp_path):
        """`start`/`end` 收窄区间 → 只产出区间内的点。"""
        adj = make_adj()
        rows = self._setup(lr, tmp_path, adj)
        snaps, _d = lr.replay_rows(rows, 1_000_000.0, adj_close=adj,
                                   last_valid_fn=make_last_valid(adj))
        pts = lr.build_nav_points(
            snaps, adj_close=adj, trading_days=DATES, initial_capital=1_000_000.0,
            value_fn=_fake_value_fn, last_valid_fn=make_last_valid(adj),
            start="2026-10-19", end="2026-10-21")
        assert [p["date"] for p in pts] == ["2026-10-19", "2026-10-20", "2026-10-21"]


def _fake_value_fn(positions, cash, adj_close, as_of):
    """极简估值函数（只为测试**标注**逻辑，不参与口径验证）。

    市值 = shares × cost × ratio；ratio = ≤as_of 的最后有效后复权价 / adj_cost；
    完全没有价 → ratio = 1.0（冻结在买入额）。
    """
    total = float(cash)
    detail = {}
    for code, p in (positions or {}).items():
        sh = float(p.get("shares", 0) or 0)
        cost = float(p.get("cost", 0) or 0)
        if sh <= 0 or cost <= 0:
            continue
        a_cost = p.get("adj_cost")
        ratio = 1.0
        if adj_close is not None and code in adj_close.columns:
            s = adj_close.loc[:pd.Timestamp(as_of), code].dropna()
            if len(s):
                v = float(s.iloc[-1])
                try:
                    a_cost_f = float(a_cost)
                except (TypeError, ValueError):
                    a_cost_f = 0.0
                if np.isfinite(a_cost_f) and a_cost_f > 0:
                    ratio = v / a_cost_f
        mv = sh * cost * ratio
        total += mv
        detail[code] = mv
    return total, detail
