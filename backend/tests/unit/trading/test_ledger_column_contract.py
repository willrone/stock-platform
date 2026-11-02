"""纸面盘 **ledger 列契约** 门禁（§12.48）。

## 背景

`append_ledger` 以前用 `fieldnames=list(rows[0].keys())`，而表头只在文件不存在时写一次。
批次首行是 BUY(13 键) / SELL(15 键) 不固定 ⇒ **同一份代码写出的行长度会变**：

| 场景 | 结果 |
|---|---|
| 首行 BUY(13) → 后来追加 SELL(15) | 新行 15 列、表头仍 13 列 → `csv.DictReader` **静默错位**：`cash_after` 读到 `cost_amount`、`portfolio_value` 读到 `realized_pnl` |
| 首行 SELL(15) → 后来追加 BUY(13) | 新行 13 列，比表头少 2 列 → `DictReader` 把后两列填成 `None` |

错位是**静默**的：不报错、值看起来也像数字 → 脏值直接进净值曲线
（§12.43 修过一次同类事故：11-02 的 `portfolio_value` 被写成单笔盈亏 -54.89）。

当前磁盘上所有 ledger 还是 13/13 整齐（现存 SELL 行是老代码写的 13 键版本），
**下一次有卖出的调仓即触发**。

## 本文件守什么

1. `append_ledger` 产出的文件：**表头 == LEDGER_FIELDS**，且**每行长度 == 表头长度**
2. 旧文件（13 列）追加时被**原子迁移**到 15 列，且迁移后按列名能读回正确的
   `cash_after` / `portfolio_value`（不是错位的值）
3. `_normalize_ledger_file` **幂等**：已是 15 列就不动（不产生无谓写盘）
4. 真实 `data/paper/` 产物：所有 ledger 行长度一致（防回归）
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from scripts import paper_portfolio as pp

BACKEND_ROOT = Path(__file__).resolve().parents[3]
PAPER_DIR = BACKEND_ROOT.parent / "data" / "paper"


def _buy_row(**over: Any) -> Dict[str, Any]:
    row = {
        "date": "2026-11-09", "action": "BUY", "code": "600000", "name": "浦发银行",
        "price": 10.0, "shares": 100, "amount": 1000.0, "commission": 5.0,
        "stamp_tax": 0.0, "slippage": 1.0, "score": 1.5,
        "cash_after": 998994.0, "portfolio_value": 1000000.0,
    }
    row.update(over)
    return row


def _sell_row(**over: Any) -> Dict[str, Any]:
    row = {
        "date": "2026-11-16", "action": "SELL", "code": "600000", "name": "浦发银行",
        "price": 11.0, "shares": 100, "amount": 1100.0, "commission": 5.0,
        "stamp_tax": 0.55, "slippage": 1.1, "score": 1.5,
        "cost_amount": 1000.0, "realized_pnl": 93.35,
        "cash_after": 1000086.8, "portfolio_value": 1000086.8,
    }
    row.update(over)
    return row


@pytest.fixture()
def paper_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pp, "PAPER_DIR", str(tmp_path))
    return tmp_path


def _rows_of(path: Path) -> List[List[str]]:
    with open(path, newline="") as fh:
        return list(csv.reader(fh))


# ══════════════════════════════════════════════════════════════
# 1. append_ledger 的列契约
# ══════════════════════════════════════════════════════════════

class TestAppendLedgerColumnContract:

    def test_new_file_header_is_full_contract(self, paper_dir):
        pp.append_ledger("t1", [_buy_row()])
        p = paper_dir / "t1_ledger.csv"
        header = _rows_of(p)[0]
        assert header == pp.LEDGER_FIELDS
        assert len(header) == 15

    def test_sell_first_batch_still_full_contract(self, paper_dir):
        """首行是 SELL 时表头也必须是完整 15 列（不是 13 列）。"""
        pp.append_ledger("t2", [_sell_row()])
        header = _rows_of(paper_dir / "t2_ledger.csv")[0]
        assert header == pp.LEDGER_FIELDS

    def test_buy_then_sell_keeps_row_width(self, paper_dir):
        """先 BUY 后 SELL：两次追加后所有行长度必须都等于表头长度。"""
        pp.append_ledger("t3", [_buy_row()])
        pp.append_ledger("t3", [_sell_row()])
        rows = _rows_of(paper_dir / "t3_ledger.csv")
        header = rows[0]
        assert header == pp.LEDGER_FIELDS
        for i, r in enumerate(rows[1:], start=2):
            assert len(r) == len(header), f"第{i}行长度 {len(r)} != 表头 {len(header)}"

    def test_sell_then_buy_keeps_row_width(self, paper_dir):
        """先 SELL 后 BUY：同样必须等宽。"""
        pp.append_ledger("t4", [_sell_row()])
        pp.append_ledger("t4", [_buy_row()])
        rows = _rows_of(paper_dir / "t4_ledger.csv")
        header = rows[0]
        assert header == pp.LEDGER_FIELDS
        for i, r in enumerate(rows[1:], start=2):
            assert len(r) == len(header), f"第{i}行长度 {len(r)} != 表头 {len(header)}"

    def test_migrated_old_file_reads_back_correctly(self, paper_dir):
        """**核心**：13 列旧文件追加 SELL 后，按列名读出的 cash_after 必须正确。

        修复前这里会读到 cost_amount（静默错位）。
        """
        p = paper_dir / "t5_ledger.csv"
        legacy_header = ["date", "action", "code", "name", "price", "shares", "amount",
                         "commission", "stamp_tax", "slippage", "score", "cash_after",
                         "portfolio_value"]
        with open(p, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(legacy_header)
            w.writerow(["2026-11-02", "BUY", "600000", "浦发银行", 10.0, 100, 1000.0,
                        5.0, 0.0, 1.0, 1.5, 31360.0, 978982.94])

        pp.append_ledger("t5", [_sell_row(cash_after=1000086.8, portfolio_value=1000086.8)])

        rows = _rows_of(p)
        assert rows[0] == pp.LEDGER_FIELDS
        assert all(len(r) == len(rows[0]) for r in rows[1:])

        with open(p, newline="") as fh:
            parsed = list(csv.DictReader(fh))
        # 旧行：cash_after / portfolio_value 必须留在原位
        assert parsed[0]["cash_after"] == "31360.0"
        assert parsed[0]["portfolio_value"] == "978982.94"
        # 新行：15 列全对位
        assert parsed[1]["cash_after"] == "1000086.8"
        assert parsed[1]["portfolio_value"] == "1000086.8"
        assert parsed[1]["cost_amount"] == "1000.0"
        assert parsed[1]["realized_pnl"] == "93.35"

    def test_backup_is_written_on_migration(self, paper_dir):
        p = paper_dir / "t6_ledger.csv"
        with open(p, "w", newline="") as fh:
            csv.writer(fh).writerow(["date", "action", "code", "name", "price", "shares",
                                     "amount", "commission", "stamp_tax", "slippage", "score",
                                     "cash_after", "portfolio_value"])
        pp.append_ledger("t6", [_buy_row()])
        assert (paper_dir / "t6_ledger.csv.bak").exists()

    def test_normalize_is_idempotent(self, paper_dir):
        p = paper_dir / "t7_ledger.csv"
        pp.append_ledger("t7", [_buy_row()])
        before = p.read_bytes()
        assert pp._normalize_ledger_file(str(p)) is False
        assert p.read_bytes() == before, "已是 15 列的文件不该被重写"

    def test_unknown_keys_are_ignored_not_crashing(self, paper_dir):
        """行里有额外键也不能炸（extrasaction='ignore'）。"""
        pp.append_ledger("t8", [_buy_row(some_new_field=1.0)])
        rows = _rows_of(paper_dir / "t8_ledger.csv")
        assert rows[0] == pp.LEDGER_FIELDS
        assert len(rows[1]) == len(rows[0])


# ══════════════════════════════════════════════════════════════
# 2. 真实产物一致性（防回归）
# ══════════════════════════════════════════════════════════════

class TestRealLedgerArtifacts:

    def _ledgers(self):
        if not PAPER_DIR.exists():
            pytest.skip("data/paper 不存在")
        return sorted(PAPER_DIR.glob("*_ledger.csv"))

    def test_all_rows_match_header_width(self):
        bad = []
        for p in self._ledgers():
            rows = _rows_of(p)
            if not rows:
                continue
            width = len(rows[0])
            for i, r in enumerate(rows[1:], start=2):
                if len(r) != width:
                    bad.append(f"{p.name} 第{i}行长度 {len(r)} != 表头 {width}")
        assert not bad, "\n  ".join(bad[:20])

    def test_single_line_ledgers_use_full_contract(self):
        """单线/A腿/袖套的 ledger 走 append_ledger → 必须已是 15 列契约。

        合并账本 multi_asset_final_ledger.csv 走全量重写路径，多一列 leg，不在此门禁内。
        """
        bad = []
        for p in self._ledgers():
            if p.name == "multi_asset_final_ledger.csv":
                continue
            rows = _rows_of(p)
            if rows and rows[0] != pp.LEDGER_FIELDS:
                bad.append(f"{p.name} 表头 {rows[0]}")
        assert not bad, "以下 ledger 尚未迁移到 15 列契约：\n  " + "\n  ".join(bad)
