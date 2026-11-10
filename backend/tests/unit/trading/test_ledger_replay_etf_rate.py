"""`ledger_replay.cash_model="etf_rate"` 的门禁测试（2026-11-05，自用化 A 项）。

## 为什么单独一个文件

袖套腿（ETF）和 A 股腿的费用结构**不同**。混用会把现金算成负数：
实测真实袖套账本（9 笔首建，成交额 ¥499,980）

| cash_model | 重放出的现金 | 真值 | 残差 |
|---|---|---|---|
| `rate`（A 股三项费用） | −479.57 | +20.27 | **−499.84** |
| `ledger_columns` | −29.73 | +20.27 | **−50.00** |
| `etf_rate` | +20.2611 | +20.27 | **−0.0089** ✓ |

本文件把这三件事钉住：谁把袖套改回 `rate`、或把费率"抄"成字面常量，测试就红。

## 守什么

| # | 门禁 | 为什么 |
|---|---|---|
| a | ETF 买/卖的现金影响 = **只有佣金**（无印花税、无二次滑点、无 5 元最低佣金） | 与生产 `rebalance_sleeve` 逐项一致 |
| b | `rate` 用在 ETF 上**多扣**的金额必须等于 `滑点 + 印花税` | 锁住"为什么必须分开"这个结论 |
| c | ETF 费率常量必须 **import 自 app 层**，不是抄数字 | 抄数字 = 口径漂移的起点 |
| d | 未知 `cash_model` 仍然报错（不静默回退） | 拼错字符串不该变成"算错了但看起来正常" |
| e | 真实袖套账本的重放残差 ≤ 0.05（有数据时才跑） | 真实产物的端到端证据 |
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Optional

import pytest

BACKEND = Path(__file__).resolve().parents[3]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

SCRIPT = BACKEND / "scripts" / "ledger_replay.py"
REAL_SLEEVE_LEDGER = BACKEND.parent / "data" / "paper" / "multi_asset_final_sleeve_ledger.csv"
REAL_SLEEVE_HOLDINGS = BACKEND.parent / "data" / "paper" / "multi_asset_final_sleeve_holdings.json"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # ⚠️ 必须先注册再 exec：模块里的 `@dataclass` + `from __future__ import annotations`
    # 会让 dataclasses 回 sys.modules 找类所在模块（不注册 → AttributeError）
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def lrmod():
    return _load(SCRIPT, "ledger_replay_etf_rate_test")


def _row(lrmod, date: str, action: str, code: str, price: float,
         shares: float, amount: float, commission: float = 0.0,
         stamp_tax: float = 0.0, slippage: float = 0.0):
    return lrmod.LedgerRow(
        date=date, action=action, code=code, name="测试ETF",
        price=price, shares=shares, amount=amount,
        commission=commission, stamp_tax=stamp_tax, slippage=slippage,
    )


# ── a：ETF 现金影响 = 只有佣金 ─────────────────────────────────────────
def test_etf_buy_and_sell_charge_commission_only(lrmod):
    buy = _row(lrmod, "2026-11-02", "BUY", "518880", 7.941, 7000, 55592.56)
    sell = _row(lrmod, "2026-11-03", "SELL", "518880", 7.941, 7000, 55586.99)

    d_buy = lrmod.cash_delta(buy, cash_model="etf_rate")
    d_sell = lrmod.cash_delta(sell, cash_model="etf_rate")

    assert d_buy == pytest.approx(
        -(55592.56 * (1.0 + lrmod.ETF_COMMISSION_RATE)), abs=1e-6)
    assert d_sell == pytest.approx(
        55586.99 * (1.0 - lrmod.ETF_COMMISSION_RATE), abs=1e-6)


# ── b：默认 rate 用在 ETF 上多扣多少，必须可解释 ───────────────────────
def test_rate_model_double_counts_slippage_and_adds_stamp_tax(lrmod):
    """`rate` 对 ETF 的**多扣**必须正好等于「滑点(重复扣一遍) + 印花税」。

    买：`rate` 多扣 `gross × SLIPPAGE_RATE`（滑点已在成交价里）。
    卖：`rate` 多扣 `gross × (STAMP_TAX_RATE + SLIPPAGE_RATE)`。
    有人觉得「反正都叫费率，统一用 rate 就行」→ 这条测试立刻红。
    """
    gross_buy, gross_sell = 55592.56, 55586.99
    buy = _row(lrmod, "2026-11-02", "BUY", "518880", 7.941, 7000, gross_buy)
    sell = _row(lrmod, "2026-11-03", "SELL", "518880", 7.941, 7000, gross_sell)

    buy_extra = (lrmod.cash_delta(buy, "etf_rate")
                 - lrmod.cash_delta(buy, "rate"))
    sell_extra = (lrmod.cash_delta(sell, "etf_rate")
                  - lrmod.cash_delta(sell, "rate"))

    assert buy_extra == pytest.approx(gross_buy * lrmod.SLIPPAGE_RATE, abs=1e-6)
    assert sell_extra == pytest.approx(
        gross_sell * (lrmod.STAMP_TAX_RATE + lrmod.SLIPPAGE_RATE), abs=1e-6)


def test_etf_buy_ignores_five_yuan_minimum_commission(lrmod):
    """小额腿（几百元）不能被 5 元最低佣金放大成本。"""
    tiny = _row(lrmod, "2026-11-02", "BUY", "159981", 1.941, 100, 194.1)
    d_etf = lrmod.cash_delta(tiny, cash_model="etf_rate")
    d_rate = lrmod.cash_delta(tiny, cash_model="rate")
    # etf: 佣金 194.1×0.0003 = 0.058（不抬到 5 元）
    assert d_etf == pytest.approx(-(194.1 + 194.1 * lrmod.ETF_COMMISSION_RATE),
                                  abs=1e-6)
    # rate: 佣金被抬到 5 元 + 又扣 0.194 滑点
    assert d_rate < d_etf


# ── c：费率必须与 app 层生产实现同源 ──────────────────────────────────
def test_etf_rates_are_imported_from_app_layer_not_copied(lrmod):
    from app.services.trading import multi_asset_service as mas
    assert lrmod.ETF_COMMISSION_RATE is mas.ETF_COMMISSION, \
        "ETF 佣金必须是 import 进来的同一个对象，不是抄的字面常量"
    assert lrmod.ETF_SLIPPAGE_RATE is mas.ETF_SLIPPAGE, \
        "ETF 滑点必须是 import 进来的同一个对象，不是抄的字面常量"


# ── d：未知 cash_model 必须报错 ───────────────────────────────────────
def test_unknown_cash_model_raises(lrmod):
    row = _row(lrmod, "2026-11-02", "BUY", "518880", 7.941, 7000, 55592.56)
    with pytest.raises(ValueError):
        lrmod.cash_delta(row, cash_model="etf")
    # `ledger_columns` 仍在（作对照/审计），但拼错的名字不能静默回退
    assert lrmod.cash_delta(row, cash_model="ledger_columns") is not None


def test_replay_rows_rejects_unknown_cash_model(lrmod):
    row = _row(lrmod, "2026-11-02", "BUY", "518880", 7.941, 7000, 55592.56)
    with pytest.raises(ValueError):
        lrmod.replay_rows([row], initial_capital=500000.0, cash_model="bogus")


# ── e：真实袖套账本端到端（有 data/paper 时才跑）───────────────────────
@pytest.mark.skipif(not (os.path.exists(REAL_SLEEVE_LEDGER)
                         and os.path.exists(REAL_SLEEVE_HOLDINGS)),
                    reason="真实 data/paper 不在（CI 无产物）")
def test_real_sleeve_ledger_replay_residual_within_gate(lrmod):
    import json
    rows, _diag = lrmod.parse_ledger(str(REAL_SLEEVE_LEDGER))
    hold = json.load(open(REAL_SLEEVE_HOLDINGS))
    cap0 = float(hold.get("initial_capital") or 500_000.0)
    snaps, _r = lrmod.replay_rows(rows, initial_capital=cap0,
                                  semantics="moving_average",
                                  cash_model="etf_rate")
    residual = float(snaps[-1]["cash"]) - float(hold.get("cash") or 0.0)
    assert abs(residual) <= 0.05, f"重放现金残差 {residual} 超门禁"

    # 反面对照：A 股费率模型必须**不过**门禁（证明分开建模不是形式主义）
    snaps_rate, _r2 = lrmod.replay_rows(rows, initial_capital=cap0,
                                        semantics="moving_average",
                                        cash_model="rate")
    residual_rate = float(snaps_rate[-1]["cash"]) - float(hold.get("cash") or 0.0)
    assert abs(residual_rate) > 0.05
