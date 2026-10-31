#!/usr/bin/env python3
"""multi_asset_final 组合线的端到端测试（不碰真实纸面盘文件）。

测什么：
  ① 两条腿都建仓：A 腿 60 只、袖套 9 只
  ② 袖套三类权重 ≈ 1/3（类间等权真的达成了，不是"看着像"）
  ③ 袖套每只 ≈ 1/9，合计 ≈ 1.0
  ④ 合并账本：两腿的成交都在，leg 标签正确
  ⑤ 幂等：同一成交日重跑 → **不重复下单**（账本行数不变）
  ⑥ 袖套调仓日网格：第二次调仓要等 21 个交易日
  ⑦ 持仓估值 = 现金 + 两条腿市值（与写入的 last_value 一致）

用法:
    cd backend && env -u PYTHONPATH -u PYTHONHOME .venv-py313/bin/python3 \
        scripts/test_multi_asset_line.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from typing import Any, Dict, List

import pandas as pd

BACKEND = "/Users/rone/Projects/stock-platform/backend"
sys.path.insert(0, BACKEND)


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PP = _load("PP", os.path.join(BACKEND, "scripts", "paper_portfolio.py"))

# 测试用的历史信号日（周五）；下一交易日 2026-10-19
REBALANCE_DAY = pd.Timestamp("2026-10-16")
CAPITAL = 1_000_000.0

FAIL: List[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        print(f"  ✅ {label}" + (f"  ({detail})" if detail else ""))
    else:
        print(f"  ❌ {label}" + (f"  ({detail})" if detail else ""))
        FAIL.append(label)


def read_ledger(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    tmp = tempfile.mkdtemp(prefix="ma_line_test_")
    PP.PAPER_DIR = tmp          # 不碰真实纸面盘
    print(f"[test] 临时目录 {tmp}")

    codes = PP.ds.list_universe()
    frames, panel, master = PP.ds.load_data(codes)
    names = PP.load_name_map()
    print(f"[test] 数据加载完成，master {master[0].date()} → {master[-1].date()}")

    extras: Dict[str, Any] = {}
    adj_close = None
    if PP.BAOSTOCK_READY:
        extras = PP.load_market_extras(codes, master)
        adj_close = PP.load_adjusted_close(codes, master)

    print("\n[test] ① 第一次建仓")
    r1 = PP.run_multi_asset_line(
        frames, panel, master, names, CAPITAL, REBALANCE_DAY,
        extras=extras, adj_close=adj_close,
    )
    print(f"      信号日 {r1['signal_date']} 成交日 {r1['trade_date']} "
          f"总值 {r1['value']:,.0f} 现金 {r1['cash']:,.0f} "
          f"持仓 {r1['n_positions']} 买 {len(r1['buys'])} 卖 {len(r1['sells'])}")

    a_pos = [c for c, p in (PP.load_holdings("multi_asset_final_a", CAPITAL)["positions"]).items()]
    s_hold = PP.load_holdings("multi_asset_final_sleeve", CAPITAL / 2)
    s_pos = s_hold["positions"]
    # ⚠️ 不是硬卡 60：A 腿的每只预算只有 50万×0.95/60 ≈ 7,900 元，
    #    买不了整手的、以及被流动性门槛拦下的（成交额 < 500 万）都会缺。
    #    缺的数量必须**可解释**（见下面 blocked_buys 的打印），不是随机的。
    check(55 <= len(a_pos) <= 60, "A 腿建仓 55~60 只", f"实际 {len(a_pos)}")
    check(len(s_pos) == 9, "袖套建仓 9 只 ETF", f"实际 {len(s_pos)}")
    check(r1["n_positions"] == len(a_pos) + len(s_pos),
          "合并持仓 = A腿 + 袖套", f"实际 {r1['n_positions']} "
          f"(A {len(a_pos)} + 袖套 {len(s_pos)})")
    blocked = r1.get("blocked_buys") or []
    print(f"      A 腿被拦下 {len(blocked)} 只："
          f"{[b.get('reason', '')[:28] for b in blocked[:4]]}")

    print("\n[test] ②③ 袖套权重结构")
    legs = r1["legs"]["sleeve"]
    cw = legs["class_weights_now"]
    tw = legs["target_weights"]
    check(all(abs(cw.get(c, 0) - 1 / 3) < 0.01 for c in ("gold", "bond", "commodity")),
          "三类权重 ≈ 1/3", json.dumps({k: round(v, 4) for k, v in cw.items()}))
    # ⚠️ 容差 1e-5 / 1e-6：目标权重是**写进 JSON 的 6 位小数**，
    #    1/9 = 0.111111... 存成 0.111111（差 1.1e-7），合计 0.999999（差 1e-6）。
    #    用 1e-9 去卡是在卡 JSON 的舍入，不是在卡逻辑。
    check(abs(sum(tw.values()) - 1.0) < 1e-5, "目标权重合计 = 1.0",
          f"{sum(tw.values()):.12f}")
    check(all(abs(v - 1 / 9) < 1e-6 for v in tw.values()),
          "每只目标权重 = 1/9", f"{min(tw.values()):.9f}~{max(tw.values()):.9f}")
    check(legs["rebalanced"] is True, "首次建仓触发调仓")
    # 投入率：整手取整不该让大笔资金躺在现金里（修之前 A 腿只有 81%）
    ir_a = (r1["legs"]["a"] or {}).get("invested_ratio")
    ir_s = (r1["legs"]["sleeve"] or {}).get("invested_ratio")
    print(f"      投入率：A 腿 {ir_a} / 袖套 {ir_s}")
    check(ir_a is not None and ir_a >= 0.90, "A 腿投入率 ≥ 90%",
          f"实际 {ir_a}")
    check(ir_s is not None and ir_s >= 0.95, "袖套投入率 ≥ 95%",
          f"实际 {ir_s}")

    print("\n[test] ④ 合并账本")
    rows = read_ledger(PP.ledger_path("multi_asset_final"))
    legs_in_rows = {r.get("leg") for r in rows}
    check(len(rows) > 0, "合并账本有行", f"{len(rows)} 行")
    check("a" in legs_in_rows and "sleeve" in legs_in_rows,
          "两腿都在账本里", str(legs_in_rows))
    a_rows = [r for r in rows if r.get("leg") == "a"]
    s_rows = [r for r in rows if r.get("leg") == "sleeve"]
    check(len(a_rows) == len(a_pos), "A 腿买入笔数 = 持仓数",
          f"{len(a_rows)} vs {len(a_pos)}")
    check(len(s_rows) == 9, "袖套 9 笔买入", f"{len(s_rows)}")
    check(all(r["action"] == "BUY" for r in s_rows), "袖套首笔全是 BUY")

    print("\n[test] ⑤ 幂等：同一成交日重跑")
    n_before = len(read_ledger(PP.ledger_path("multi_asset_final")))
    r2 = PP.run_multi_asset_line(
        frames, panel, master, names, CAPITAL, REBALANCE_DAY,
        extras=extras, adj_close=adj_close,
    )
    n_after = len(read_ledger(PP.ledger_path("multi_asset_final")))
    check(n_after == n_before, "账本行数不变（没重复下单）",
          f"{n_before} → {n_after}")
    check(len(r2["buys"]) == 0, "第二次运行没有新买入", f"{len(r2['buys'])} 笔")

    print("\n[test] ⑥⑦ 估值自洽 + 调仓网格")
    h = PP.load_holdings("multi_asset_final", CAPITAL)
    check(abs(h["last_value"] - r2["value"]) < 0.01,
          "持仓文件 last_value 与运行结果一致",
          f"{h['last_value']:.2f} vs {r2['value']:.2f}")
    check(h["initial_capital"] == CAPITAL, "初始资金 = 100 万")
    check(len(h["nav_history"]) == 1, "净值历史 1 个点（首笔）",
          f"{len(h['nav_history'])}")
    check(h["nav_history"][0]["date"] == r1["trade_date"], "净值日期 = 成交日")
    check(h["legs"]["a"]["weight"] == 0.5 and h["legs"]["sleeve"]["weight"] == 0.5,
          "两腿权重各 50%")
    check(legs["next_rebalance_in_trading_days"] == 21,
          "下次调仓在 21 个交易日后",
          str(legs["next_rebalance_in_trading_days"]))

    # 账本合并后现金是否自洽
    last_cash = float(rows[-1]["cash_after"]) if rows else 0.0
    check(abs(last_cash - h["cash"]) < 1.0,
          "账本末行 cash_after ≈ 持仓文件 cash",
          f"{last_cash:.2f} vs {h['cash']:.2f}")

    print("\n" + "=" * 60)
    if FAIL:
        print(f"❌ 失败 {len(FAIL)} 项：{FAIL}")
        sys.exit(1)
    print("✅ 全部通过")
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
