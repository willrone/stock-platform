#!/usr/bin/env python3
"""每日信号摘要（供 Hermes cron 推送到微信，--no-agent 模式直接输出 stdout）。

设计要点：
  - **不经过 LLM**：cron 用 `--no-agent` 跑本脚本，stdout 原样推送 → 零延迟零成本
  - **空输出 = 静默**：非交易日 / 无新信号时不打扰用户
  - **内容要能照着做**：代码 + 名称 + 价格 + 股数 + 金额 + 成本
  - **只推主线策略**：默认推 honest_42f（周频夏普最高的一条），
    --all 可推全部有信号的线

用法:
  env -u PYTHONPATH backend/.venv-py313/bin/python3 backend/scripts/push_daily_signal.py
  ... --all          # 推全部策略线
  ... --strategy rsi_71421
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAPER_DIR = os.path.join(ROOT, "data", "paper")
RAW_DIR = os.path.join(ROOT, "data", "tdx_daily", "raw")

# 主线策略（按 3 年全市场周频夏普排序）
DEFAULT_STRATEGY = "honest_42f"
DISPLAY = {
    "honest_42f": "42 因子等权",
    "rsi_71421": "RSI 7/14/21",
    "gbm_voltgt": "LightGBM + 波动率目标",
    "factor_ranking": "动量/低波动/低振幅",
    "22f": "22 因子（对照）",
}


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def name_map() -> dict:
    if not os.path.isdir(RAW_DIR):
        return {}
    dates = sorted(d for d in os.listdir(RAW_DIR) if d.startswith("date="))
    if not dates:
        return {}
    try:
        import pandas as pd

        df = pd.read_parquet(
            os.path.join(RAW_DIR, dates[-1], "part.parquet"),
            columns=["ts_code", "name"],
        )
        return dict(zip(df["ts_code"].astype(str), df["name"].astype(str)))
    except Exception:
        return {}


def fmt_money(v) -> str:
    try:
        return f"{float(v):,.0f}"
    except (TypeError, ValueError):
        return "-"


def build_line(key: str, portfolio: dict, names: dict) -> str | None:
    lines = (portfolio.get("lines") or {}).get(key)
    if not lines:
        return None
    holdings = load_json(os.path.join(PAPER_DIR, f"{key}_holdings.json")) or {}
    positions = holdings.get("positions") or {}

    buys = lines.get("buys") or []
    sells = lines.get("sells") or []
    holds = lines.get("holds") or []

    # 没有买卖且不是首次建仓 → 不打扰
    if not buys and not sells and positions:
        return None

    out = [f"【{DISPLAY.get(key, key)}】{lines.get('date','')}"]
    if buys:
        out.append("买入：")
        for b in buys:
            code = b.get("code", "")
            nm = b.get("name") or names.get(code, "")
            shares = b.get("shares")
            amount = b.get("amount")
            cost = (b.get("commission") or 0) + (b.get("slippage") or 0)
            out.append(
                f"  {code} {nm}  {b.get('price')} 元 × {shares} 股"
                f" = {fmt_money(amount)} 元（费 {cost:.1f}）"
            )
    if sells:
        out.append("卖出：")
        for s in sells:
            code = s.get("code", "")
            nm = s.get("name") or names.get(code, "")
            shares = s.get("shares")
            amount = s.get("amount")
            fee = (s.get("commission") or 0) + (s.get("stamp_tax") or 0) + (s.get("slippage") or 0)
            out.append(
                f"  {code} {nm}  {s.get('price')} 元 × {shares} 股"
                f" = {fmt_money(amount)} 元（费 {fee:.1f}）"
            )
    if holds and not buys and not sells:
        out.append(f"继续持有 {len(holds)} 只，无操作")

    # 未成交的信号（实盘同样买不进/卖不出）—— 必须告诉用户，
    # 否则他会以为平台漏单
    blocked = (lines.get("blocked_buys") or []) + (lines.get("blocked_sells") or [])
    if blocked:
        out.append("未成交（实盘同样做不到）：")
        for b in blocked[:6]:
            kind = "买不进" if b.get("code") in {
                x.get("code") for x in (lines.get("blocked_buys") or [])
            } else "卖不出"
            out.append(f"  {kind} {b.get('code')} {b.get('name','')} —— {b.get('reason','')}")

    val = holdings.get("last_value")
    cash = holdings.get("cash")
    cap = holdings.get("initial_capital") or 1_000_000
    if val:
        pnl = (float(val) / cap - 1) * 100
        out.append(f"组合 ¥{fmt_money(val)}（{pnl:+.2f}%）现金 ¥{fmt_money(cash)}")
    out.append("按下一交易日可成交价执行；涨停买不进、跌停卖不出")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default=DEFAULT_STRATEGY)
    ap.add_argument("--all", action="store_true", help="推全部策略线")
    args = ap.parse_args()

    portfolio = load_json(os.path.join(PAPER_DIR, "portfolio.json"))
    if not portfolio:
        return  # 无数据 → 空输出 → 静默

    meta = portfolio.get("meta") or {}
    sig_date = meta.get("date")
    if not sig_date:
        return

    # 新鲜度保险：信号日期距今超过 7 天不推（防止数据更新失败时推旧信号）
    try:
        sig_dt = datetime.strptime(str(sig_date), "%Y-%m-%d")
    except (TypeError, ValueError):
        return
    if (datetime.now() - sig_dt).days > 7:
        return

    names = name_map()
    keys = list((portfolio.get("lines") or {}).keys()) if args.all else [args.strategy]
    chunks = []
    for k in keys:
        c = build_line(k, portfolio, names)
        if c:
            chunks.append(c)

    if not chunks:
        return  # 无操作 → 静默

    header = f"📊 每日信号 {sig_date}（{meta.get('data_latest', sig_date)} 数据）"
    print(header)
    print("=" * 28)
    print("\n\n".join(chunks))


if __name__ == "__main__":
    main()
