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
    # 多资产组合线（2026-11-01）：平台旗舰线，A 股腿 + 黄金/债券/商品袖套。
    # 默认推送仍走 honest_42f（主线），这条用 --strategy multi_asset_final 推。
    "multi_asset_final": "多资产组合（A股+黄金/债券/商品）",
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
    # 该线的历史业绩 + 信号质量（可信度凭证）
    _perf = load_json(os.path.join(PAPER_DIR, f"{key}_holdings.json")) or {}
    _q = load_json(os.path.join(PAPER_DIR, f"{key}_signal_quality.json")) or {}
    _cap = float(_perf.get("initial_capital") or 1_000_000)
    _val = float(_perf.get("last_value") or 0)
    if _val > 0:
        out.append(f"  组合 ¥{_val:,.0f}（{(_val/_cap-1)*100:+.2f}%）· "
                   f"{len(_perf.get('positions') or {})} 只")
    _t5 = (_q.get("horizons") or {}).get("T+5")
    if _t5:
        out.append(f"  信号质量 T+5 胜率 {_t5['win_rate']*100:.0f}% · "
                   f"平均超额 {_t5['avg_excess']*100:+.2f}%（{_t5['n']} 个历史信号）")
    # ⚠️ 2026-10-27：最近 12 个月的表现（诚实披露「当下处于什么状态」）
    # 背景：6 个半年窗口的稳定性检验显示，最近窗口（2025Q4）夏普只有 0.221，
    # 而 2025 上半年是 2.887 —— 策略当下处于低效期，必须让用户知道。
    _nav = _perf.get("nav_history") or []
    if len(_nav) >= 2:
        try:
            _last = _nav[-1]
            # 找到约 12 个月前的点
            _tgt = None
            _d0 = str(_last.get("date") or "")[:10]
            _best = None
            for _p in _nav:
                _d = str(_p.get("date") or "")[:10]
                if not _d or not _d0:
                    continue
                _gap = (int(_d0[:4]) - int(_d[:4])) * 365 + \
                       (int(_d0[5:7]) - int(_d[5:7])) * 30
                if _best is None or _gap > _best[0]:
                    _best = (_gap, _p)
            _tgt = _best[1] if _best else None
            if _tgt and _tgt is not _last:
                _r12 = (float(_last.get("value") or 0)
                        / float(_tgt.get("value") or 1) - 1.0)
                out.append(f"  近 12 个月 {_r12*100:+.1f}%"
                           f"（{_tgt.get('date')} → {_last.get('date')}）")
        except Exception:
            pass
    def _shares_of(d: dict):
        """兼容 shares / quantity 两种字段名（portfolio.json 用 quantity，
        ledger.csv 用 shares —— 2026-10-27 前推送读错了字段名导致显示 None 股）"""
        v = d.get("shares")
        if v is None:
            v = d.get("quantity")
        try:
            return int(v or 0)
        except (TypeError, ValueError):
            return 0

    def _fee_of(d: dict):
        """费用 = 佣金 + 印花税 + 滑点（fee 字段缺失或为 0 时按明细求和）"""
        detail = ((d.get("commission") or 0) + (d.get("stamp_tax") or 0)
                  + (d.get("slippage") or 0))
        v = d.get("fee")
        try:
            fv = float(v or 0.0)
        except (TypeError, ValueError):
            fv = 0.0
        return detail if detail else fv

    if buys:
        out.append("买入：")
        for b in buys:
            code = b.get("code", "")
            nm = b.get("name") or names.get(code, "")
            amount = b.get("amount")
            out.append(
                f"  {code} {nm}  {b.get('price')} 元 × {_shares_of(b)} 股"
                f" = {fmt_money(amount)} 元（费 {_fee_of(b):.1f}）"
            )
    if sells:
        out.append("卖出：")
        for s in sells:
            code = s.get("code", "")
            nm = s.get("name") or names.get(code, "")
            amount = s.get("amount")
            out.append(
                f"  {code} {nm}  {s.get('price')} 元 × {_shares_of(s)} 股"
                f" = {fmt_money(amount)} 元（费 {_fee_of(s):.1f}）"
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

    trade_date = meta.get("trade_date")
    if trade_date and trade_date != sig_date:
        # lag1 口径：信号在周五收盘产生，下一交易日才成交。
        # 用户要照着做，必须告诉他**哪天操作**。
        header = (f"📊 交易信号（信号日 {sig_date} → 成交日 {trade_date}）\n"
                  f"   成交价用 {trade_date} 收盘价，涨停买不进、跌停卖不出")
    else:
        header = f"📊 每日信号 {sig_date}（{meta.get('data_latest', sig_date)} 数据）"
    print(header)
    print("=" * 28)
    print("\n\n".join(chunks))


if __name__ == "__main__":
    main()
