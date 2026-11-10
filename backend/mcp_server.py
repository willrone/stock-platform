#!/usr/bin/env python3
"""股票平台 MCP server (stdio).

暴露的工具:
  list_strategies   - 列出所有可用策略
  stock_quote       - 单只股票最近行情 (不复权)
  paper_portfolio   - 22f 纸面组合持仓 + 按最新价 mark-to-market
  latest_signals    - 最新一次 22f 周频信号快照 (只读)
  daily_signal      - 跑 42f 日频信号 (Top60 买/卖/留, 只读)
  weekly_rebalance  - 跑 22f 周频调仓 (会更新纸面账本!)
  run_backtest      - 用任意策略跑回测 (直驱 BacktestExecutor)

运行:
  cd backend && env -u PYTHONPATH -u PYTHONHOME .venv/bin/python3 mcp_server.py
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND = os.path.join(ROOT, "backend")
DATA = os.path.join(ROOT, "data")
STOCK_DIR = os.path.join(DATA, "parquet", "stock_data")
SIGNAL_DIR = os.path.join(DATA, "signals")
sys.path.insert(0, BACKEND)

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("stock-platform")

PY313 = os.path.join(BACKEND, ".venv-py313", "bin", "python3")

# 策略说明 (给 MCP 调用方看的)
_STRATEGY_NOTES: Dict[str, str] = {
    # 2026-10-25 全部换成实测口径（3 年全市场 5513 / 周频调仓）
    "honest_composite": "42f 等权复合, 免费档主力 (3年周频夏普 0.965, 回撤 -17.6%, 交易 1390 笔)",
    "gbm_voltgt": "GBM + 波动率目标, 付费档 (3年周频夏普 0.835, 回撤 -29.8%, 交易 620 笔)",
    "factor_ranking": "横截面复合因子 + TopK-Dropout (3年周频夏普 0.768, 回撤 -14.2%)",
    "rsi_71421": "RSI 7/14/21 均值回归 (3年周频夏普 0.803, 回撤 -16.3%, 收益 +37.8%)",
    "low_volatility": "低波动横截面 (3年周频夏普 0.835, 2026-10-25 修复只买不卖缺陷)",
    "multi_factor": "value/momentum/low_vol 横截面 (3年周频夏普 0.411, 表现偏弱)",
    "moving_average": "MA 趋势跟随 (3年全市场 -1.5%, 夏普 0.110 —— 不赚钱, 不建议使用)",
    "lightgbm": "LightGBM 模型 (2026-10-25 修复 pkl 段错误; 3年全市场 +27.1%/0.430)",
    "model_signal": "模型预测信号 (段错误已修; 500只 +1.6%/0.184, 交易仅 8 笔)",
    "model_topk_dropout": "模型排序 TopK (段错误已修; 500只 +82.3%/0.647, 交易仅 8 笔)",
}


def _clean_env() -> Dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}


@contextlib.contextmanager
def _app_quiet():
    """导入/调用 app.* 时把其 stdout 日志赶到 stderr，避免污染 MCP 的 JSON-RPC 流。"""
    old = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old


def _run_signal_script(script: str, *args: str) -> str:
    """跑 daily/weekly_signal 脚本, 返回 stdout。"""
    r = subprocess.run(
        [PY313, os.path.join(BACKEND, "scripts", script), *args],
        capture_output=True, text=True, cwd=BACKEND, env=_clean_env(), timeout=900,
    )
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed rc={r.returncode}\n{r.stderr[-1000:]}")
    return r.stdout


def _load_parquet(code: str):
    import pandas as pd

    c = code.split(".")[0]
    # 优先带交易所后缀的 (真数据), 裸代码的可能是旧桩 (如 600000.parquet 止于 2023)
    for suffix in ("_SH", "_SZ", ""):
        p = os.path.join(STOCK_DIR, f"{c}{suffix}.parquet")
        if os.path.exists(p):
            return pd.read_parquet(p).sort_values("date").reset_index(drop=True)
    raise FileNotFoundError(f"找不到 {code} 的 parquet (试了 {c}_SH, {c}_SZ, {c})")


def _norm_code(code: str) -> str:
    """600000 / 600000.SH / 600000_SH → 统一返回, 不做存在性检查。"""
    return code.replace("_", ".").split(".")[0]


def _list_universe() -> List[str]:
    codes = []
    for f in sorted(os.listdir(STOCK_DIR)):
        if f.endswith(".parquet"):
            codes.append(f[:-8].replace("_", "."))
    return codes


def _latest_signal_file() -> Optional[str]:
    cands = [f for f in os.listdir(SIGNAL_DIR)
             if f.startswith("signal_") and f.endswith(".json") and "_42f" not in f]
    if not cands:
        return None
    return os.path.join(SIGNAL_DIR, sorted(cands)[-1])


@mcp.tool()
def list_strategies() -> str:
    """列出所有可用策略及说明。正式三件套: honest_composite (免费), gbm_voltgt (付费), factor_ranking (新)。"""
    with _app_quiet():
        from app.services.backtest.strategies.strategy_factory import StrategyFactory
        names = list(StrategyFactory._strategies.keys())
    out = {
        "strategies": [
            {"name": n, "note": _STRATEGY_NOTES.get(n, "")} for n in names
        ]
    }
    return json.dumps(out, ensure_ascii=False, indent=1)


@mcp.tool()
def stock_quote(code: str, days: int = 10) -> str:
    """查单只股票最近 N 天行情 (不复权)。code 如 600000 或 600000.SH。"""
    c = _norm_code(code)
    df = _load_parquet(c)
    tail = df.tail(max(1, days))[["date", "open", "high", "low", "close", "volume"]]
    rows = [
        {"date": str(r.date)[:10], "open": float(r.open), "high": float(r.high),
         "low": float(r.low), "close": float(r.close), "volume": float(r.volume)}
        for r in tail.itertuples(index=False)
    ]
    last = rows[-1]
    return json.dumps({
        "code": c, "rows": rows,
        "latest_close": last["close"],
        "data_through": last["date"],
    }, ensure_ascii=False, indent=1)


@mcp.tool()
def paper_portfolio(strategy: str = "") -> str:
    """多策略纸面盘: 每条线独立 100 万, 返回各线汇总 + 明细。

    2026-10-25 起默认读多策略盘（data/paper/portfolio.json，由
    backend/scripts/paper_portfolio.py 生成）。**当前 7 条线**：
      22f / honest_42f / rsi_71421 / factor_ranking / gbm_voltgt /
      honest_42f_neu / multi_asset_final（多资产组合：A 股腿 + 黄金/债券/商品袖套）。
    传 strategy=<名称> 只返回该条线的持仓明细。

    注意：`multi_asset_final` 是**严格 lag1**（周五信号 → 下一交易日成交），
    它的第一个净值点要等下一个交易日的收盘数据落地才出现。
    """
    import os as _os

    portfolio_path = _os.path.join(DATA, "paper", "portfolio.json")
    if _os.path.exists(portfolio_path):
        with open(portfolio_path) as fh:
            portfolio = json.load(fh)
        lines = portfolio.get("lines", {})
        # 观测口径层（附录 A，2026-11-02）：
        #   门禁 d → pnl_pct **原字段取值不变**（前端 TS 类型是 number，改 null
        #   会编译不过）；诚实信号放在新增字段上。
        #   为什么必须有这一层：7 条线暴露度差 5 倍（17.5%~92.1%），只看总盈亏
        #   会把"没建仓"读成"打平"，把"选股亏钱"读成"仓位小所以没亏"。
        _asofs = sorted({str(ln.get("as_of")) for ln in lines.values()
                         if ln.get("as_of")})
        _asof_consistent = len(_asofs) <= 1
        summary = {
            "date": portfolio.get("meta", {}).get("date"),
            "data_latest": portfolio.get("meta", {}).get("data_latest"),
            "capital_per_line": portfolio.get("meta", {}).get("capital_per_line"),
            # ── as-of 诚实：不一致就不许把排名当结论 ─────────────────────
            "as_of_dates": _asofs,
            "as_of_consistent": _asof_consistent,
            "note": (
                "" if _asof_consistent else
                "各线 as-of 不一致 → ranking_by_invested_return 仅供参考，"
                "不是同一时点的比较"
            ),
            "ranking_by_invested_return": [
                {
                    "strategy": _nm,
                    "invested_return_pct": _ln.get("invested_return_pct"),
                    "exposure": _ln.get("exposure"),
                    "pnl_display": _ln.get("pnl_display"),
                    "as_of": _ln.get("as_of"),
                    "status": _ln.get("status"),
                }
                for _nm, _ln in sorted(
                    lines.items(),
                    key=lambda kv: (kv[1].get("invested_return_pct") is None,
                                    -(kv[1].get("invested_return_pct") or 0.0)))
            ],
            "lines": {
                name: {
                    "value": ln.get("value"),
                    "cash": ln.get("cash"),
                    "n_positions": ln.get("n_positions"),
                    # 原字段：取值不变（门禁 d）
                    "pnl_pct": round(
                        (float(ln.get("value", 0))
                         / float(portfolio.get("meta", {}).get("capital_per_line", 1))
                         - 1) * 100, 2),
                    # 展示用：从未成交的线是 "—"，不是 0.00%
                    # （落盘值优先；旧 portfolio.json 没有该字段时现算）
                    "pnl_display": (
                        ln.get("pnl_display")
                        or ("—" if ln.get("n_trades") == 0
                            else (f"{float(ln.get('pnl_pct')):.2f}%"
                                  if ln.get("pnl_pct") is not None else "—"))
                    ),
                    # 暴露度 / 投入段收益 / as-of / 成交笔数 / 状态
                    # → 让不同仓位量的线**可比**（暴露度差 5 倍时总盈亏不可比）
                    "exposure": ln.get("exposure"),
                    "invested_return_pct": ln.get("invested_return_pct"),
                    "as_of": ln.get("as_of"),
                    "n_trades": ln.get("n_trades"),
                    "status": ln.get("status"),
                    "desc": ln.get("desc"),
                }
                for name, ln in lines.items()
            },
        }
        if not strategy:
            return json.dumps(summary, ensure_ascii=False, indent=1)
        if strategy not in lines:
            raise ValueError(
                f"未知策略 {strategy}，可选: {list(lines)}"
            )
        ln = lines[strategy]
        summary["detail"] = {
            "strategy": strategy,
            "date": ln.get("date"),
            "cash": ln.get("cash"),
            "value": ln.get("value"),
            "n_positions": ln.get("n_positions"),
            "buys": ln.get("buys"),
            "sells": ln.get("sells"),
            "holds": ln.get("holds"),
        }
        return json.dumps(summary, ensure_ascii=False, indent=1)

    # 回落到旧的单策略 22f 组合
    pfile = os.path.join(SIGNAL_DIR, "paper_holdings.json")
    if not os.path.exists(pfile):
        raise FileNotFoundError(
            "纸面组合不存在 (还没跑过 paper_portfolio.py 或 weekly_signal.py)"
        )
    with open(pfile) as fh:
        p = json.load(fh)
    cash = float(p.get("cash", 0.0))
    positions = p.get("positions", {})
    holdings, total_mv = [], 0.0
    for code, pos in positions.items():
        try:
            df = _load_parquet(code)
            close = float(df.iloc[-1]["close"])
        except FileNotFoundError:
            continue
        shares = float(pos.get("shares", 0))
        cost = float(pos.get("cost", 0.0))
        mv = shares * close
        total_mv += mv
        holdings.append({
            "code": code, "shares": int(shares), "close": close,
            "cost": cost, "market_value": round(mv, 2),
            "pnl_pct": round((close / cost - 1) * 100, 2) if cost else None,
            "buy_date": pos.get("buy_date", ""),
        })
    holdings.sort(key=lambda h: h["market_value"], reverse=True)
    total = cash + total_mv
    return json.dumps({
        "note": "旧版单策略 22f 组合（多策略盘尚未生成）",
        "cash": round(cash, 2),
        "market_value": round(total_mv, 2),
        "total_value": round(total, 2),
        "n_holdings": len(holdings),
        "position_pct": round(total_mv / total * 100, 1) if total else 0,
        "holdings": holdings,
    }, ensure_ascii=False, indent=1)


@mcp.tool()
def latest_signals() -> str:
    """最新一次 22f 周频信号快照 (买/卖/留, 只读, 不更新账本)。"""
    f = _latest_signal_file()
    if not f:
        raise FileNotFoundError("没有信号快照, 先跑 weekly_rebalance 或 daily_signal")
    with open(f) as fh:
        snap = json.load(fh)
    return json.dumps(snap, ensure_ascii=False, indent=1)


@mcp.tool()
def daily_signal() -> str:
    """跑 42f 日频信号 (honest_composite 口径, Top60 买/卖/留)。只读, 约 30-60s。"""
    stdout = _run_signal_script("daily_signal.py")
    # 找输出文件
    out_dir = os.path.join(DATA, "strategy_baseline")
    today = datetime.now().strftime("%Y-%m-%d")
    f = os.path.join(out_dir, f"daily_signal_{today}.json")
    if not os.path.exists(f):
        cands = [x for x in os.listdir(out_dir) if x.startswith("daily_signal_") and x.endswith(".json")]
        if not cands:
            raise RuntimeError(f"跑了但没找到输出文件, stdout 尾部: {stdout[-500:]}")
        f = os.path.join(out_dir, sorted(cands)[-1])
    with open(f) as fh:
        snap = json.load(fh)
    return json.dumps(snap, ensure_ascii=False, indent=1)


@mcp.tool()
def weekly_rebalance(topk: int = 30, ndrop: int = 3) -> str:
    """跑 22f 周频调仓 (买/卖/留 + 记账)。注意: 会更新纸面账本 (ledger.csv + paper_holdings.json)!"""
    _run_signal_script("weekly_signal.py", "--force", "--topk", str(topk), "--ndrop", str(ndrop))
    f = _latest_signal_file()
    with open(f) as fh:
        snap = json.load(fh)
    return json.dumps(snap, ensure_ascii=False, indent=1)


@mcp.tool()
def run_backtest(
    strategy: str = "honest_composite",
    codes: Optional[List[str]] = None,
    start_date: str = "2021-08-16",
    end_date: Optional[str] = None,
    initial_cash: float = 1_000_000.0,
    strategy_config: Optional[Dict[str, Any]] = None,
) -> str:
    """跑回测 (直驱 BacktestExecutor, 不走 API 的 1000 只上限)。

    strategy: honest_composite | factor_ranking | gbm_voltgt | moving_average | ...
    codes:    股票代码列表, 默认全市场 (5595 只)
    strategy_config: 如 {"topk": 60, "n_drop": 5}
    返回核心指标 (总收益/年化/夏普/最大回撤/交易次数) + 权益曲线采样点。
    """
    with _app_quiet():
        from app.services.backtest.execution.backtest_executor import BacktestExecutor
        from app.services.backtest.models.data_models import BacktestConfig

        if codes is None:
            codes = _list_universe()
        codes = [_norm_code(c) for c in codes]

        executor = BacktestExecutor(data_dir=DATA, enable_performance_profiling=False)
        cfg = BacktestConfig(initial_cash=initial_cash)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()

        report = asyncio.run(executor.run_backtest(
            strategy_name=strategy,
            stock_codes=codes,
            start_date=datetime.strptime(start_date, "%Y-%m-%d"),
            end_date=end_dt,
            strategy_config=strategy_config or {},
            backtest_config=cfg,
        ))

    pm = report.get("performance_metrics", report)
    metrics = {k: pm[k] for k in (
        "total_return", "annualized_return", "sharpe_ratio", "max_drawdown",
        "win_rate", "total_trades", "volatility", "sortino_ratio",
    ) if k in pm}
    # 权益曲线采样 (最多 30 点)
    eq = report.get("equity_curve") or []
    if isinstance(eq, list) and len(eq) > 30:
        step = len(eq) // 30
        eq = [eq[0]] + eq[::step] + [eq[-1]]
    return json.dumps({
        "strategy": strategy,
        "n_stocks": len(codes),
        "start": start_date, "end": str(end_dt)[:10],
        "initial_cash": initial_cash,
        "metrics": metrics,
        "equity_curve_sample": [
            {"date": str(e.get("date", e.get("index", "")))[:10],
             "value": e.get("value", e.get("close", e.get("equity")))}
            for e in eq
        ] if eq else [],
        "raw_keys": [k for k in report.keys()],
    }, ensure_ascii=False, indent=1, default=str)


if __name__ == "__main__":
    mcp.run()
