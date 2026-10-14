"""10 年风险平价研究 (方向 A): 2016-05-12 → 2026-05-08, 全市场 point-in-time 宇宙.

目的: 给风险平价的 DSR 终审 (3 年 12/12 fail, SR0=1.56 撑不住) 提供 10 年 T≈2400 观测,
覆盖 2015 股灾/2018 熊市/2019-21 结构牛/2022 熊市/2024 行情, 一锤定音 edge 真伪。

数据管线 (纯 monkeypatch, 零引擎改动):
  DataLoader.load_stock_data 单股链路 → stock_data 原始 parquet (前复权, 1991 年起)
    探针结论: stock_data ≡ qlib_data 同源同价(比例恒 1.0000); 前复权(送转台阶命中 5%);
    qlib 段(2021+) 覆盖 <70% → 原 loader 拒收 → 回退 stock_data 全 10 年 (设计意图内)
  清洗: close 跳变 |r|>1.2×涨跌停制度 → 当日 NaN → 前值填充 (拼接点假跳变,
    密度 10 年 0.06%/天; 两窗口同一清洗保可比)
  宇宙 (point-in-time, 固定): 窗口中点(2021-05-09)前已上市 且 窗口末(2026-05-08)前退市
    的股票。中点前上市 = 排除 2021+ 注册制新股潮(10 年宇宙会随 IPO 膨胀失真);
    中点前已退市 = 10 年窗口内有真实交易段(自然 point-in-time, 无未来函数)。
    停牌/数据缺失 → 引擎既有停牌处理路径。

三腿: stochastic / factor_ranking / bollinger (同 3 年实验, 指标全 OHLCV 自算)。
产物: data/strategy_baseline/risk_parity_10y_experiment.json
"""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BACKEND_ROOT)

_spec = importlib.util.spec_from_file_location(
    "rsb", os.path.join(BACKEND_ROOT, "scripts", "run_strategy_baseline.py")
)
assert _spec is not None
rsb = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(rsb)

from app.core.config import settings  # noqa: E402
from app.services.backtest import BacktestExecutor  # noqa: E402
from app.services.backtest.models import BacktestConfig  # noqa: E402
from app.services.backtest.core import portfolio_manager_array as pma_module  # noqa: E402
from app.services.backtest.execution import data_loader as dl_module  # noqa: E402

DATA_ROOT = "/Users/rone/projects/stock-platform/data"
STOCK_DATA_DIR = os.path.join(DATA_ROOT, "parquet", "stock_data")
OUT_JSON = os.path.join(DATA_ROOT, "strategy_baseline", "risk_parity_10y_experiment.json")

START = datetime(2016, 5, 12)
END = datetime(2026, 5, 8)
MIDPOINT = datetime(2021, 5, 9)  # point-in-time 宇宙分界: 此前上市
LEGS = ["stochastic", "factor_ranking", "bollinger"]
NAN_FILL_NOTE = "cleaned"  # 清洗标记(记录用)


def _limit_at(stock_code: str, dates: pd.DatetimeIndex) -> pd.Series:
    """时段化涨跌停制度（向量化, 按日期分段）.
    创业板 30x: 2020-08-24 前 ±10%, 后 ±20% (注册制);
    科创板 68x: 2019-07 开板即 ±20%; 主板 60/00x ±10%; 北交 92/83/87 ±30%。
    ST 股 ±5% 无法从 OHLCV 判状态, 用板块制度上限 (已知局限)。"""
    c2 = stock_code[:2]
    if c2 == "30":
        lim = pd.Series(0.10, index=dates)
        lim = lim.mask(dates >= pd.Timestamp("2020-08-24"), 0.20)
        return lim
    if c2 == "68":
        return pd.Series(0.20, index=dates)
    if c2 in ("92", "83", "87"):
        return pd.Series(0.30, index=dates)
    return pd.Series(0.10, index=dates)


def _clean_price_jumps(df: pd.DataFrame, stock_code: str,
                       window_start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """close 超 1.2×时段化涨跌停制度 → 当日 NaN → 前值填充; 联动 open/high/low 同向缩放.
    拼接点/复权基准漂移的物理特征。创业板按 2020-08-24 分段, 清掉 2016-2022 的 12-24% 假跳变。
    首行守卫: 数据首行晚于窗口起点(中途上市) → 首行收益置 0 (上市价无前值可比)。"""
    close = df["close"]
    prev = close.shift(1)
    r = close / prev - 1.0
    lim = _limit_at(stock_code, df.index) * 1.2
    bad = (r.abs() > lim).fillna(False)
    if window_start is not None and len(df) and pd.Timestamp(df.index[0]) > window_start:
        bad.iloc[0] = True
    if not bad.any():
        df.attrs["_cleaned_jumps"] = 0
        return df
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        vals = out[col].where(~bad)
        out[col] = vals.ffill()
    out["volume"] = out["volume"].where(~bad, 0)
    out.attrs["_cleaned_jumps"] = int(bad.sum())
    return out


def _raw_stock_data(stock_code: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """stock_data 原始 parquet → 单股 DataFrame(date index, 真实交易行, 不 reindex).

    不做 bdate 重索引: 引擎交易日历 = 宇宙股票索引日期的并集, reindex 注入的假日
    会在无真实交易日期上制造幽灵交易日 (引擎按该日执行交易)。停牌日在原始数据中
    天然缺席 → 引擎自然停该股。拼接点假跳变清洗在调用方。"""
    safe = stock_code.replace(".", "_")
    f = os.path.join(STOCK_DATA_DIR, f"{safe}.parquet")
    x = pd.read_parquet(f, columns=["date", "open", "high", "low", "close", "volume"])
    x["date"] = pd.to_datetime(x["date"])
    x = x.set_index("date").sort_index()
    x = x.loc[start_date:end_date]
    if x.empty:
        return x
    x = x[~x.index.duplicated(keep="first")]
    x.attrs["stock_code"] = stock_code
    x.attrs["from_precomputed"] = False
    # 预计算指标列 (与 data_loader._load_from_parquet_and_calculate 同口径,
    # 供 bollinger 的 MA/STD 优先复用路径; 段内值 = qlib 段同式计算)
    close = x["close"]
    for p in (20, 50, 60):
        if f"MA{p}" not in x.columns:
            x[f"MA{p}"] = close.rolling(window=p).mean()
    for p in (20, 60):
        if f"STD{p}" not in x.columns:
            x[f"STD{p}"] = close.rolling(window=p).std()
    if "RSI14" not in x.columns:
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        x["RSI14"] = 100 - (100 / (1 + rs))
    x.attrs["from_precomputed"] = False
    return x


_orig_load_single = dl_module.DataLoader.load_stock_data


def _covers_window(df: pd.DataFrame, start_date: datetime, end_date: datetime) -> bool:
    """数据是否覆盖 [start, end] 全窗口: 首日期距 start ≤7 天 且 行数足够."""
    idx = df.index
    if len(idx) < 20:
        return False
    first = pd.Timestamp(idx[0])
    if (first - pd.Timestamp(start_date)).days > 7:
        return False
    expected = max(1, (pd.bdate_range(pd.Timestamp(start_date).date(),
                                       pd.Timestamp(end_date).date())).size)
    return len(idx) / expected >= 0.7


def _patched_load_single(self, stock_code: str, start_date: datetime,
                         end_date: datetime,
                         required_columns=None) -> pd.DataFrame:
    """回退链补丁: 先原链路(qlib 段); 未覆盖全窗口(2016-2021 缺口) → stock_data 全历史.
    两源统一跳变清洗 → 3年/10年口径一致。"""
    try:
        df = _orig_load_single(self, stock_code, start_date, end_date,
                               required_columns=required_columns)
    except Exception:
        df = None
    if df is None or df.empty or not _covers_window(df, start_date, end_date):
        df = _raw_stock_data(stock_code, start_date, end_date)
    if df is None or df.empty:
        return pd.DataFrame()
    df = _clean_price_jumps(df, stock_code, window_start=pd.Timestamp(start_date))
    return df


def patch_loader() -> None:
    dl_module.DataLoader.load_stock_data = _patched_load_single


def list_universe() -> list:
    """point-in-time 固定宇宙: 中点(2021-05-09)前已上市 且 中点时点仍在市.
    用 stock_data 每只的首末日期判断 (首日期 ≤ MIDPOINT 上市; 末日 ≥ MIDPOINT 未退市)。"""
    codes = []
    files = sorted(os.listdir(STOCK_DATA_DIR))
    mid = pd.Timestamp(MIDPOINT)
    for f in files:
        if not f.endswith(".parquet"):
            continue
        code = f[:-8].replace("_", ".")
        try:
            x = pd.read_parquet(os.path.join(STOCK_DATA_DIR, f), columns=["date"])
        except Exception:
            continue
        d = pd.to_datetime(x["date"])
        if d.min() <= mid and d.max() >= mid:
            codes.append(code)
    return sorted(codes)


_orig_snapshot = pma_module.PortfolioManagerArray.record_portfolio_snapshot


def install_stash(stash: dict) -> None:
    def patched(self, date, current_prices):
        stash["pm"] = self
        _orig_snapshot(self, date, current_prices)

    pma_module.PortfolioManagerArray.record_portfolio_snapshot = patched


def uninstall() -> None:
    pma_module.PortfolioManagerArray.record_portfolio_snapshot = _orig_snapshot


def run_leg(ex, strat: str, codes: list, cfg) -> list:
    stash: dict = {}
    install_stash(stash)
    try:
        # factor_ranking 用官方 config(0.019 仓位持 ~50 只, 每日换 5 只 = 真实意图);
        # 全局 0.2 仓位只允许 5 只 → TopK 退化成冷启动买入持有(8 笔/10 年), 腿无效。
        # stochastic/bollinger 保持全局 config (与 3 年研究同口径)。
        cfg_use = cfg
        if strat == "factor_ranking":
            cfg_use = BacktestConfig(
                initial_cash=1_000_000.0,
                commission_rate=0.0003,
                slippage_rate=0.0001,
                max_position_size=0.019,
                cash_reserve_ratio=0.05,
                board_lot_size=100,
                stop_loss_pct=0.0,
                take_profit_pct=0.0,
                rebalance_frequency="daily",
            )
        asyncio.run(
            ex.run_backtest(
                strategy_name=strat,
                stock_codes=codes,
                start_date=START,
                end_date=END,
                strategy_config={},
                backtest_config=cfg_use,
            )
        )
        pm = stash.get("pm")
        eq = getattr(pm, "equity_curve", None) if pm else None
        assert eq, f"{strat}: no equity curve captured"
        return eq
    finally:
        uninstall()


def _align(equities: list) -> tuple:
    maps = {}
    for s, eq in zip(LEGS, equities):
        m = {}
        for d, v in eq:
            dd = d.date() if hasattr(d, "date") else d
            m[dd] = float(v)
        maps[s] = m
    dates = sorted(set.intersection(*[set(m) for m in maps.values()]))
    if not dates:
        raise RuntimeError("leg equity curves share no dates")
    vals = {}
    for s, m in maps.items():
        arr = np.array([m.get(d, np.nan) for d in dates], dtype=float)
        arr = pd.Series(arr).ffill().to_numpy()
        vals[s] = arr
    return [pd.Timestamp(d) for d in dates], vals


def _ret_stats_from_returns(r: np.ndarray, init=1_000_000.0):
    eq = init * np.cumprod(1.0 + r)
    vals = np.concatenate([[init], eq])
    base = np.where(vals[:-1] > 0, vals[:-1], 1.0)
    rets = np.diff(vals) / base
    years = max(1.0, len(rets) / 252.0)
    total = vals[-1] / vals[0] - 1.0
    ann = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    vol = float(rets.std() * np.sqrt(252))
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(vals)
    mdd = float(((vals - peak) / peak).min())
    return {
        "total_return": round(total, 6),
        "annualized_return": round(ann, 6),
        "volatility": round(vol, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(mdd, 6),
    }


def _yearly_from_returns(r: np.ndarray, dates: list) -> dict:
    """上年末 → 本年末 净值端点口径 (与 DSR 终审同一方法; 首年基准=1.0; 累乘严格=总收益)."""
    by: dict = {}
    cum = 1.0
    prev_end = 1.0
    last_year = None
    for d, ri in zip(dates, r):
        y = int(d.year) if hasattr(d, "year") else int(str(d)[:4])
        if last_year is not None and y != last_year:
            by[last_year] = cum / prev_end - 1.0
            prev_end = cum
        cum *= 1.0 + ri
        last_year = y
    if last_year is not None:
        by[last_year] = cum / prev_end - 1.0
    return {str(y): round(v, 4) for y, v in by.items()}


def risk_parity_weights(vals: dict, dates: list, lookback: int, rebal: int) -> np.ndarray:
    names = LEGS
    n = len(dates)
    rets = {s: np.diff(vals[s]) / np.where(vals[s][:-1] > 0, vals[s][:-1], 1.0) for s in names}
    w = np.zeros((n, len(names)))
    last_w = np.ones(len(names)) / len(names)
    for t in range(n):
        if t > 0 and (t % rebal == 0):
            invs = []
            for s in names:
                window = rets[s][max(0, t - lookback): t - 1]  # ≤t-1 收益, 无未来函数
                v = float(np.std(window) * np.sqrt(252)) if len(window) >= 20 else 1.0
                invs.append(1.0 / max(v, 1e-4))
            ssum = sum(invs)
            last_w = np.array(invs) / ssum
        w[t] = last_w
    return w


def combine(vals: dict, dates: list, w: np.ndarray) -> np.ndarray:
    names = LEGS
    n = len(dates)
    r = np.zeros(n)
    for i, s in enumerate(names):
        leg_r = np.zeros(n)
        leg_r[1:] = np.diff(vals[s]) / np.where(vals[s][:-1] > 0, vals[s][:-1], 1.0)
        r += leg_r * w[:, i]
    return r


def main() -> None:
    patch_loader()
    codes = list_universe()
    print(f"[universe] point-in-time 固定宇宙 {len(codes)} 只 (中点 {MIDPOINT.date()} 在市)",
          flush=True)
    ex = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True,
        max_workers=8,
        enable_performance_profiling=False,
        use_multiprocessing=False,
    )
    cfg = BacktestConfig(
        initial_cash=1_000_000.0,
        commission_rate=0.0003,
        slippage_rate=0.0001,
        max_position_size=0.2,
        cash_reserve_ratio=0.05,
        board_lot_size=100,
        stop_loss_pct=0.05,
        take_profit_pct=0.15,
        rebalance_frequency="daily",
        open_cost=0.0,
        close_cost=0.0,
        min_cost=0.0,
    )

    leg_eqs = {}
    for s in LEGS:
        t0 = time.perf_counter()
        leg_eqs[s] = run_leg(ex, s, codes, cfg)
        print(f"[leg] {s}: n={len(leg_eqs[s])} wall={time.perf_counter()-t0:.0f}s", flush=True)

    dates, vals = _align([leg_eqs[s] for s in LEGS])
    print(f"[align] 共同交易日 {len(dates)}: {dates[0].date()} → {dates[-1].date()}", flush=True)

    results: dict = {}

    def save():
        payload = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "window": [str(START), str(END)],
                "universe": f"point-in-time-fixed-{len(codes)}",
                "midpoint": str(MIDPOINT),
                "legs": LEGS,
                "note": "10年风险平价研究: stock_data前复权管线+跳变清洗; 纯组合层, 无未来函数",
            },
            "results": results,
        }
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1, default=str)

    # R0 静态等权
    w0 = np.ones((len(dates), 3)) / 3
    r0 = combine(vals, dates, w0)
    rs0 = _ret_stats_from_returns(r0)
    results["R0_等权13"] = {
        "stats": rs0,
        "yearly": _yearly_from_returns(r0, dates),
        "returns": [float(x) for x in r0],
        "dates": [d.isoformat() for d in dates],
    }
    print(f"[R0] 等权: sharpe={rs0['sharpe_ratio']} ann={rs0['annualized_return']} "
          f"mdd={rs0['max_drawdown']} vol={rs0['volatility']}", flush=True)
    save()

    for tag, lb, rb in (("R1_风险平价月度60d", 60, 21), ("R2_风险平价月度20d", 20, 21)):
        w = risk_parity_weights(vals, dates, lookback=lb, rebal=rb)
        rc = combine(vals, dates, w)
        rsc = _ret_stats_from_returns(rc)
        avg_w = {s: float(w[int(len(w) * 0.1):, i].mean()) for i, s in enumerate(LEGS)}
        results[tag] = {
            "lookback": lb, "rebal_days": rb, "stats": rsc,
            "avg_weights": avg_w, "yearly": _yearly_from_returns(rc, dates),
            "returns": [float(x) for x in rc],
            "dates": [d.isoformat() for d in dates],
        }
        print(f"[{tag}] sharpe={rsc['sharpe_ratio']} ann={rsc['annualized_return']} "
              f"mdd={rsc['max_drawdown']} avg_w={ {k: round(v,3) for k,v in avg_w.items()} }",
              flush=True)
        save()

    for s in ("stochastic", "factor_ranking", "bollinger"):
        arr = vals[s]
        lr = np.zeros(len(arr))
        lr[1:] = np.diff(arr) / np.where(arr[:-1] > 0, arr[:-1], 1.0)
        rsc = _ret_stats_from_returns(lr)
        results[f"C_{s}单腿"] = {
            "stats": rsc,
            "yearly": _yearly_from_returns(lr, dates),
            "returns": [float(x) for x in lr],
            "dates": [d.isoformat() for d in dates],
        }
        print(f"[C_{s}] sharpe={rsc['sharpe_ratio']} ann={rsc['annualized_return']} "
              f"mdd={rsc['max_drawdown']}", flush=True)
        save()

    print(f"[saved] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
