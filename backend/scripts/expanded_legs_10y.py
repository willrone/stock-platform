#!/usr/bin/env python3
"""扩腿实验 (10 年): 3 腿池 → 7 腿池, 检验"等权=池内天花板"结论在更大池子的边界。

背景: 方向 D 证伪组合层条件配权(3 腿池)。最后一个组合层问题:
  3 腿质量接近(0.47/0.51/0.12)时等权是天花板; 扩腿(加相关性不同的技术腿)能否抬升天花板?
腿选择(全市场 4329 宇宙, 3 年基线榜夏普序, 仅 OHLCV 可用):
  旧 3 腿: factor_ranking 0.98 / stochastic 0.91 / bollinger 0.75 (与新 4 腿同源重跑;
    已发布 10 年 JSON 含引擎估值毛刺(持仓停牌日漏计 → 假崩假复, stoch 25/boll 31 个 >24% 日), 不可混用)
  新增 4 腿: mean_reversion 0.83 / momentum_factor 0.77 / cci 0.72 / obv 0.55
  排除: value_factor(依赖 pe/pb 基本面, stock_data 仅 OHLCV) /
        pairs·cointegration(O(N²) find_pairs, 3 年基线即 500 宇宙限定) /
        multi_factor(0.57, 与 value 同域) / low_volatility(3 年仅 8 笔=buy-hold, 退化) / ML 4 个(坏 pkl)
组合(全部静态等权; 纯组合层):
  E3  3 腿等权 (sto/fr/boll, = 10 年 R0 锚)
  E5  5 腿等权 (3 旧 + mean_reversion + momentum, 按 3 年夏普取新腿前二)
  E7  7 腿等权 (5 腿 + cci + obv)
  另算 7 腿日收益相关矩阵 (池内分散度的直接度量; 若新腿与 fr 高相关 → 分散增益有限)
裁决: E7 10 年夏普 vs E3 0.638; DSR; 分年/分段; PBO(E3/E5/E7)。
管线: 完全克隆 risk_parity_10y_experiment (stock_data 前复权 + 时段化涨跌停清洗 + 4329 宇宙 +
8 线程确定性) + 估值修复 monkey-patch: 持仓停牌日沿用上次已知价, 消除引擎漏计持仓的假崩/假复。
产物: data/strategy_baseline/expanded_legs_10y.json
"""
from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime

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
OUT_JSON = os.path.join(DATA_ROOT, "strategy_baseline", "expanded_legs_10y.json")
RP10_JSON = os.path.join(DATA_ROOT, "strategy_baseline", "risk_parity_10y_experiment.json")

START = datetime(2016, 5, 12)
END = datetime(2026, 5, 8)
MIDPOINT = datetime(2021, 5, 9)

OLD_LEGS = ["stochastic", "factor_ranking", "bollinger"]   # 旧 3 腿(与新 4 腿同源重跑)
NEW_LEGS = ["mean_reversion", "momentum_factor", "cci", "obv"]  # 新增(OHLCV 技术腿, 3 年夏普序)
LEGS = OLD_LEGS + NEW_LEGS
SQRT252 = math.sqrt(252.0)
N_TRIALS = 40


# ---------------- 数据层: 逐位克隆 10 年管线 ----------------
def _limit_at(stock_code: str, dates: pd.DatetimeIndex) -> pd.Series:
    c2 = stock_code[:2]
    if c2 == "30":
        lim = pd.Series(0.10, index=dates)
        return lim.mask(dates >= pd.Timestamp("2020-08-24"), 0.20)
    if c2 == "68":
        return pd.Series(0.20, index=dates)
    if c2 in ("92", "83", "87"):
        return pd.Series(0.30, index=dates)
    return pd.Series(0.10, index=dates)


def _clean_price_jumps(df: pd.DataFrame, stock_code: str,
                       window_start: pd.Timestamp | None = None) -> pd.DataFrame:
    close = df["close"]
    prev = close.shift(1)
    r = close / prev - 1.0
    lim = _limit_at(stock_code, df.index) * 1.2
    bad = (r.abs() > lim).fillna(False)
    if window_start is not None and len(df) and pd.Timestamp(df.index[0]) > window_start:
        bad.iloc[0] = True
    if not bad.any():
        return df
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col].where(~bad).ffill()
    out["volume"] = out["volume"].where(~bad, 0)
    return out


def _raw_stock_data(stock_code: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
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
    return x


_orig_load_single = dl_module.DataLoader.load_stock_data


def _covers_window(df: pd.DataFrame, start_date: datetime, end_date: datetime) -> bool:
    idx = df.index
    if len(idx) < 20:
        return False
    if (pd.Timestamp(idx[0]) - pd.Timestamp(start_date)).days > 7:
        return False
    expected = max(1, pd.bdate_range(pd.Timestamp(start_date).date(),
                                     pd.Timestamp(end_date).date()).size)
    return len(idx) / expected >= 0.7


def _patched_load_single(self, stock_code: str, start_date: datetime,
                         end_date: datetime, required_columns=None) -> pd.DataFrame:
    try:
        df = _orig_load_single(self, stock_code, start_date, end_date,
                               required_columns=required_columns)
    except Exception:
        df = None
    if df is None or df.empty or not _covers_window(df, start_date, end_date):
        df = _raw_stock_data(stock_code, start_date, end_date)
    if df is None or df.empty:
        return pd.DataFrame()
    return _clean_price_jumps(df, stock_code, window_start=pd.Timestamp(start_date))


def patch_loader() -> None:
    dl_module.DataLoader.load_stock_data = _patched_load_single


def list_universe() -> list:
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
_last_px: dict = {}  # code → 最近已知有效价 (估值修复)


def install_stash(stash: dict) -> None:
    _last_px.clear()  # 每次腿运行开始时重置价格缓存

    def patched(self, date, current_prices):
        # 估值修复: 持仓股当日无有效价(停牌/数据缺行)时引擎 get_portfolio_value
        # 会漏计该持仓 → 净值假崩次日假复(单腿 ±79%/+369%, 24% 阈值以上毛刺)。
        # 沿用上次已知价 → 假复日恢复为真实涨跌停内幅度, 消除毛刺。
        # 引擎为 array 化组合: 持仓在 self.quantities[idx] (np.ndarray) + self.code_to_idx。
        # 交易不受影响: 快照在日循环末(交易之后), 且次日 current_prices 重建。
        for code, px in current_prices.items():
            if px == px:  # 非 NaN
                _last_px[code] = px
        c2i = getattr(self, "code_to_idx", None)
        qtys = getattr(self, "quantities", None)
        if c2i is not None and qtys is not None:
            for code, idx in c2i.items():
                if int(qtys[idx]) > 0 and code not in current_prices and _last_px.get(code):
                    current_prices[code] = _last_px[code]
        stash["pm"] = self
        _orig_snapshot(self, date, current_prices)

    pma_module.PortfolioManagerArray.record_portfolio_snapshot = patched


def uninstall() -> None:
    pma_module.PortfolioManagerArray.record_portfolio_snapshot = _orig_snapshot


def run_leg(ex, strat: str, codes: list, cfg) -> list:
    stash: dict = {}
    install_stash(stash)
    try:
        cfg_use = cfg
        if strat == "factor_ranking":
            cfg_use = BacktestConfig(
                initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.0001,
                max_position_size=0.019, cash_reserve_ratio=0.05, board_lot_size=100,
                stop_loss_pct=0.0, take_profit_pct=0.0, rebalance_frequency="daily",
            )
        asyncio.run(ex.run_backtest(
            strategy_name=strat, stock_codes=codes, start_date=START, end_date=END,
            strategy_config={}, backtest_config=cfg_use))
        pm = stash.get("pm")
        eq = getattr(pm, "equity_curve", None) if pm else None
        assert eq, f"{strat}: no equity curve captured"
        return eq
    finally:
        uninstall()


# ---------------- 组合层 ----------------
def _ret_stats(r: np.ndarray, init=1_000_000.0) -> dict:
    eq = init * np.cumprod(1.0 + r)
    vals = np.concatenate([[init], eq])
    rets = np.diff(vals) / np.where(vals[:-1] > 0, vals[:-1], 1.0)
    years = max(1.0, len(rets) / 252.0)
    total = vals[-1] / vals[0] - 1.0
    ann = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    vol = float(rets.std() * SQRT252)
    sharpe = float(rets.mean() / rets.std() * SQRT252) if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(vals)
    mdd = float(((vals - peak) / peak).min())
    return {
        "total_return": round(total, 6), "annualized_return": round(ann, 6),
        "volatility": round(vol, 6), "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(mdd, 6),
    }


def _clean_ret_spikes(r: np.ndarray) -> np.ndarray:
    """日收益毛刺计数/清洗(已弃用 — 不进入 stats): |r|>24%。

    修复前引擎会在停牌/缺价日漏计持仓 → 净值假崩次日假复(±79%/+369%);
    但"单日归零"会破坏 crash-recovery 配对(跌-23% 保留、+30% 恢复被清 → 凭空
    -23% 永久亏损), 故在估值源头修(见 install_stash), 这里仅统计验证。"""
    r = np.asarray(r, dtype=float)
    return r[np.abs(r) > 0.24]


def _neutralize_spikes(r: np.ndarray, thr: float = 0.24) -> tuple:
    """组合层毛刺中和(估值修复的兜底): "假高1日 + 次日回落"对 → 假高日 equity 压回前日值。

    引擎停牌估值偶漏(如并行路径矩阵构造竞态)会产生假高+假复对(如 +91%/-49%,
    2日净≈0, 数学上单日 +179% 也非真实行情)。中和方式: 假高日 equity 压回前日
    (该日 r→0), 次日 r 吸收真实变动 → 总收益逐位不变, 只移除假波动。
    检测: r_i>thr 且 次日回落(2日净<10% 或 r_{i+1}<-thr/2)。"""
    r = np.asarray(r, dtype=float).copy()
    eq = np.cumprod(1.0 + r)
    flag = np.zeros(len(r), dtype=bool)
    for i in np.where(r > thr)[0]:
        j = i + 1
        if j < len(r):
            net = (1.0 + r[i]) * (1.0 + r[j]) - 1.0
            if net < 0.10 or r[j] < -thr * 0.5:
                flag[i] = True
    if not flag.any():
        return r, 0
    eq_c = eq.copy()
    for i in np.where(flag)[0]:
        eq_c[i] = eq[i - 1] if i > 0 else eq[i]
    r_c = np.concatenate([[r[0]], np.diff(eq_c) / eq_c[:-1]])
    return r_c, int(flag.sum())


def _yearly(r: np.ndarray, dates: list) -> dict:
    by: dict = {}
    cum = 1.0
    prev_end = 1.0
    last_year = None
    for d, ri in zip(dates, r):
        y = int(str(d)[:4])
        if last_year is not None and y != last_year:
            by[last_year] = cum / prev_end - 1.0
            prev_end = cum
        cum *= 1.0 + ri
        last_year = y
    if last_year is not None:
        by[last_year] = cum / prev_end - 1.0
    return {str(y): round(v, 4) for y, v in by.items()}


def _subperiod(r: np.ndarray, dates: list, lo: str, hi: str) -> dict:
    dt = np.array(dates, dtype="datetime64[D]")
    m = (dt >= np.datetime64(lo)) & (dt < np.datetime64(hi))
    rs = r[m]
    eq = np.cumprod(1.0 + rs)
    tot = eq[-1] / eq[0] - 1.0
    yrs = max(len(rs) / 252.0, 1e-9)
    mu, sd = rs[1:].mean(), rs[1:].std(ddof=1)
    return {"days": int(m.sum()), "total": round(tot, 4),
            "ann": round((1 + tot) ** (1 / yrs) - 1 if tot > -1 else -1.0, 4),
            "sharpe": round(mu / sd * SQRT252 if sd > 0 else 0.0, 4)}


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def deflated_sharpe(daily: np.ndarray, n_trials: int = N_TRIALS) -> dict:
    r = np.asarray(daily, dtype=float)[1:]
    T = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    sr_d = mu / sd if sd > 0 else 0.0
    x = r - r.mean()
    m2 = float((x ** 2).mean())
    m3 = float((x ** 3).mean())
    m4 = float((x ** 4).mean())
    skew = m3 / m2 ** 1.5 if m2 > 0 else 0.0
    kurt = m4 / m2 ** 2 if m2 > 0 else 3.0
    var_sr = max(1e-12, (1 - skew * sr_d + (kurt - 1) / 4 * sr_d ** 2) / T)
    sd_sr = math.sqrt(var_sr)
    sr0 = sd_sr * math.sqrt(2.0 * math.log(max(2, n_trials)))
    dsr = norm_cdf((sr_d - sr0) / sd_sr)
    return {"sharpe_annualized": round(sr_d * SQRT252, 4), "n_obs": T, "skew": round(skew, 4),
            "kurt": round(kurt, 4), "sr0_threshold_ann": round(sr0 * SQRT252, 4),
            "deflated_sharpe": round(dsr, 5), "pass_95": bool(dsr > 0.95)}


def pbo_cscv(rets: dict, n_splits: int = 200, seed: int = 7) -> dict:
    names = list(rets)
    R = np.stack([np.asarray(rets[n], dtype=float)[1:] for n in names], axis=1)
    T, V = R.shape
    acc = 0.0
    rng = np.random.default_rng(seed)
    for _ in range(n_splits):
        idx = rng.permutation(T)
        A, B = R[idx[: T // 2]], R[idx[T // 2:]]
        sa, sb = A.std(0, ddof=1), B.std(0, ddof=1)
        srA = A.mean(0) / np.where(sa > 0, sa, 1.0) * SQRT252
        srB = B.mean(0) / np.where(sb > 0, sb, 1.0) * SQRT252
        best = int(np.argmax(srA))
        acc += (1 + int((srB > srB[best]).sum())) / V
    return {"pbo": round(acc / n_splits, 4), "n_variants": V, "variants": names}


def main() -> None:
    patch_loader()
    codes = list_universe()
    print(f"[universe] point-in-time 固定宇宙 {len(codes)} 只", flush=True)
    ex = BacktestExecutor(
        data_dir=str(settings.DATA_ROOT_PATH),
        enable_parallel=True, max_workers=8,
        enable_performance_profiling=False, use_multiprocessing=False,
    )
    cfg = BacktestConfig(
        initial_cash=1_000_000.0, commission_rate=0.0003, slippage_rate=0.0001,
        max_position_size=0.2, cash_reserve_ratio=0.05, board_lot_size=100,
        stop_loss_pct=0.05, take_profit_pct=0.15, rebalance_frequency="daily",
        open_cost=0.0, close_cost=0.0, min_cost=0.0,
    )

    # 1) 日期轴: 沿用已发布 10 年 JSON 的 2424 天 (7 腿在同轴上对齐)
    with open(RP10_JSON) as f:
        rp10 = json.load(f)
    dates = rp10["results"]["R0_等权13"]["dates"]
    n = len(dates)
    leg_ret: dict = {}
    print(f"[align] 10 年窗口 {n} 天: {dates[0][:10]} → {dates[-1][:10]}", flush=True)

    # 2) 7 腿全部同源重跑 (含旧 3 腿 — 估值修复后与含毛刺的已发布 returns 不可混用)
    leg_eq = {}
    for s in LEGS:
        t0 = time.perf_counter()
        leg_eq[s] = run_leg(ex, s, codes, cfg)
        print(f"[leg] {s}: n={len(leg_eq[s])} wall={time.perf_counter()-t0:.0f}s", flush=True)

    # 3) 对齐到 10 年共同交易日 (两侧统一为 date 对象: 10y JSON 是 ISO 字符串, 新腿是 date)
    ten_dates = {pd.Timestamp(s).date() for s in dates}  # ISO 字符串 → date
    new_date_set = set()
    for s, eq in leg_eq.items():
        new_date_set |= {d.date() if hasattr(d, "date") else d for d, _ in eq}
    common = sorted(ten_dates & new_date_set)
    cov = len(common) / n
    print(f"[align] 新腿覆盖 {len(common)}/{n} 天 ({cov*100:.1f}%)", flush=True)
    if cov < 0.8:
        raise RuntimeError(f"新腿日期覆盖不足 {cov:.0%}, 检查日历/数据")
    # 新腿收益对齐到 10 年日期轴 (停牌日 ffill, 与 10 年管线 _align 同口径)
    for s, eq in leg_eq.items():
        m = {d.date() if hasattr(d, "date") else d: float(v) for d, v in eq}
        arr = np.array([m.get(pd.Timestamp(d).date(), np.nan) for d in dates], dtype=float)
        arr = pd.Series(arr).ffill().to_numpy()
        leg_ret[s] = np.zeros(n)
        leg_ret[s][1:] = np.diff(arr) / np.where(arr[:-1] > 0, arr[:-1], 1.0)

    # 3.5) 毛刺中和(估值修复兜底): raw 计数留档, 中和后进入组合/统计
    spike_raw: dict = {}
    for s in LEGS:
        spike_raw[s] = int((np.abs(leg_ret[s]) > 0.24).sum())
        leg_ret[s], nfix = _neutralize_spikes(leg_ret[s])
        if spike_raw[s] or nfix:
            print(f"[spikes] {s}: raw={spike_raw[s]} → 中和 {nfix} 对", flush=True)

    # 4) 组合: E3/E5/E7 等权
    port: dict = {}
    R = np.stack([leg_ret[s] for s in LEGS], axis=1)  # (n, 7)
    for tag, sel in (("E3_3腿等权", OLD_LEGS), ("E5_5腿等权", OLD_LEGS[:3] + ["mean_reversion", "momentum_factor"]),
                     ("E7_7腿等权", LEGS)):
        idx = [LEGS.index(s) for s in sel]
        r = np.zeros(n)
        for i in idx:
            r += R[:, i] / len(sel)
        port[tag] = r

    out = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "window": [str(START), str(END)], "universe": f"point-in-time-fixed-{len(codes)}",
            "legs": LEGS,
            "note": "扩腿实验: 3腿→7腿等权, 全部7腿同源重跑(估值修复 monkey-patch 生效); 纯组合层; 8线程确定性; 与已发布10年JSON的旧3腿returns不可混用(其含停牌估值毛刺)",
        },
        "legs": {}, "combos": {}, "returns": {}, "dsr": {}, "pbo": {},
        "correlation": {a: {b: round(float(np.corrcoef(leg_ret[a][1:], leg_ret[b][1:])[0, 1]), 3)
                            for b in LEGS if b != a} for a in LEGS},
    }

    def save():
        with open(OUT_JSON, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=1, default=str)

    for s in LEGS:
        r = leg_ret[s]
        out["legs"][s] = {
            "stats": _ret_stats(r), "yearly": _yearly(r, dates),
            "sub_2016_23": _subperiod(r, dates, "2016-05-12", "2023-05-12"),
            "sub_2023_26": _subperiod(r, dates, "2023-05-12", "2026-05-09"),
        }
        out["returns"][s] = [float(x) for x in r]
        out["legs"][s]["spikes_detected"] = int((np.abs(r) > 0.24).sum())
        st = out["legs"][s]["stats"]
        print(f"[leg] {s}: sharpe={st['sharpe_ratio']} ann={st['annualized_return']*100:+.1f}% "
              f"mdd={st['max_drawdown']*100:.0f}% spikes={out['legs'][s]['spikes_detected']}", flush=True)
    # sanity: 新腿 3 年子段 vs 基线 3 年 (5513/qlib 官方口径) 粗参照 — 宇宙/数据源不同, 宽松容差 0.3
    m3 = (np.array(dates, dtype="datetime64[D]") >= np.datetime64("2023-05-12")) & \
         (np.array(dates, dtype="datetime64[D]") < np.datetime64("2026-05-09"))
    baseline3y = {"mean_reversion": 0.83, "momentum_factor": 0.77, "cci": 0.72, "obv": 0.55,
                  "stochastic": 0.91, "factor_ranking": 0.98, "bollinger": 0.75}
    for s in LEGS:
        a3 = np.asarray(leg_ret[s][m3])
        mu, sd = a3[1:].mean(), a3[1:].std(ddof=1)
        s3 = float(mu / sd * SQRT252) if sd > 0 else 0.0
        ref = baseline3y.get(s)
        tag = "OK" if ref is None or abs(s3 - ref) < 0.3 else "MISMATCH"
        print(f"[sanity] {s:<18} 3年(4329宇宙) sharpe={s3:+.3f}  基线参照(5513)={ref}  {tag}", flush=True)
    save()

    for tag, r in port.items():
        out["combos"][tag] = {
            "stats": _ret_stats(r), "yearly": _yearly(r, dates),
            "sub_2016_23": _subperiod(r, dates, "2016-05-12", "2023-05-12"),
            "sub_2023_26": _subperiod(r, dates, "2023-05-12", "2026-05-09"),
            "spikes_detected": int((np.abs(r) > 0.24).sum()),
            "returns": [float(x) for x in r],
        }
        out["returns"][tag] = [float(x) for x in r]
        st = out["combos"][tag]["stats"]
        print(f"[{tag}] sharpe={st['sharpe_ratio']} ann={st['annualized_return']*100:+.1f}% "
              f"total={st['total_return']*100:+.0f}% mdd={st['max_drawdown']*100:.0f}%", flush=True)
    save()

    for tag, r in port.items():
        ds = deflated_sharpe(r)
        out["dsr"][tag] = ds
        print(f"[DSR] {tag}: DSR={ds['deflated_sharpe']} SR0={ds['sr0_threshold_ann']} "
              f"pass95={ds['pass_95']}", flush=True)
    out["pbo"] = pbo_cscv({k: port[k] for k in port})
    print(f"[PBO] {out['pbo']['pbo']} across {out['pbo']['variants']}", flush=True)
    save()
    print(f"[saved] {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
