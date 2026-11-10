#!/usr/bin/env python3
"""多策略纸面盘赛马: 每条线 100 万, 独立持仓/记账, 同一起始日、同一全市场宇宙。

4 条线:
  22f            22 因子等权复合, topk30/ndrop3, 周频(周五)
  42f            42 因子等权复合(23 诚实+17 Alpha158), topk60/ndrop5, 日频
  factor_ranking 动量/低波动/低振幅 等权 z, topk50/ndrop5, 日频
  gbm_voltgt     LightGBM 预测 fwd5(40 因子 z)+波动率目标仓位, topk60/ndrop5, 日频

用法:
  env -u PYTHONPATH backend/.venv-py313/bin/python3 backend/scripts/multi_line.py \
      --start 2026-10-16 --capital 1000000
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BACKEND = os.path.join(ROOT, "backend")
sys.path.insert(0, BACKEND)

STOCK_DIR = os.path.join(ROOT, "data", "parquet", "stock_data")
PANEL_DIR = os.path.join(ROOT, "data", "qlib_data", "features", "day")
RAW_DIR = os.path.join(ROOT, "data", "tdx_daily", "raw")
OUT_DIR = os.path.join(ROOT, "data", "multi_line")

COMMISSION = 0.0003
MIN_COMMISSION = 5.0
STAMP_TAX = 0.0005
SLIPPAGE = 0.001
CASH_RESERVE = 0.05
BOARD_LOT = 100

LOOKBACK = 80
MIN_CROSS = 30
LIMIT_UP = 0.095


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ds = _load("ds", os.path.join(BACKEND, "scripts", "daily_signal.py"))
ws = _load("ws", os.path.join(BACKEND, "scripts", "weekly_signal.py"))


def load_name_map() -> dict:
    if not os.path.isdir(RAW_DIR):
        return {}
    dates = sorted(d for d in os.listdir(RAW_DIR) if d.startswith("date="))
    if not dates:
        return {}
    try:
        df = pd.read_parquet(os.path.join(RAW_DIR, dates[-1], "part.parquet"),
                             columns=["ts_code", "name"])
        return dict(zip(df["ts_code"].astype(str), df["name"].astype(str)))
    except Exception:
        return {}


# ── 复合分 ────────────────────────────────────────────────────────────
def compute_22f_residual(frames, master):
    """22f 残差版: 同样的 22 因子/符号, 但在'去掉市场'的价格上算。
    残差价格 = 个股超额收益(减等权市场)累积, 只留相对大盘的信号。"""
    close = ds.to_mat(frames, master, "close")
    vol = ds.to_mat(frames, master, "volume")
    ret = close.pct_change(fill_method=None)
    vchg = vol.pct_change(fill_method=None)
    mkt_ret = ret.mean(axis=1)
    excess_ret = ret.sub(mkt_ret, axis=0)
    residual_close = (1.0 + excess_ret.fillna(0.0)).cumprod()
    with np.errstate(divide="ignore", invalid="ignore"):
        cands = {}
        for n in (5, 10, 20, 30, 60):
            cands[f"MIN{n}"] = residual_close.rolling(n).min() / residual_close - 1.0
            cands[f"MAX{n}"] = residual_close.rolling(n).max() / residual_close - 1.0
            cands[f"STD{n}"] = residual_close.rolling(n).std() / residual_close
            cands[f"CORR{n}"] = residual_close.rolling(n).corr(vol)
            cands[f"CORD{n}"] = excess_ret.rolling(n).corr(vchg)
    acc = np.zeros((len(master), len(frames)))
    wsum = np.zeros_like(acc)
    for f in ws.F22:
        ds.zscore_add(acc, wsum, cands[f].to_numpy(dtype=np.float64),
                      ws.SIGNS_22F[f], ds.MIN_CROSS)
    comp = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)
    lu = ret.to_numpy() >= ds.LIMIT_UP
    comp[lu] = np.nan
    return pd.DataFrame(comp, index=master, columns=list(frames))


def compute_residual_mom(frames, master):
    """残差动量 / 相对强度: 个股相对大盘的超额动量, 多窗口等权 z 复合。
    治本: 从'均值回归 beta'(22f)换成'相对强度 alpha'。
    每个窗口跳过最近 5 日(避开 A 股短期反转), 只留中期相对强度。"""
    close = ds.to_mat(frames, master, "close")
    ret = close.pct_change(fill_method=None)
    mkt = close.mean(axis=1)
    acc = np.zeros((len(master), len(frames)))
    wsum = np.zeros_like(acc)
    for n in (10, 20, 40, 60):
        mom = close.shift(5) / close.shift(n + 5) - 1.0          # 跳过最近 5 日
        mkt_mom = mkt.shift(5) / mkt.shift(n + 5) - 1.0
        residual_mom = mom.sub(mkt_mom, axis=0)                  # 相对大盘的超额动量
        ds.zscore_add(acc, wsum, residual_mom.to_numpy(dtype=np.float64),
                      1.0, ds.MIN_CROSS)
    comp = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)
    lu = ret.to_numpy() >= ds.LIMIT_UP
    comp[lu] = np.nan
    return pd.DataFrame(comp, index=master, columns=list(frames))


def compute_rsi_mr(frames, master):
    """RSI 均值回归 + 低波动 + 量价: 用 RSI(超卖)替代 MIN/MAX(距低点) 做均值回归信号。
    RSI 比'距低点距离'更准: 同时考虑跌的幅度和速度, 且对波动自适应。
    辅助因子(低波动/量价)与 22f 完全一致, 只换均值回归那部分, 直接可比。"""
    close = ds.to_mat(frames, master, "close")
    vol = ds.to_mat(frames, master, "volume")
    ret = close.pct_change(fill_method=None)
    vchg = vol.pct_change(fill_method=None)

    def rsi(series, n):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = (-delta.clip(upper=0)).rolling(n).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)

    acc = np.zeros((len(master), len(frames)))
    wsum = np.zeros_like(acc)
    for n in (5, 10, 20, 30, 60):
        ds.zscore_add(acc, wsum, -rsi(close, n).to_numpy(dtype=np.float64),
                      1.0, ds.MIN_CROSS)                 # RSI 超卖(低RSI=买), 替代 MIN/MAX
        ds.zscore_add(acc, wsum, -(close.rolling(n).std() / close).to_numpy(dtype=np.float64),
                      1.0, ds.MIN_CROSS)                 # 低波动
        ds.zscore_add(acc, wsum, -close.rolling(n).corr(vol).to_numpy(dtype=np.float64),
                      1.0, ds.MIN_CROSS)                 # 量价相关
        ds.zscore_add(acc, wsum, -ret.rolling(n).corr(vchg).to_numpy(dtype=np.float64),
                      1.0, ds.MIN_CROSS)                 # 收益-量变相关
    comp = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)
    lu = ret.to_numpy() >= ds.LIMIT_UP
    comp[lu] = np.nan
    return pd.DataFrame(comp, index=master, columns=list(frames))


def compute_rsi_mr_flex(frames, master, periods=(5, 10, 20, 30, 60), use_bb=False):
    """可调参数的 RSI 均值回归。periods=RSI 周期列表; use_bb=是否叠加布林带下轨过滤。
    辅助因子(低波动/量价)恒定, 只扫 RSI 周期和布林带过滤。"""
    close = ds.to_mat(frames, master, "close")
    vol = ds.to_mat(frames, master, "volume")
    ret = close.pct_change(fill_method=None)
    vchg = vol.pct_change(fill_method=None)

    def rsi(series, n):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = (-delta.clip(upper=0)).rolling(n).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)

    acc = np.zeros((len(master), len(frames)))
    wsum = np.zeros_like(acc)
    for n in periods:
        ds.zscore_add(acc, wsum, -rsi(close, n).to_numpy(dtype=np.float64),
                      1.0, ds.MIN_CROSS)
        ds.zscore_add(acc, wsum, -(close.rolling(n).std() / close).to_numpy(dtype=np.float64),
                      1.0, ds.MIN_CROSS)
        ds.zscore_add(acc, wsum, -close.rolling(n).corr(vol).to_numpy(dtype=np.float64),
                      1.0, ds.MIN_CROSS)
        ds.zscore_add(acc, wsum, -ret.rolling(n).corr(vchg).to_numpy(dtype=np.float64),
                      1.0, ds.MIN_CROSS)
    comp = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)
    if use_bb:
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        lower = bb_mid - 2 * bb_std
        comp = np.where(close.to_numpy() < lower.to_numpy(), comp, np.nan)
    lu = ret.to_numpy() >= ds.LIMIT_UP
    comp[lu] = np.nan
    return pd.DataFrame(comp, index=master, columns=list(frames))


def compute_factor_ranking(frames, master):
    """动量(126d 跳 20d) / 低波动(60d) / 低振幅(10d) 等权 z, 横截面。"""
    close = ds.to_mat(frames, master, "close")
    high = ds.to_mat(frames, master, "high")
    low = ds.to_mat(frames, master, "low")
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close.pct_change(fill_method=None)
        momentum = close.shift(20) / close.shift(146) - 1.0
        low_vol = -ret.rolling(60).std()
        low_amp = -(((high - low) / close).rolling(10).mean())
    acc = np.zeros((len(master), len(frames)))
    wsum = np.zeros_like(acc)
    for mat in (momentum, low_vol, low_amp):
        m = mat.to_numpy()
        ok = np.isfinite(m)
        cnt = ok.sum(axis=1, keepdims=True)
        mu = np.nanmean(np.where(ok, m, np.nan), axis=1, keepdims=True)
        sd = np.nanstd(np.where(ok, m, np.nan), axis=1, keepdims=True)
        z = (m - mu) / np.where(sd > 1e-12, sd, np.nan)
        z = np.where(cnt >= MIN_CROSS, z, np.nan)
        acc += z
        wsum += np.where(np.isfinite(z), 1, 0)
    comp = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)
    lu = close.pct_change(fill_method=None).to_numpy() >= LIMIT_UP
    comp[lu] = np.nan
    return pd.DataFrame(comp, index=master, columns=list(frames))


def compute_gbm_voltgt(frames, panel, master, start, target_vol=0.20, vt_window=20,
                       n_est=100, depth=3, lr=0.05, window=750):
    """LightGBM 预测 fwd5 (40 因子 z + presence), 起始日前训练一次, 预测 [start, 最新]。
    返回 (score_df, topk_scale_df)。"""
    import lightgbm as lgb

    close = ds.to_mat(frames, master, "close")
    vol = ds.to_mat(frames, master, "volume")
    cands = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        r = close.pct_change(fill_method=None)
        vchg = vol.pct_change(fill_method=None)
        for n in (5, 10, 20, 30, 60):
            cands[f"MIN{n}"] = close.rolling(n).min() / close - 1.0
            cands[f"MAX{n}"] = close.rolling(n).max() / close - 1.0
            cands[f"STD{n}"] = close.rolling(n).std() / close
            cands[f"CORD{n}"] = r.rolling(n).corr(vchg)
            cands[f"CORR{n}"] = close.rolling(n).corr(vol)

    def zscore(mat):
        ok = np.isfinite(mat)
        cnt = ok.sum(axis=1, keepdims=True)
        mu = np.nanmean(np.where(ok, mat, np.nan), axis=1, keepdims=True)
        sd = np.nanstd(np.where(ok, mat, np.nan), axis=1, keepdims=True)
        z = (mat - mu) / np.where(sd > 1e-12, sd, np.nan)
        return np.where(cnt >= MIN_CROSS, z, np.nan), ok

    feat_list, pres_list = [], []
    for f in ("MIN", "MAX", "STD", "CORD", "CORR"):
        for n in (5, 10, 20, 30, 60):
            z, ok = zscore(cands[f"{f}{n}"].to_numpy())
            feat_list.append(z); pres_list.append(ok.astype(float))
    for name in ("VSUMP5", "VSUMP10", "VSUMP20", "VSUMN5", "VSUMN10", "VSUMN20",
                 "VSUMD5", "VSUMD10", "VSUMD20", "QTLD5", "QTLD10", "QTLD20",
                 "IMAX5", "IMIN5", "WVMA5", "VMA5", "VSTD5"):
        z, ok = zscore(ds.to_panel_mat(panel, master, name))
        feat_list.append(z); pres_list.append(ok.astype(float))

    n_d, n_s = len(master), len(frames)
    F = np.stack(feat_list, axis=2)
    P = np.stack(pres_list, axis=2)
    nfeat = F.shape[2]
    fwd = (close.shift(-1) / close - 1.0).to_numpy()  # 1天周期(对齐日频换仓)

    start_idx = int(master.searchsorted(pd.Timestamp(start)))
    lo = max(0, start_idx - window)
    Xtr = np.concatenate([F[lo:start_idx].reshape(-1, nfeat),
                          P[lo:start_idx].reshape(-1, nfeat)], axis=1)
    ytr = fwd[lo:start_idx].reshape(-1)
    mask = np.isfinite(ytr) & np.isfinite(Xtr[:, :nfeat]).all(axis=1)
    dtrain = lgb.Dataset(Xtr[mask], label=ytr[mask])
    model = lgb.train(
        {"objective": "regression", "num_leaves": 31, "max_depth": depth,
         "learning_rate": lr, "verbose": -1, "num_threads": 8},
        dtrain, num_boost_round=n_est)
    print(f"[gbm] {nfeat} factors, trained on {mask.sum():,} rows "
          f"(window {lo}..{start_idx})", flush=True)

    scores = np.full((n_d, n_s), np.nan)
    for t in range(start_idx, n_d):
        Xt = np.concatenate([F[t].reshape(-1, nfeat),
                             P[t].reshape(-1, nfeat)], axis=1)
        ok_t = np.isfinite(Xt[:, :nfeat]).all(axis=1)
        scores[t] = np.where(ok_t, model.predict(Xt), np.nan)

    score_df = pd.DataFrame(scores, index=master, columns=list(frames))
    mkt = close.mean(axis=1)
    mv = mkt.pct_change().rolling(vt_window).std() * np.sqrt(244)
    scale = (target_vol / mv).clip(upper=1.0).to_numpy()
    scale[:start_idx] = 1.0
    return score_df, pd.Series(scale, index=master)


# ── 纸面盘 TopK-Dropout 调仓 (与 weekly_signal 同一成本模型) ──────────
def limit_pct(code: str) -> float:
    c2 = code[:2]
    if c2 in ("30", "68"):
        return 0.20
    if c2 in ("92", "83", "87"):
        return 0.30
    return 0.10


def rebalance(rank, positions, cash, topk, close, vol, day_ret, names, date,
              ledger, pos_scale=1.0, threshold=0.0):
    rank = rank.dropna().sort_values(ascending=False)
    st = {c for c in rank.index if "ST" in names.get(c, "").upper()}
    if st:
        rank = rank.drop(index=[c for c in st if c in rank.index])
    ndrop = rank.attrs.get("ndrop", 5)
    total_value = cash + sum(positions[c]["shares"] * _px(close, c) for c in positions)
    target_weight = (1.0 - CASH_RESERVE) / topk

    def do_sell(c):
        nonlocal cash
        p = float(close.get(c, 0) or 0)
        if not np.isfinite(p) or p <= 0 or float(vol.get(c, 0) or 0) <= 0:
            return False
        if float(day_ret.get(c, 0) or 0) <= -limit_pct(c) + 1e-4:
            return False
        shares = int(positions[c]["shares"])
        gross = shares * p
        comm = max(MIN_COMMISSION, gross * COMMISSION)
        stamp = gross * STAMP_TAX
        slip = gross * SLIPPAGE
        cash += gross - comm - stamp - slip
        del positions[c]
        ledger.append({"date": str(date)[:10], "action": "sell", "code": c,
                       "name": names.get(c, ""), "price": p, "shares": shares,
                       "amount": round(gross, 2)})
        return True

    def do_buy(c):
        nonlocal cash
        p = float(close.get(c, 0) or 0)
        if not np.isfinite(p) or p <= 0 or float(vol.get(c, 0) or 0) <= 0:
            return False
        target_amount = total_value * target_weight * pos_scale
        shares = int(target_amount / p / BOARD_LOT) * BOARD_LOT
        if shares <= 0:
            return False
        gross = shares * p
        comm = max(MIN_COMMISSION, gross * COMMISSION)
        slip = gross * SLIPPAGE
        cost = gross + comm + slip
        if cost > cash:
            shares = int(cash * 0.999 / p / BOARD_LOT) * BOARD_LOT
            if shares <= 0:
                return False
            gross = shares * p
            comm = max(MIN_COMMISSION, gross * COMMISSION)
            slip = gross * SLIPPAGE
            cost = gross + comm + slip
        cash -= cost
        positions[c] = {"shares": shares, "cost": p, "buy_date": str(date)[:10]}
        ledger.append({"date": str(date)[:10], "action": "buy", "code": c,
                       "name": names.get(c, ""), "price": p, "shares": shares,
                       "amount": round(gross, 2)})
        return True

    # 1. 空仓/不满仓: 直接买最强的填满 topk (无阈值)
    top = list(rank.index[:topk])
    while len(positions) < topk:
        bought = False
        for c in top:
            if c in positions:
                continue
            if do_buy(c):
                bought = True
                break
        if not bought:
            break

    # 2. 满仓: 无交易带换仓
    #    卖候选 = 手里掉出 topk 的 (分数低的); 买候选 = 非持仓但进了 topk 的 (分数高的)
    #    只换 "买候选分数 - 卖候选分数 > threshold" 的, 不值得换的不换
    #    threshold=0 时 = 卖最差ndrop买最强ndrop (现在的瞎买行为)
    sell_cands = [c for c in positions if c not in set(top)]
    sell_cands.sort(key=lambda c: rank.get(c, float("-inf")))  # 最差(分数最低)优先
    buy_cands = [c for c in top if c not in positions]
    buy_cands.sort(key=lambda c: -rank.get(c, 0))  # 最强(分数最高)优先
    for i in range(min(ndrop, len(sell_cands), len(buy_cands))):
        s_sell = rank.get(sell_cands[i], float("-inf"))
        s_buy = rank.get(buy_cands[i], float("-inf"))
        if s_buy - s_sell <= threshold:
            break  # 边缘不够大, 不值得换, 停
        if not do_sell(sell_cands[i]):
            break
        do_buy(buy_cands[i])

    return cash, positions


def _px(close, c):
    v = close.get(c)
    if v is None:
        return 0.0
    v = float(v)
    return v if np.isfinite(v) else 0.0


def liquidate(positions, cash, close, vol, day_ret, names, date, ledger):
    """清仓: 全部卖出(跌停卖不掉就留着), 转现金。"""
    for c in list(positions.keys()):
        p = float(close.get(c, 0) or 0)
        if not np.isfinite(p) or p <= 0 or float(vol.get(c, 0) or 0) <= 0:
            continue
        if float(day_ret.get(c, 0) or 0) <= -limit_pct(c) + 1e-4:
            continue  # 跌停, 卖不出
        shares = int(positions[c]["shares"])
        gross = shares * p
        comm = max(MIN_COMMISSION, gross * COMMISSION)
        stamp = gross * STAMP_TAX
        slip = gross * SLIPPAGE
        cash += gross - comm - stamp - slip
        del positions[c]
        ledger.append({"date": str(date)[:10], "action": "sell", "code": c,
                       "name": names.get(c, ""), "price": p, "shares": shares,
                       "amount": round(gross, 2)})
    return cash, positions


def mark_to_market(positions, cash, close):
    mv = sum(positions[c]["shares"] * _px(close, c) for c in positions)
    return cash + mv


def run_line(name, comp, topk, ndrop, freq, capital, start, master,
             close_mat, vol_mat, day_ret_mat, names, topk_scale=None,
             pos_scale=None, threshold=0.0):
    start_ts = pd.Timestamp(start)
    all_dates = master[master >= start_ts]
    if freq == "weekly":
        rebal = {d for d in all_dates if d.dayofweek == 4}  # 周五
        if not rebal:
            rebal = {all_dates[-1]}
    else:
        rebal = set(all_dates)
    positions, cash, ledger = {}, float(capital), []
    curve = []
    for d in all_dates:
        scale_d = float(pos_scale.loc[d]) if pos_scale is not None else 1.0
        if d in rebal:
            rank = comp.loc[d].copy()
            rank.attrs["ndrop"] = ndrop
            eff_topk = max(5, int(round(topk * topk_scale.loc[d]))) if topk_scale is not None else topk
            if scale_d <= 0:
                # 趋势向下: 清仓空仓
                cash, positions = liquidate(positions, cash,
                                            close_mat.loc[d], vol_mat.loc[d],
                                            day_ret_mat.loc[d], names, d, ledger)
            else:
                cash, positions = rebalance(rank, positions, cash, eff_topk,
                                            close_mat.loc[d], vol_mat.loc[d],
                                            day_ret_mat.loc[d], names, d, ledger,
                                            pos_scale=scale_d, threshold=threshold)
        curve.append({"date": str(d)[:10], "value": round(mark_to_market(
            positions, cash, close_mat.loc[d]), 2)})
    latest_close = close_mat.iloc[-1]
    final = mark_to_market(positions, cash, latest_close)
    # Sharpe / 最大回撤 (按每日净值算)
    vals = pd.Series([c["value"] for c in curve])
    rets = vals.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(244)) if rets.std() > 0 else 0.0
    mdd = float(((vals - vals.cummax()) / vals.cummax()).min())
    return {
        "line": name, "topk": topk, "ndrop": ndrop, "freq": freq,
        "capital": capital, "final_value": round(final, 2),
        "return_pct": round((final / capital - 1) * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(mdd * 100, 2),
        "cash": round(cash, 2),
        "n_holdings": len(positions),
        "n_trades": len(ledger),
        "n_rebalance_days": len(rebal),
        "curve": curve,
        "holdings": [
            {"code": c, "name": names.get(c, ""), "shares": h["shares"],
             "cost": h["cost"],
             "close": float(latest_close.get(c, 0) or 0),
             "pnl_pct": round((float(latest_close.get(c, 0) or 0) / h["cost"] - 1) * 100, 2)}
            for c, h in sorted(positions.items(),
                               key=lambda x: -x[1]["shares"] * float(latest_close.get(x[0], 0) or 0))
        ],
        "ledger": ledger,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-10-16")
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    args = ap.parse_args()

    codes = ds.list_universe()
    frames, panel, master = ds.load_data(codes)
    names = load_name_map()

    close_mat = ds.to_mat(frames, master, "close").ffill(axis=0)  # 停牌股用最后成交价
    vol_mat = ds.to_mat(frames, master, "volume")
    day_ret_mat = close_mat.pct_change(fill_method=None)

    # 市场仓位系数 (波动率目标化): 等权宇宙日收益 -> 20 日已实现波动 -> 目标 20%
    mkt_ret = day_ret_mat.mean(axis=1)
    realized_vol = mkt_ret.rolling(20).std() * np.sqrt(244)
    pos_scale = (0.20 / realized_vol).clip(upper=1.0).fillna(1.0)
    pos_scale = pos_scale.loc[master]

    # 方向择时: 等权市场在 200 日均线之上满仓, 之下清仓空仓
    mkt_close = close_mat.mean(axis=1)
    ma200 = mkt_close.rolling(200).mean()
    timing_scale = (mkt_close > ma200).astype(float).fillna(1.0)
    timing_scale = timing_scale.loc[master]

    print(f"[comp] 22f ...", flush=True)
    comp_22f = ws.compute_composite_22f(frames, master)
    print("[comp] 22f_residual ...", flush=True)
    comp_22f_res = compute_22f_residual(frames, master)
    print("[comp] residual_mom ...", flush=True)
    comp_res_mom = compute_residual_mom(frames, master)
    print("[comp] rsi_mr ...", flush=True)
    comp_rsi = compute_rsi_mr(frames, master)
    print("[comp] rsi_14 ...", flush=True)
    comp_rsi_14 = compute_rsi_mr_flex(frames, master, periods=(14,), use_bb=False)
    print("[comp] rsi_71421 ...", flush=True)
    comp_rsi_71421 = compute_rsi_mr_flex(frames, master, periods=(7, 14, 21), use_bb=False)
    print("[comp] rsi_bb ...", flush=True)
    comp_rsi_bb = compute_rsi_mr_flex(frames, master, periods=(5, 10, 20, 30, 60), use_bb=True)
    print("[comp] 42f ...", flush=True)
    comp_42f = ds.compute_composite(frames, panel, master)
    print("[comp] factor_ranking ...", flush=True)
    comp_fr = compute_factor_ranking(frames, master)
    print("[comp] gbm_voltgt (训练 LightGBM, 1天标签) ...", flush=True)
    comp_gbm, scale = compute_gbm_voltgt(frames, panel, master, args.start)

    os.makedirs(OUT_DIR, exist_ok=True)
    results = {
        "22f": run_line("22f", comp_22f, 30, 3, "weekly", args.capital, args.start, master, close_mat, vol_mat, day_ret_mat, names),
        "22f_voltgt": run_line("22f_voltgt", comp_22f, 30, 3, "weekly", args.capital, args.start, master, close_mat, vol_mat, day_ret_mat, names, pos_scale=pos_scale),
        "22f_timing": run_line("22f_timing", comp_22f, 30, 3, "weekly", args.capital, args.start, master, close_mat, vol_mat, day_ret_mat, names, pos_scale=timing_scale),
        "22f_residual": run_line("22f_residual", comp_22f_res, 30, 3, "weekly", args.capital, args.start, master, close_mat, vol_mat, day_ret_mat, names),
        "residual_mom": run_line("residual_mom", comp_res_mom, 30, 3, "weekly", args.capital, args.start, master, close_mat, vol_mat, day_ret_mat, names),
        "rsi_mr": run_line("rsi_mr", comp_rsi, 30, 3, "weekly", args.capital, args.start, master, close_mat, vol_mat, day_ret_mat, names),
        "rsi_14": run_line("rsi_14", comp_rsi_14, 30, 3, "weekly", args.capital, args.start, master, close_mat, vol_mat, day_ret_mat, names),
        "rsi_71421": run_line("rsi_71421", comp_rsi_71421, 30, 3, "weekly", args.capital, args.start, master, close_mat, vol_mat, day_ret_mat, names),
        "rsi_bb": run_line("rsi_bb", comp_rsi_bb, 30, 3, "weekly", args.capital, args.start, master, close_mat, vol_mat, day_ret_mat, names),
        "42f": run_line("42f", comp_42f, 60, 5, "daily", args.capital, args.start, master, close_mat, vol_mat, day_ret_mat, names),
        "factor_ranking": run_line("factor_ranking", comp_fr, 50, 5, "daily", args.capital, args.start, master, close_mat, vol_mat, day_ret_mat, names),
        "gbm_voltgt": run_line("gbm_voltgt", comp_gbm, 60, 5, "daily", args.capital, args.start, master, close_mat, vol_mat, day_ret_mat, names, topk_scale=scale),
        "gbm_weekly": run_line("gbm_weekly", comp_gbm, 60, 5, "weekly", args.capital, args.start, master, close_mat, vol_mat, day_ret_mat, names, topk_scale=scale),
        "gbm_30_weekly": run_line("gbm_30_weekly", comp_gbm, 30, 3, "weekly", args.capital, args.start, master, close_mat, vol_mat, day_ret_mat, names, topk_scale=scale),
    }

    # 对比表
    print("\n" + "=" * 72)
    print(f"{'线':<16}{'topk/ndrop':<12}{'频率':<8}{'终值':>14}{'收益%':>9}{'Sharpe':>9}{'回撤%':>9}{'持仓':>6}{'交易':>7}")
    print("-" * 80)
    for r in results.values():
        print(f"{r['line']:<16}{str(r['topk'])+'/'+str(r['ndrop']):<12}{r['freq']:<8}"
              f"{r['final_value']:>14,.0f}{r['return_pct']:>9.2f}{r['sharpe']:>9.3f}{r['max_drawdown_pct']:>9.2f}"
              f"{r['n_holdings']:>6}{r['n_trades']:>7}")
    print("=" * 80)

    with open(os.path.join(OUT_DIR, "multi_line_result.json"), "w") as f:
        json.dump({k: v for k, v in results.items()}, f, ensure_ascii=False, indent=1)
    print(f"\n[done] 结果写入 {OUT_DIR}/multi_line_result.json")


if __name__ == "__main__":
    main()
