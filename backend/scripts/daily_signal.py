#!/usr/bin/env python3
"""每日因子排名清单 (研究工具) —— 不是纸面盘的一条线, 不要照着下单。

【口径定位 2026-11-05 定 / 2026-11-06 写死】
本脚本产出的清单 **没有接入** paper_portfolio / 每日盯市 (daily_nav) / 微信推送 / 前端 UI。
纸面盘实际在跑、且在计超额的是 `honest_42f` (周频 topk60/ndrop5) 与 `multi_asset_final` (定稿线),
推送脚本写死 `--strategy honest_42f multi_asset_final`, 与本脚本无关。
=> 这里的清单是**研究侧快照**: 用来观察因子排名的日间变化, 不是可执行指令。
拿它当"今天买什么"下单 = 用一个没有盯市曲线、没有成本核算、没有基准对照的口径真钱交易。

实际默认口径 (旧文档写的 "42f / TopK=60, n_drop=5" 与代码不符, 已改):
  - 因子: 25 个诚实公式因子 (默认 use_panel=False, 不含 17 个 Alpha158 面板因子)
  - 等权截面 z-score, 涨停掩码 (当日涨幅 >= 9.5% 不可买, 分数置 NaN)
  - TopK=30, n_drop=3 (--topk/--ndrop 可改; 但日频 ndrop3 的换手上限 ~1400 笔/年,
    与自用纪律 <=800 笔/年 冲突, 见 docs/PERSONAL_USE_PIVOT_2026-11-05.md)
  - 与 22f 线近乎共线 (25 因子 vs 22 因子, 只差 MAX30/MAX60 的符号) —— 不会给出新信息

数据要求: 先用 tdx_pipeline 拉当日数据 + merge_to_stock_data 落 parquet。
输出: 终端表格 + data/strategy_baseline/daily_signal_<date>_topk<N>.json (meta.caliber 标明研究口径)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 2026-10-26/27：与回测引擎共用的可成交性规则（动态涨跌停阈值 + ST 5%）
from app.services.trading.execution_rules import (  # noqa: E402
    limit_up_mask_matrix,
    load_st_flags,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STOCK_DATA_DIR = os.path.join(ROOT, "data", "parquet", "stock_data")
PANEL_DIR = os.path.join(ROOT, "data", "qlib_data", "features", "day")
OUT_DIR = os.path.join(ROOT, "data", "strategy_baseline")

LIMIT_UP = 0.095
LOOKBACK = 80  # 交易日, 覆盖 n=60 的滚动窗
MIN_CROSS = 30

# 与 honest_composite_strategy 完全一致的因子定义
_HONEST_FORMULAS: dict[str, float] = {}
for _n in (5, 10, 20, 30, 60):
    _HONEST_FORMULAS[f"MIN{_n}"] = 1.0
    _HONEST_FORMULAS[f"MAX{_n}"] = -1.0
    _HONEST_FORMULAS[f"STD{_n}"] = -1.0
    _HONEST_FORMULAS[f"CORR{_n}"] = -1.0
    _HONEST_FORMULAS[f"CORD{_n}"] = -1.0

_ALPHAPANEL_FACTORS: dict[str, float] = {
    "VSUMP5": -1.0, "VSUMP10": -1.0, "VSUMP20": -1.0,
    "VSUMN5": 1.0, "VSUMN10": 1.0, "VSUMN20": 1.0,
    "VSUMD5": -1.0, "VSUMD10": -1.0, "VSUMD20": -1.0,
    "QTLD5": 1.0, "QTLD10": 1.0, "QTLD20": 1.0,
    "IMAX5": -1.0, "IMIN5": -1.0, "WVMA5": -1.0,
    "VMA5": 1.0, "VSTD5": -1.0,
}


def list_universe() -> list[str]:
    """所有有 stock_data parquet 的 A 股代码。

    2026-10-25 修正：原来只判断 `.parquet` 后缀，把
    `000001.parquet` / `000858.parquet` 这类**无交易所后缀的旧命名残留**
    也收进了股票池（stock_data 里同时存在 `000001_SZ.parquet` 和
    `000001.parquet` 两套命名）。后果：
      - 股票池多出重复标的（同一只股票被算两次）
      - 旧的 `000001.parquet` 数据只到 2023-06-30，之后全是 NaN
      - 污染横截面统计（市场宽度、等权指数波动率 → 直接影响
        gbm_voltgt 的波动率目标仓位）

    现在只接受 `<code>_<SH|SZ|BJ>.parquet`，交易所后缀是硬要求。
    """
    codes = []
    for f in sorted(os.listdir(STOCK_DATA_DIR)):
        if not f.endswith(".parquet"):
            continue
        stem = f[: -len(".parquet")]
        # 必须是 <6位数字>_<SH|SZ|BJ> 形式
        if "_" not in stem:
            continue
        code, _, market = stem.partition("_")
        if market not in ("SH", "SZ", "BJ") or not code.isdigit():
            continue
        codes.append(f"{code}.{market}")
    return codes


def load_data(codes: list[str]) -> tuple[dict, dict, pd.DatetimeIndex]:
    """加载 close/high/low/volume, 返回 (frames, panel, master)。"""
    t0 = time.time()
    frames: dict[str, pd.DataFrame] = {}
    panel: dict[str, pd.DataFrame | None] = {}
    for c in codes:
        safe = c.replace(".", "_")
        try:
            df = pd.read_parquet(os.path.join(STOCK_DATA_DIR, f"{safe}.parquet"),
                                 columns=["date", "open", "high", "low", "close", "volume"])
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        if len(df) < 60:
            continue
        frames[c] = df
        try:
            pf = pd.read_parquet(os.path.join(PANEL_DIR, f"{safe}.parquet"))
            pf = pf.reset_index().set_index("date").sort_index()
            pf = pf[~pf.index.duplicated(keep="first")]
            panel[c] = pf
        except Exception:
            panel[c] = None
    codes = list(frames)
    master = pd.DatetimeIndex(sorted(set().union(*[set(f.index) for f in frames.values()])))
    print(f"[load] {len(codes)} stocks, {len(master)} days, "
          f"panel={sum(1 for c in codes if panel.get(c) is not None)} {time.time()-t0:.0f}s", flush=True)
    return frames, panel, master


def to_mat(frames: dict, master: pd.DatetimeIndex, col: str) -> pd.DataFrame:
    n_d, n_s = len(master), len(frames)
    codes = list(frames)
    out = np.full((n_d, n_s), np.nan)
    for j, c in enumerate(codes):
        pos = master.get_indexer(pd.DatetimeIndex(frames[c].index))
        ok = pos >= 0
        out[pos[ok], j] = pd.to_numeric(frames[c][col], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan)[ok]
    return pd.DataFrame(out, index=master, columns=codes)


def to_panel_mat(panel: dict, master: pd.DatetimeIndex, col: str) -> np.ndarray:
    codes = list(panel)
    out = np.full((len(master), len(codes)), np.nan)
    for j, c in enumerate(codes):
        if panel.get(c) is None:
            continue
        pos = master.get_indexer(pd.DatetimeIndex(panel[c].index))
        ok = pos >= 0
        out[pos[ok], j] = pd.to_numeric(panel[c][col], errors="coerce").to_numpy(
            dtype=np.float64, na_value=np.nan)[ok]
    return out


#: 最近一次 compute_composite 在 explain_date 上的因子 z 值分解
#: {factor_name: 1d array(n_stocks)}，供「为什么买它」归因使用
_LAST_FACTOR_Z: Dict[str, np.ndarray] = {}



# ── 行业中性化（2026-10-25 新增）────────────────────────────────────
# 动机（实测）：42 因子复合存在**系统性行业暴露** —— 分数高的票集中在
# 少数行业里。把它做成「行业内选相对最强的」后（先按行业去均值、
# 再在行业内重新标准化），3 年全市场回测：
#   基线           夏普 0.965  年化 +13.25%  回撤 -17.63%  1390 笔
#   行业中性+重归一 夏普 1.441  年化 +25.32%  回撤 -17.86%  1400 笔
# 分年度 4 年全部提升（2023 +27.7% vs +11.5%），不是单年拉起来的。
#
# 实现要点：所有操作都是**逐日横截面**的（只用当日全市场数据），
# 不含未来函数。行业分类是静态映射（申万一级）。
INDUSTRY_MAP_PATH = os.path.join(ROOT, "data", "industry", "industry_map.json")


def load_industry_map() -> dict:
    """{code: 行业名}；文件不存在返回空 dict。"""
    if not os.path.exists(INDUSTRY_MAP_PATH):
        return {}
    try:
        with open(INDUSTRY_MAP_PATH) as fh:
            return json.load(fh)
    except Exception:
        return {}


def industry_ids(codes: list, imap: dict) -> np.ndarray:
    """把代码映射成行业 id 数组（-1 = 无行业分类，不参与中性化）。"""
    names = sorted({imap[c] for c in codes if c in imap})
    lut = {n: i for i, n in enumerate(names)}
    return np.array([lut.get(imap.get(c, ""), -1) for c in codes], dtype=np.int64)


def _row_nan_mean(block: np.ndarray) -> np.ndarray:
    """逐行 nanmean，全 NaN 行返回 0（不产生 RuntimeWarning）。"""
    mask = np.isfinite(block)
    cnt = mask.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        s = np.where(mask, block, 0.0).sum(axis=1, keepdims=True)
    mu = np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)
    return mu


def _row_nan_std(block: np.ndarray) -> np.ndarray:
    """逐行 nanstd，全 NaN 行返回 0（不产生 RuntimeWarning）。"""
    mask = np.isfinite(block)
    cnt = mask.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        mu = _row_nan_mean(block)
        d = np.where(mask, block - mu, 0.0)
        s2 = (d * d).sum(axis=1, keepdims=True)
    var = np.where(cnt > 1, s2 / np.maximum(cnt - 1, 1), 0.0)
    return np.sqrt(np.maximum(var, 0.0))


def industry_demean(z: np.ndarray, ind_ids: np.ndarray, n_ind: int) -> np.ndarray:
    """逐日按行业去均值（nan-aware）。"""
    out = z.copy()
    for g in range(n_ind):
        cols = np.where(ind_ids == g)[0]
        if len(cols) == 0:
            continue
        block = z[:, cols]
        mu = _row_nan_mean(block)
        out[:, cols] = block - mu
    return out


def industry_renorm(z: np.ndarray, ind_ids: np.ndarray, n_ind: int) -> np.ndarray:
    """逐日行业内重新标准化：(z - 行业均值) / 行业标准差。"""
    out = z.copy()
    for g in range(n_ind):
        cols = np.where(ind_ids == g)[0]
        if len(cols) == 0:
            continue
        block = z[:, cols]
        mu = _row_nan_mean(block)
        sd = _row_nan_std(block)
        with np.errstate(invalid="ignore"):
            out[:, cols] = np.where(sd > 1e-12, (block - mu) / sd, np.nan)
    return out


def _zscore_of(mat: np.ndarray, sign: float, min_cross: int) -> np.ndarray:
    """逐日截面 z-score（带符号），不累加。"""
    ok = np.isfinite(mat)
    cnt = ok.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(np.where(ok, mat, np.nan), axis=1, keepdims=True)
        sd = np.nanstd(np.where(ok, mat, np.nan), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (mat - mu) / np.where(sd > 1e-12, sd, np.nan) * sign
    return np.where(cnt >= min_cross, z, np.nan)


def zscore_add(acc: np.ndarray, wsum: np.ndarray, mat: np.ndarray,
               sign: float, min_cross: int,
               collect: Optional[Dict[str, np.ndarray]] = None,
               collect_row: int = -1,
               factor_name: str = "") -> None:
    """把 sign*zscore 累加进 acc。

    collect / collect_row（2026-10-25 新增）：可选地把**某一行**的 z 值
    收集起来，用于「为什么买它」的因子归因。只存一行 → 42 个因子
    仅多占 42×5544×8 ≈ 1.9 MB，不需要第二遍全量计算。
    """
    ok = np.isfinite(mat)
    cnt = ok.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(np.where(ok, mat, np.nan), axis=1, keepdims=True)
        sd = np.nanstd(np.where(ok, mat, np.nan), axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (mat - mu) / np.where(sd > 1e-12, sd, np.nan) * sign
    z = np.where(cnt >= min_cross, z, np.nan)
    good = np.isfinite(z)
    acc[good] += np.nan_to_num(z)[good]
    wsum[good] += 1
    if collect is not None and factor_name:
        collect[factor_name] = z[collect_row].copy()


def compute_composite(frames: dict, panel: dict, master: pd.DatetimeIndex,
                      use_panel: bool = False,
                      explain_date: Optional[pd.Timestamp] = None,
                      industry_neutral: bool = False,
                      exposure_neutral: bool = False,
                      ) -> pd.DataFrame:
    """返回 composite DataFrame (index=date, columns=code), 已应用涨停掩码。

    Args:
        industry_neutral: 行业中性模式。因子 z-score 先按申万一级行业去均值
            （消除系统性行业暴露），再在行业内重新标准化
            （保证行业内可比）。**逐日横截面**，无未来函数。
    """
    n_d, n_s = len(master), len(frames)
    close = to_mat(frames, master, "close")
    vol = to_mat(frames, master, "volume")
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close.pct_change(fill_method=None)
        vchg = vol.pct_change(fill_method=None)
        cands: dict[str, pd.DataFrame] = {}
        for n in (5, 10, 20, 30, 60):
            cands[f"MIN{n}"] = close.rolling(n).min() / close - 1.0
            cands[f"MAX{n}"] = close.rolling(n).max() / close - 1.0
            cands[f"STD{n}"] = close.rolling(n).std() / close
            cands[f"CORR{n}"] = close.rolling(n).corr(vol)
            cands[f"CORD{n}"] = ret.rolling(n).corr(vchg)

    acc = np.zeros((n_d, n_s))
    wsum = np.zeros((n_d, n_s))
    # 因子归因：只收集 explain_date 那一行的 z 值（内存 +1.9MB）
    collect: Optional[Dict[str, np.ndarray]] = None
    collect_row = -1
    if explain_date is not None:
        collect_row = int(master.get_indexer([pd.Timestamp(explain_date)])[0])
        if collect_row < 0:
            collect = None
        else:
            collect = {}

    # 行业中性模式：需要拿到每个因子的 z 矩阵再处理
    if industry_neutral:
        codes = list(frames)
        imap = load_industry_map()
        ind_ids = industry_ids(codes, imap)
        n_ind = int(ind_ids.max()) + 1 if len(ind_ids) else 0
        covered = int((ind_ids >= 0).sum())
        print(f"[neutral] 行业中性: {covered}/{len(codes)} 只有行业分类，"
              f"{n_ind} 个行业", flush=True)
        factor_z: List[Tuple[str, float, np.ndarray]] = []
        for fname, sign in _HONEST_FORMULAS.items():
            z = _zscore_of(cands[fname].to_numpy(), sign, MIN_CROSS)
            factor_z.append((fname, sign, z))
        if use_panel:
            # 2026-10-27：默认**不启用** Alpha158 面板因子 ——
            # 实测 42 因子版夏普 0.799 / 回撤 −16.46%，
            # 25 公式因子版 0.880 / −13.37% → 面板因子净负面。
            for fname, sign in _ALPHAPANEL_FACTORS.items():
                z = _zscore_of(to_panel_mat(panel, master, fname), sign, MIN_CROSS)
                factor_z.append((fname, sign, z))
        # 逐因子：行业去均值 → 行业内重归一 → 带符号累加
        for fname, sign, z in factor_z:
            zn = industry_demean(z, ind_ids, n_ind)
            zn = industry_renorm(zn, ind_ids, n_ind)
            z = zn * sign
            if collect is not None and collect_row >= 0:
                collect[fname] = z[collect_row].copy()
            good = np.isfinite(z)
            acc[good] += np.nan_to_num(z)[good]
            wsum[good] += 1
    else:
        for fname, sign in _HONEST_FORMULAS.items():
            zscore_add(acc, wsum, cands[fname].to_numpy(), sign, MIN_CROSS,
                       collect=collect, collect_row=collect_row, factor_name=fname)
        if use_panel:
            # 2026-10-27：默认**不启用** Alpha158 面板因子
            # （42 因子版夏普 0.799 < 25 公式因子版 0.880）
            for fname, sign in _ALPHAPANEL_FACTORS.items():
                zscore_add(acc, wsum, to_panel_mat(panel, master, fname), sign,
                           MIN_CROSS, collect=collect, collect_row=collect_row,
                           factor_name=fname)

    composite = np.where(wsum > 0, acc / np.where(wsum > 0, wsum, 1.0), np.nan)

    if collect is not None:
        _LAST_FACTOR_Z.clear()
        _LAST_FACTOR_Z.update(collect)
    # 涨停掩码：按板块动态阈值 + ST 5%（2026-10-26 P0-2）
    # 原来硬编码 0.095：20% 板涨 9.5%~20% 被误剔，ST 股 5% 涨停拦不住
    _codes = list(frames)
    _lu = limit_up_mask_matrix(
        close.pct_change(fill_method=None).to_numpy(), _codes,
        is_st=load_st_flags(_codes),
    )
    composite[_lu] = np.nan  # 涨停掩码: 当日不可买

    # ── 暴露中性化（2026-10-27）────────────────────────────────
    # Barra APT 风格：逐日截面 OLS（y=复合分, X=[size, illiq, 1]）取残差。
    # 原生引擎分段验证：Δ夏普 +0.222 / +0.241 / +0.307，回撤每段改善 2~3pp。
    if exposure_neutral:
        _amount = (close * vol * 100.0)  # qlib volume 单位是「手」
        with np.errstate(divide="ignore", invalid="ignore"):
            _log_amt = np.log(np.where(_amount.to_numpy() > 0,
                                      _amount.to_numpy(), np.nan))
        _size = pd.DataFrame(_log_amt).rolling(20, min_periods=5).mean().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            _r = close / close.shift(1) - 1.0
            _daily = (_r.abs() / _amount.where(_amount > 0)).to_numpy()
        _ill = pd.DataFrame(_daily).rolling(
            20, min_periods=10).mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            _illiq = np.log1p(_ill.to_numpy())
        _resid = np.full_like(composite, np.nan)
        for _t in range(len(master)):
            _y = composite[_t]
            _X = np.column_stack([_size[_t], _illiq[_t]])
            _m = np.isfinite(_y) & np.isfinite(_X).all(axis=1)
            if int(_m.sum()) < 50:
                continue
            _A = np.column_stack([_X[_m], np.ones(int(_m.sum()))])
            _beta, *_ = np.linalg.lstsq(_A, _y[_m], rcond=None)
            _resid[_t, _m] = _y[_m] - _A @ _beta
        composite = _resid


    return pd.DataFrame(composite, index=master, columns=list(frames))


def main() -> None:
    ap = argparse.ArgumentParser(description="honest_composite 每日买卖信号")
    ap.add_argument("--topk", type=int, default=30, help="持仓数量 (默认 30)")
    ap.add_argument("--ndrop", type=int, default=3, help="每天最多换几只 (默认 3)")
    args = ap.parse_args()
    topk, ndrop = args.topk, args.ndrop

    codes = list_universe()
    frames, panel, master = load_data(codes)
    comp = compute_composite(frames, panel, master)
    comp = comp.tail(LOOKBACK)
    today = comp.index[-1]
    prev = comp.index[-2]
    close_today = to_mat(frames, master, "close").tail(2).iloc[-1]

    # 排名 (分数降序), 涨停掩码已把涨停票分数置 NaN → 自动落榜不可买
    rank_today = comp.iloc[-1].dropna().sort_values(ascending=False)
    rank_prev = comp.iloc[-2].dropna().sort_values(ascending=False)
    top_today = list(rank_today.index[:topk])
    top_prev = list(rank_prev.index[:topk])

    # TopK-Dropout 执行: 持仓 = top_prev。今天掉出 topk 的持仓里,
    # 只卖最差的 ndrop 只; topk 里新进的, 只买最强的 ndrop 只。
    held = set(top_prev)
    fell_out = [c for c in top_prev if c not in set(top_today)]   # 掉出持仓的
    new_in = [c for c in top_today if c not in held]              # 新进 topk 的
    sells = fell_out[:ndrop]
    buys = new_in[:ndrop]
    holds = [c for c in top_today if c in held and c not in sells]

    result = {
        "meta": {
            "date": str(today.date()), "prev_date": str(prev.date()),
            "n_stocks": len(frames), "topk": topk, "n_drop": ndrop,
            "max_position_size": round((1.0 - 0.05) / topk, 6),
            "generated_at": datetime.now().isoformat(),
            # 口径声明: 这张清单不在纸面盘的 7 条线里, 没有盯市曲线/成本核算/基准对照,
            # 也不在微信推送与前端 UI 的数据源里。研究快照, 不是可执行指令。
            "caliber": "research_only",
            "tracked_in_paper_book": False,
            "pushed_to_wechat": False,
            "note": "纸面盘在计超额的是 honest_42f(周频 topk60/ndrop5) 与 multi_asset_final; 本清单与其口径不同 (25 因子/日频 topk%d/ndrop%d)" % (topk, ndrop),
        },
        "buys": [{"code": c, "score": round(float(rank_today[c]), 4),
                  "close": round(float(close_today[c]), 2)} for c in buys],
        "sells": [{"code": c, "prev_score": round(float(rank_prev[c]), 4)} for c in sells],
        "holds": [{"code": c, "score": round(float(rank_today[c]), 4)} for c in holds],
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"daily_signal_{today.date()}_topk{topk}.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)

    print(f"\n{'='*60}", flush=True)
    print("口径: research_only —— 不在纸面盘 7 条线内, 无盯市曲线/成本核算/基准对照, 别照着下单", flush=True)
    print(f"每日信号  {today.date()}  (vs {prev.date()})", flush=True)
    print(f"宇宙 {len(frames)} 只 | 持仓 top{topk} | 每天最多换 {ndrop} 只 "
          f"| 单仓上限 {(1.0-0.05)/topk*100:.1f}%", flush=True)
    print(f"\n◆ 买入 ({len(buys)} 只)", flush=True)
    for b in result["buys"]:
        print(f"  {b['code']}  score={b['score']:+.3f}  收盘 {b['close']}", flush=True)
    print(f"\n◆ 卖出 ({len(sells)} 只)", flush=True)
    for s in result["sells"]:
        print(f"  {s['code']}  前日 score={s['prev_score']:+.3f}", flush=True)
    print(f"\n◆ 持有 ({len(holds)} 只)", flush=True)
    print("  " + "  ".join(holds[:14]) + (" ..." if len(holds) > 14 else ""), flush=True)
    print(f"\n[saved] {out_path}", flush=True)


if __name__ == "__main__":
    main()
