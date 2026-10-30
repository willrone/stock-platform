#!/usr/bin/env python3
"""P3：低相关资产替换 —— 把「债券类里的股票 beta」换掉。

## 问题（由 P2 + 两轮勘察逼出来的）

P2 已经证明 **ρ(袖套, A 股腿) 是组合夏普的决定性变量**：
bond 腿波动只有 3.3%（最低）但 ρ 高达 0.5536 / 0.6206（最高）→ 按波动率配权 = 反分散。

两轮勘察（`p3_recon.py` / `p3_bond_recon.py`）找到了 ρ 高的**具体原因**：

| 标的 | 名称 | ρ(与 510300, main/oos) | 年化波动 | 性质 |
|---|---|---:|---:|---|
| 511380 | 博时可转债 ETF | **+0.7643 / +0.7976** | 8.14% | **可转债 = 带债底的股票** |
| 511180 | 海富通上证投资级可转债 ETF | +0.8074 | 5.94% | 同上 |
| 511360 | 海富通中证短融 ETF | −0.1573 | 0.34% | 纯债（低波） |
| 511260 | 上证 10 年期国债 ETF | **−0.3070 / −0.2010** | 1.97% | 长久期利率债（真避险） |
| 511010 | 国泰上证 5 年期国债 ETF | −0.2444 / −0.1399 | 1.42% | 长久期利率债 |

**当前定稿的 bond 类（main = 511360 + 511520 + 511380）里混着一只纯股票 beta 的可转债。**
这是**分类错误**，不是参数问题 —— 可转债根本不是"债券腿"的分散化工具。

## 预注册假设

> 把 bond 类里的可转债替换成**长久期利率债**，bond 类的 ρ 会从 0.55 掉到接近 0，
> 组合夏普提升，且这个提升在**推导窗口与样本外窗口都同向**。

## 选腿规则（严格 point-in-time）

- **ρ 依据**：窗口**开始前** 250 个交易日，候选 vs **510300**（沪深300 ETF，代理 A 股 beta）。
  为什么用代理不用 A 股腿：A 股腿面板数据从 2021-05-11 起，样本外窗口没有"窗口前"数据，
  用推导窗口算就等于用未来选资产。510300 数据可回溯到 2012 年，两个窗口都能算。
- **流动性门槛**：窗口**开始前** 120 个交易日日均成交额 ≥ `MIN_TURNOVER`（默认 5000 万元），
  新选的标的必须过门槛；**基线不动**（基线用的是既定的窗口内流动性排序口径）。
- **候选池**：全部有 parquet 数据的 ETF（567 只），按窗口独立筛选。

## 对照（排除"碰巧选中赢家"）

- `swap_rand_s{seed}`：把可转债换成**同池随机** 1 只（6 个固定种子）
- `extra_rand_s{seed}`：新增第 4 类 = **同池随机** 3 只（6 个固定种子）

## 臂

P3a（bond 类内替换，改动最小、因果最干净）
- `base`                 定稿现状
- `bond_no_cb`           去掉 511380（剩 2 只）
- `bond_cb_to_lowrho`    511380 → 债券池内 ρ 最低且流动性达标的那只
- `bond_lowrho3`         bond 类 = ρ 最低 top-3
- `bond_all_rate`        bond 类 = 全部非可转债候选（按流动性 top-3）
- `swap_rand_s{seed}`    511380 → 同池随机 1 只

P3b（全池低相关搜索：新增第 4 类）
- `extra_lowrho3`        第 4 类 = 全池 ρ 最低 top-3（排除现有三类持仓）
- `extra_rand_s{seed}`   第 4 类 = 同池随机 3 只

## 口径

- A 股腿 50%（honest_composite），基线复现门不通过直接退出
- 袖套 50%，类内等权，类间等权，每月（21 交易日）再平衡回等权
- 成交：信号日收盘 → 次日 open，佣金 0.03% + 滑点 0.01%（ETF 免印花税）

## 用法
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p3_bond.py --window main
    cd backend && env -u PYTHONPATH BASELINE_LOG_LEVEL=ERROR .venv-py313/bin/python3 \
        scripts/multi_asset_p3_bond.py --window oos
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

import multi_asset_l3_momentum as L3  # noqa: E402
import multi_asset_p2 as P2  # noqa: E402
from multi_asset_leg_experiment import (  # noqa: E402
    END,
    OOS_END,
    OOS_START,
    START,
    load_etf_universe,
    maxdd,
    merge_by_date,
    sharpe,
    total_ret,
    yearly,
)

K_IN_CLASS = 3
REB_EVERY = 21
SEEDS = (20261030, 7, 99, 1234, 555, 8888)
CLASSES = ["gold", "bond", "commodity"]

PROXY = "510300"
LONG_START = datetime(2019, 1, 1)
LONG_END = datetime(2026, 10, 30)
RHO_LOOKBACK = 250        # 窗口前算 ρ 的回看交易日
MIN_HISTORY = 200         # 窗口前至少要有这么多天重叠
LIQ_LOOKBACK = 120        # 窗口前流动性回看
MIN_TURNOVER = 5e7        # 窗口前日均成交额门槛 5000 万元
START_TOLERANCE_DAYS = 7

ETF_DIR = os.path.join(ROOT, "data", "etf")
PARQUET_DIR = os.path.join(ETF_DIR, "parquet")


# ───────────────────────── 候选池 ─────────────────────────


def _load_code(code: str):
    import pandas as pd
    path = os.path.join(PARQUET_DIR, f"{code}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path).sort_values("date")
    df["date"] = pd.to_datetime(df["date"])
    return df


def _rets(df):
    return df.set_index("date")["close"].pct_change().dropna()


def build_pool(wstart: datetime) -> List[Dict[str, Any]]:
    """窗口开始前的候选池：ρ(与 510300) + 流动性 + 是否可转债。

    严格 point-in-time：所有指标只用窗口开始前的数据。
    """
    import numpy as np
    import pandas as pd

    ws = pd.Timestamp(wstart.strftime("%Y-%m-%d"))
    w0 = ws - pd.Timedelta(days=START_TOLERANCE_DAYS)

    proxy = _load_code(PROXY)
    if proxy is None:
        raise SystemExit("[p3] ❌ 代理 510300 无数据")
    pre = _rets(proxy)
    pre = pre[pre.index < ws]
    if len(pre) < MIN_HISTORY:
        raise SystemExit(f"[p3] ❌ 代理窗口前只有 {len(pre)} 天数据")
    pre = pre.iloc[-RHO_LOOKBACK:]

    universe = load_etf_universe()
    have_data = {os.path.basename(p)[:-8]
                 for p in glob.glob(os.path.join(PARQUET_DIR, "*.parquet"))}

    rows: List[Dict[str, Any]] = []
    for code in sorted(have_data):
        meta = universe["etfs"].get(code, {})
        tags = meta.get("tags", [])
        df = _load_code(code)
        if df is None or df.empty:
            continue
        if df["date"].iloc[0] > w0:
            continue
        if df["date"].iloc[-1] < ws:
            continue
        r = _rets(df)
        r_pre = r[(r.index >= pre.index[0]) & (r.index <= pre.index[-1])]
        joined = pd.concat([r_pre, pre], axis=1, join="inner").dropna()
        if len(joined) < MIN_HISTORY:
            continue
        a, b = joined.iloc[:, 0], joined.iloc[:, 1]
        if a.std() == 0 or b.std() == 0:
            continue
        rho = float(np.corrcoef(a, b)[0, 1])
        liq = df[df["date"] < ws].tail(LIQ_LOOKBACK)
        turnover = float((liq["volume"].fillna(0) * liq["close"] * 100).mean()) if len(liq) else 0.0
        rows.append({
            "code": code,
            "name": meta.get("name", ""),
            "tags": tags,
            "is_convertible": any("可转债" in (meta.get("name", "") + t) for t in [meta.get("name", "")] + list(tags)),
            "rho": round(rho, 4),
            "vol": round(float(a.std() * np.sqrt(244)), 4),
            "turnover": round(turnover, 0),
            "n": int(len(joined)),
        })
    rows.sort(key=lambda x: x["rho"])
    return rows


# ───────────────────────── 组合构建 ─────────────────────────


def fixed_weights_for(classes: List[str]):
    w = {c: 1.0 / len(classes) for c in classes}
    return lambda _d: dict(w)


def build_targets(calendar, by_class_held, weight_fn, reb_every=REB_EVERY):
    targets: Dict[str, Dict[str, float]] = {}
    log: List[Dict[str, Any]] = []
    for i in range(1, len(calendar)):
        if (i - 1) % reb_every != 0:
            continue
        signal_date, exec_date = calendar[i - 1], calendar[i]
        cw = weight_fn(signal_date)
        if not cw:
            continue
        t: Dict[str, float] = {}
        for c, w in cw.items():
            if w <= 0:
                continue
            codes = by_class_held.get(c) or []
            if not codes:
                continue
            for code in codes:
                t[code] = t.get(code, 0.0) + w / len(codes)
        if t:
            targets[exec_date] = t
            log.append({"date": signal_date, "exec": exec_date,
                        "class_weights": {k: round(v, 4) for k, v in cw.items()}})
    return targets, {"n_rebalances": len(log), "weight_log": log}


def rolling_sharpe_win(nav_a, nav_b, window: int) -> Optional[float]:
    """滚动 window 日夏普：nav_b 胜 nav_a 的窗口占比（P1 的标准判据）。"""
    if not nav_a or not nav_b:
        return None
    da = dict(nav_a)
    db = dict(nav_b)
    dates = sorted(set(da) & set(db))
    if len(dates) < window + 10:
        return None
    ra = [da[dates[i]] / da[dates[i - 1]] - 1 for i in range(1, len(dates))]
    rb = [db[dates[i]] / db[dates[i - 1]] - 1 for i in range(1, len(dates))]
    wins = 0
    total = 0
    for i in range(len(dates) - 1 - window):
        sa = ra[i:i + window]
        sb = rb[i:i + window]
        if len(sa) < window or len(sb) < window:
            continue
        va = sum(sa) / window
        vb = sum(sb) / window
        da_ = (sum((x - va) ** 2 for x in sa) / (window - 1)) ** 0.5
        db_ = (sum((x - vb) ** 2 for x in sb) / (window - 1)) ** 0.5
        if da_ <= 0 or db_ <= 0:
            continue
        total += 1
        if vb / db_ > va / da_:
            wins += 1
    return round(wins / total, 4) if total else None



def run_long(args) -> None:
    """长历史（2019-01 → 2026-10）交叉验证：固定 bond 持仓集的袖套表现。

    为什么长历史只能测"固定持仓集"：
    - 510300 代理与所有 ETF 的 parquet **都从 2019-01-02 起** → 长历史窗口
      **没有"窗口前"数据**，无法做 point-in-time 的 ρ 选腿。
    - 长历史也没有 A 股腿（面板数据从 2021-05-11 起）→ 测不到 ρ 这个决定性变量
      （P2 已证：袖套夏普升 ≠ 组合夏普升）。

    所以长历史的角色是：**交叉验证 P3 选出的持仓集在 7.7 年里不是坏的**，
    而不是独立验证选腿规则。结论必须按这个定位写。
    """
    import multi_asset_long_history as LH

    s_str, e_str = LONG_START.strftime("%Y-%m-%d"), LONG_END.strftime("%Y-%m-%d")
    out_path = os.path.join(ROOT, "data", "backtest", "multi_asset_p3_long.json")

    universe = load_etf_universe()
    by_class_raw = LH.pick_point_in_time(universe, CLASSES, s_str, e_str)
    held = {c: lst[:K_IN_CLASS] for c, lst in by_class_raw.items()}
    print(f"[p3] 长历史持有: {json.dumps(held, ensure_ascii=False)}", flush=True)

    base_codes = sorted({c for lst in by_class_raw.values() for c in lst})
    extra = ["511260", "511010", "511020", "511380", "511520", "511180"]
    prices = L3.load_prices(sorted(set(base_codes) | set(extra)))
    calendar = L3.build_calendar(prices, base_codes, s_str, e_str)
    print(f"[p3] 交易日历 {len(calendar)} 天 {calendar[0]} → {calendar[-1]}", flush=True)
    if len(calendar) < 500:
        raise SystemExit("[p3] ❌ 长历史日历过短，退出")

    results: Dict[str, Any] = {}
    navs: Dict[str, List[Tuple[str, float]]] = {}

    def run(label: str, held_cfg: Dict[str, List[str]]):
        tg, tmeta = build_targets(calendar, held_cfg, fixed_weights_for(CLASSES))
        c_of = {c: cls for cls, lst in held_cfg.items() for c in lst}
        nav, srec = L3.simulate(calendar, prices, tg, c_of,
                                {c: 1.0 / len(CLASSES) for c in CLASSES})
        if not nav:
            print(f"  {label}: ❌ 无净值", flush=True)
            return
        navs[label] = nav
        rec = {
            "held": held_cfg,
            "sleeve_sharpe": round(sharpe(nav), 4),
            "sleeve_return": round(total_ret(nav), 4),
            "sleeve_maxdd": round(maxdd(nav), 4),
            "sleeve_vol": round(P2.nav_vol(nav), 4),
            "cost": srec, **tmeta,
        }
        results[label] = rec
        print(f"  {label:26s} 袖套夏普 {rec['sleeve_sharpe']:.4f}  "
              f"波动 {rec['sleeve_vol']:.2%}  收益 {rec['sleeve_return']:+.2%}  "
              f"回撤 {rec['sleeve_maxdd']:.2%}  换手 {srec['turnover_ratio']:.2f}x", flush=True)

    base_bond = held["bond"]
    run("base_long", held)
    low3 = ["511260", "511010", "511020"]
    if set(low3) != set(base_bond):
        run("lowrho3_long", {**held, "bond": low3})
    # 含可转债的示意臂：长历史里 511380 2020-04 才上市（前期是现金）
    run("plus_cb_long", {**held, "bond": base_bond[:2] + ["511380"]})

    if "base_long" in navs:
        print(f"\n[p3] === 滚动夏普胜率 vs base_long ===", flush=True)
        for label, nav in navs.items():
            if label == "base_long":
                continue
            w252 = rolling_sharpe_win(navs["base_long"], nav, 252)
            w504 = rolling_sharpe_win(navs["base_long"], nav, 504)
            results[label]["rolling_sharpe_252d_win_rate"] = w252
            results[label]["rolling_sharpe_504d_win_rate"] = w504
            print(f"  {label:26s} 滚动252日 {w252}  滚动504日 {w504}", flush=True)

    store = {"window": "long", "baseline_held": held,
             "current_liquidity": current_liquidity(prices),
             "calendar": {"n": len(calendar), "start": calendar[0], "end": calendar[-1]},
             "results": results}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=1)
    print(f"\n✓ 已写入 {out_path}", flush=True)


def current_liquidity(prices) -> Dict[str, Any]:
    """各标的近 120 交易日日均成交额（截至产物生成日）—— 实盘可执行性核查。"""
    cur_liq: Dict[str, Any] = {}
    try:
        import pandas as pd
        for _c in sorted(prices.keys()):
            _p = os.path.join(L3.ETF_PARQUET, f"{_c}.parquet")
            if not os.path.exists(_p):
                continue
            _df = pd.read_parquet(_p).sort_values("date")
            _df["date"] = pd.to_datetime(_df["date"])
            _t = _df.tail(120)
            _amt = float((_t["volume"].fillna(0) * _t["close"].fillna(0) * 100).mean())
            cur_liq[_c] = {"avg_amount_120d": round(_amt, 2),
                           "last_date": str(_df["date"].iloc[-1].date())}
    except Exception as _e:  # pragma: no cover
        cur_liq = {"error": str(_e)}
    return cur_liq


# ───────────────────────── 主流程 ─────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="main", choices=["main", "oos", "long"])
    ap.add_argument("--quick", action="store_true", help="读缓存的 A 股腿")
    args = ap.parse_args()

    if args.window == "long":
        run_long(args)
        return

    import multi_asset_leg_experiment as M

    if args.window == "main":
        start, end = START, END
    elif args.window == "oos":
        start, end = OOS_START, OOS_END
    else:
        start, end = LONG_START, LONG_END
    s_str, e_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    out_path = os.path.join(ROOT, "data", "backtest", f"multi_asset_p3_{args.window}.json")
    nav_cache = os.path.join(ROOT, "data", "backtest", f"_a_share_nav_{args.window}.json")

    # ── A 股腿 ──
    M.START, M.END = start, end   # ⚠️ 模块全局窗口，必须显式同步
    a_nav: List[Tuple[str, float]] = []
    a_rec: Dict[str, Any] = {}
    if args.quick and os.path.exists(nav_cache):
        with open(nav_cache, encoding="utf-8") as fh:
            cached = json.load(fh)
        a_nav = [(d, float(v)) for d, v in cached["a_share_nav"]]
        a_rec = cached.get("a_share_leg", {})
        print("[p3] 读缓存 A 股腿", flush=True)
    else:
        from verify_methods_engine import QLIB_OFFICIAL, universe_from_instruments
        codes = universe_from_instruments(QLIB_OFFICIAL)
        print(f"[p3] 跑 A 股腿（{len(codes)} 只）...", flush=True)
        a_rec, a_nav = M.run_a_share_leg(codes)
        ver = M.verify_baseline(a_rec)
        if args.window == "main":
            print(f"[p3] 基线复现门: {json.dumps(ver, ensure_ascii=False)}", flush=True)
            if ver.get("checked") and not ver.get("ok"):
                raise SystemExit("[p3] ❌ 基线复现失败，退出")
    if not a_nav:
        raise SystemExit("[p3] ❌ A 股腿净值为空")
    if not (s_str <= a_nav[0][0] <= e_str):
        raise SystemExit(f"[p3] ❌ A 股腿起点 {a_nav[0][0]} 不在窗口 {s_str}~{e_str} 内")
    base_sharpe = sharpe(a_nav)

    # ── 候选池 ──
    pool = build_pool(start)
    by_code = {r["code"]: r for r in pool}
    universe = load_etf_universe()
    by_class_raw = L3.candidates_by_class(universe, start, end)
    held = {c: lst[:K_IN_CLASS] for c, lst in by_class_raw.items()}
    print(f"[p3] 基线持有: {json.dumps(held, ensure_ascii=False)}", flush=True)
    print(f"[p3] 候选池 {len(pool)} 只；ρ 最低 12 只：", flush=True)
    for r in pool[:12]:
        print(f"    {r['code']}  ρ={r['rho']:+.4f}  vol={r['vol']:.2%}  "
              f"成交额 {r['turnover']/1e4:,.0f}万  {r['name']}", flush=True)

    # 债券池（ρ 最低的可用者）
    bond_pool = [r for r in pool if "bond" in r["tags"]]
    bond_liquid = [r for r in bond_pool if r["turnover"] >= MIN_TURNOVER]
    print(f"[p3] 债券池 {len(bond_pool)} 只，流动性达标 {len(bond_liquid)} 只: "
          f"{[r['code'] for r in bond_liquid]}", flush=True)

    # ⚠️ **价格索引 与 交易日历 必须分开处理**（本项目踩过）：
    # - 价格索引要覆盖**所有可能被选中的**代码（否则 simulate 护栏直接报错；
    #   随机对照会抽到索引外的代码）
    # - 但**交易日历只能按基线候选并集**算：把全池并进日历会让基线从 1.3767 漂到
    #   1.4103（多出来的候选带来额外交易日）→ 所有配置不可比
    base_codes = sorted({c for lst in by_class_raw.values() for c in lst})
    all_codes = sorted(set(base_codes) | {r["code"] for r in pool})
    prices = L3.load_prices(all_codes)
    calendar = L3.build_calendar(prices, base_codes, s_str, e_str)
    print(f"[p3] 交易日历 {len(calendar)} 天 {calendar[0]} → {calendar[-1]}", flush=True)
    if len(calendar) < 300:
        raise SystemExit("[p3] ❌ 交易日历过短，退出")

    results: Dict[str, Any] = {}
    navs: Dict[str, List[Tuple[str, float]]] = {}
    cur_liq = current_liquidity(prices)

    store: Dict[str, Any] = {"window": args.window,
        "baseline_held": held,
        "pool_size": len(pool),
        "bond_pool": [{k: r[k] for k in ("code", "name", "rho", "vol", "turnover")}
                      for r in bond_pool],
        "rho_top20": [{k: r[k] for k in ("code", "name", "rho", "vol", "turnover", "tags")}
                      for r in pool[:20]],
        "calendar": {"n": len(calendar), "start": calendar[0], "end": calendar[-1]},
        "current_liquidity": cur_liq,
    }
    if a_rec:
        store["a_share_leg"] = a_rec

    def run_cfg(label: str, held_cfg: Dict[str, List[str]], classes: List[str],
                meta: Optional[Dict[str, Any]] = None):
        # ── 不变量：同一类内持仓代码不得重复（重复会被 dict 覆盖 → 有效持仓数减少、
        #    权重之和小于 1，静默扭曲结果）──
        for _c, _lst in held_cfg.items():
            if len(set(_lst)) != len(_lst):
                raise SystemExit(f"[p3] ❌ {label}: 类 {_c} 持仓重复 {_lst}，退出")
        tg, tmeta = build_targets(calendar, held_cfg, fixed_weights_for(classes))
        c_of = {c: cls for cls, lst in held_cfg.items() for c in lst}
        sleeve_nav, srec = L3.simulate(
            calendar, prices, tg, c_of, {c: 1.0 / len(classes) for c in classes})
        if not sleeve_nav:
            print(f"  {label}: ❌ 无净值", flush=True)
            return
        combined = merge_by_date([a_nav, sleeve_nav], [0.5, 0.5])
        cs = sharpe(combined)
        navs[label] = combined
        # 剔 gold 口径：去掉 gold 类后，把 gold 的仓位**重新分配给剩余两类并补满**
        # （scale = len(classes)/len(ng)）。
        # ⚠️ 旧公式乘的是 (1/len(classes))/(1/len(ng)) = 2/3 → 袖套仓位只剩 4/9，
        #    既不是"gold 变现金"也不是"补满仓" → 夏普与满仓口径不可比。已修正。
        ng = [c for c in classes if c != "gold"]
        sharpe_ex_gold: Optional[float] = None
        if ng and len(ng) < len(classes):
            ng_t: Dict[str, Dict[str, float]] = {}
            for d, t in tg.items():
                # 按**当日 gold 实际权重**归一（动态权重下不能写死 1/len(classes)）；
                # gold 的份额按原比例分给剩余类 → 总仓位补满 1，夏普与满仓口径可比。
                rest_w = sum(w for code, w in t.items() if c_of.get(code) != "gold")
                if rest_w <= 1e-12:
                    continue
                scale = 1.0 / rest_w
                nt = {code: w * scale for code, w in t.items()
                      if c_of.get(code) != "gold"}
                if nt:
                    ng_t[d] = nt
            # 不变量：剔 gold 后每个调仓日的总权重必须为 1（否则仓位变化污染夏普）
            for _d, _t in ng_t.items():
                if abs(sum(_t.values()) - 1.0) > 1e-6:
                    raise SystemExit(
                        f"[p3] ❌ 剔 gold 后权重和 {sum(_t.values())} != 1（{_d}）")
            ng_sleeve, _ = L3.simulate(calendar, prices, ng_t, c_of,
                                       {c: 1.0 / len(ng) for c in ng})
            ng_comb = merge_by_date([a_nav, ng_sleeve], [0.5, 0.5]) if ng_sleeve else []
            sharpe_ex_gold = round(sharpe(ng_comb), 4) if ng_comb else None

        # 每类单独与 A 股腿的 ρ
        cls_rho: Dict[str, Any] = {}
        for c in classes:
            codes = held_cfg.get(c) or []
            if not codes:
                continue
            sub = {cc: cc for cc in codes}
            t2 = {d: {k: v for k, v in t.items() if k in codes} for d, t in tg.items()}
            t2 = {d: t for d, t in t2.items() if t}
            n2, _ = L3.simulate(calendar, prices, t2, sub, {c: 1.0})
            cls_rho[c] = {"rho_a": P2.nav_corr(n2, a_nav) if n2 else None,
                          "sleeve_sharpe": round(sharpe(n2), 4) if n2 else None}
        rec = {
            "held": {c: held_cfg.get(c) for c in classes},
            "classes": classes,
            "combined_sharpe": round(cs, 4),
            "vs_baseline": round(cs / base_sharpe - 1, 4),
            "combined_return": round(total_ret(combined), 4),
            "combined_maxdd": round(maxdd(combined), 4),
            "combined_yearly": yearly(combined),
            "sleeve_sharpe": round(sharpe(sleeve_nav), 4),
            "sleeve_vol": round(P2.nav_vol(sleeve_nav), 4),
            "sleeve_return": round(total_ret(sleeve_nav), 4),
            "corr_sleeve_ashare": P2.nav_corr(sleeve_nav, a_nav),
            "class_rho": cls_rho,
            "sharpe_ex_gold": sharpe_ex_gold,
            "cost": srec,
            **tmeta,
            **(meta or {}),
        }
        results[label] = rec
        print(f"  {label:28s} 组合夏普 {cs:.4f} ({rec['vs_baseline']:+.2%})  "
              f"收益 {rec['combined_return']:+.2%}  回撤 {rec['combined_maxdd']:+.2%}  | "
              f"袖套 {rec['sleeve_sharpe']:.4f} 波动 {rec['sleeve_vol']:.2%}  "
              f"ρ(袖套,A) {rec['corr_sleeve_ashare']}  剔gold {sharpe_ex_gold}  "
              f"换手 {srec['turnover_ratio']:.2f}x",
              flush=True)
        for c in classes:
            r = cls_rho.get(c, {})
            print(f"      {c:10s} {held_cfg.get(c)}  ρ(类,A) {r.get('rho_a')}  "
                  f"夏普 {r.get('sleeve_sharpe')}", flush=True)

    # ═══════════ P3a · bond 类内替换 ═══════════
    print(f"\n[p3] === P3a · bond 类内替换（gold/commodity 固定 top-3）===", flush=True)
    base_bond = held["bond"]
    cb_code = "511380"
    has_cb = cb_code in base_bond
    print(f"[p3] 基线 bond = {base_bond}（含可转债 {cb_code}: {has_cb}）", flush=True)

    run_cfg("base", held, CLASSES, {"note": "定稿现状"})
    # ── 基线复现门：base 必须逐位复现 P2 定稿的组合夏普，否则全部配置不可比 ──
    EXPECT_BASE = {"main": 1.3767, "oos": 1.0382}
    if abs(results["base"]["combined_sharpe"] - EXPECT_BASE[args.window]) > 5e-4:
        raise SystemExit(
            f"[p3] ❌ 基线未复现：base {results['base']['combined_sharpe']} "
            f"≠ 定稿 {EXPECT_BASE[args.window]}（日历/成本/口径漂移），退出")
    print(f"[p3] ✅ 基线复现门通过：{results['base']['combined_sharpe']}", flush=True)

    if has_cb:
        run_cfg("bond_no_cb", {**held, "bond": [c for c in base_bond if c != cb_code]},
                CLASSES, {"note": "仅去掉可转债"})

    # 低 ρ 首选：债券池内 ρ 最低且流动性达标
    low1 = bond_liquid[0] if bond_liquid else None
    if low1 and low1["code"] != cb_code:
        # 注意：low1 可能已经在 base_bond 里（如 OOS 的 511260）→ 必须去重，
        # 否则同一只 ETF 在 dict 里被覆盖，实际只剩 2 只有效持仓、权重之和 < 1。
        rest = [c for c in base_bond if c != cb_code and c != low1["code"]]
        run_cfg("bond_cb_to_lowrho",
                {**held, "bond": [low1["code"]] + rest},
                CLASSES, {"note": f"可转债 → ρ 最低 {low1['code']}",
                          "swap": {"from": cb_code, "to": low1["code"]}})

    # bond 类 = ρ 最低 top-3（流动性达标；不足 3 只就少几只）
    low3 = [r["code"] for r in bond_liquid[:K_IN_CLASS]]
    if low3:
        run_cfg("bond_lowrho3", {**held, "bond": low3}, CLASSES,
                {"note": "bond 类 = 流动性达标池内 ρ 最低 top-3", "picked": low3})
    # 不限流动性的理论上限（仅参考，可能不可实盘）
    low3_nl = [r["code"] for r in bond_pool[:K_IN_CLASS]]
    if low3_nl and low3_nl != low3:
        run_cfg("bond_lowrho3_noliq", {**held, "bond": low3_nl}, CLASSES,
                {"note": "bond 类 = ρ 最低 top-3（不限流动性，理论上限）", "picked": low3_nl})

    # bond 类 = 全部非可转债候选（流动性达标，按流动性 top-3）
    non_cb_liq = [r["code"] for r in bond_pool
                  if not r["is_convertible"] and r["turnover"] >= MIN_TURNOVER]
    if len(non_cb_liq) >= 2:
        run_cfg("bond_all_rate", {**held, "bond": non_cb_liq[:K_IN_CLASS]}, CLASSES,
                {"note": "bond 类 = 流动性达标的全部非可转债", "picked": non_cb_liq[:K_IN_CLASS]})
    non_cb_all = [r["code"] for r in bond_pool if not r["is_convertible"]]
    if len(non_cb_all) >= 2 and non_cb_all[:K_IN_CLASS] != non_cb_liq[:K_IN_CLASS]:
        run_cfg("bond_all_rate_noliq", {**held, "bond": non_cb_all[:K_IN_CLASS]}, CLASSES,
                {"note": "bond 类 = 全部非可转债（不限流动性）", "picked": non_cb_all[:K_IN_CLASS]})

    # 同池随机对照：把可转债换成随机 1 只
    if has_cb:
        # 对照池：债券池内的**非可转债**，排除已在持有集里的代码。
        # 刻意不限流动性门槛 —— OOS 窗口(2021-05)流动性达标的债券只有 2 只，
        # 若加门槛则 6 个随机种子全部退化成同一只，随机对照形同虚设。
        # 每个 pick 的流动性会单独记录在 meta 里，文档如实标注。
        rng_pool = [r for r in bond_pool
                    if not r["is_convertible"] and r["code"] not in base_bond]
        picks: Dict[int, str] = {}
        for seed in SEEDS:
            rng = random.Random(seed)
            pick = rng.choice(rng_pool)["code"] if rng_pool else None
            if not pick:
                continue
            picks[seed] = pick
            run_cfg(f"swap_rand_s{seed}",
                    {**held, "bond": [pick] + [c for c in base_bond if c != cb_code]},
                    CLASSES, {"note": "可转债 → 同池随机 1 只（非可转债）",
                              "swap": {"from": cb_code, "to": pick},
                              "pick_turnover": next((x["turnover"] for x in bond_pool
                                                     if x["code"] == pick), None),
                              "pick_is_liquid": pick in {x["code"] for x in bond_liquid}})
        # 不变量：多个随机种子必须产生互不相同的持仓集，否则随机对照失效
        uniq = {tuple(sorted(results[f"swap_rand_s{s}"]["held"]["bond"]))
                for s in picks if f"swap_rand_s{s}" in results}
        print(f"[p3] 随机对照：{len(picks)} 个种子 → {len(uniq)} 个不同持仓集 "
              f"{sorted(uniq)}", flush=True)
        if len(picks) >= 3 and len(uniq) < 2:
            raise SystemExit("[p3] ❌ 随机对照退化（所有种子同一配置），退出")

    # ═══════════ P3a-2 · 可转债独立成第 4 类 ═══════════
    # 可转债 ρ 0.76 高，但它有独立的收益来源（债底 + 转股溢价 + 正股期权）。
    # 「因 ρ 高就丢掉」和「放在 bond 类里污染 ρ」都不对 —— 单独成类才是公平处理。
    if has_cb:
        cb_only = [cb_code]
        rate_only = [c for c in non_cb_liq] if non_cb_liq else [c for c in base_bond if c != cb_code]
        if rate_only:
            h_cb = {**held, "bond": rate_only, "cb": cb_only}
            run_cfg("bond_cb_sep", h_cb, CLASSES + ["cb"],
                    {"note": "可转债独立成第 4 类；bond 类 = 流动性达标的纯债",
                     "picked": {"bond": rate_only, "cb": cb_only}})

    # ═══════════ P3b · 全池低相关搜索：新增第 4 类 ═══════════
    print(f"\n[p3] === P3b · 新增第 4 类（全池 ρ 最低 top-3）===", flush=True)
    used = {c for lst in held.values() for c in lst}
    extra_pool = [r for r in pool if r["code"] not in used and r["turnover"] >= MIN_TURNOVER]
    lowx = [r["code"] for r in extra_pool[:K_IN_CLASS]]
    print(f"[p3] 可用池 {len(extra_pool)} 只（排除现有持仓 + 流动性门槛），"
          f"ρ 最低 top-3 = {lowx}", flush=True)
    if len(lowx) == K_IN_CLASS:
        h4 = {**held, "lowrho": lowx}
        run_cfg("extra_lowrho3", h4, CLASSES + ["lowrho"],
                {"note": "新增第 4 类 = 全池 ρ 最低 top-3", "picked": lowx})
        for seed in SEEDS:
            rng = random.Random(seed)
            pick = rng.sample([r["code"] for r in extra_pool], K_IN_CLASS) \
                if len(extra_pool) >= K_IN_CLASS else []
            if len(pick) != K_IN_CLASS:
                continue
            run_cfg(f"extra_rand_s{seed}", {**held, "lowrho": pick},
                    CLASSES + ["lowrho"], {"note": "第 4 类 = 同池随机 3 只", "picked": pick})

    # ── 滚动窗口夏普胜率（对比 base）：P1 的标准判据 ──
    if "base" in navs:
        print(f"\n[p3] === 滚动夏普胜率 vs base ===", flush=True)
        for label, nav in navs.items():
            if label == "base":
                continue
            w252 = rolling_sharpe_win(navs["base"], nav, 252)
            w504 = rolling_sharpe_win(navs["base"], nav, 504)
            results[label]["rolling_sharpe_252d_win_rate"] = w252
            results[label]["rolling_sharpe_504d_win_rate"] = w504
            print(f"  {label:28s} 滚动252日胜率 {w252}  滚动504日胜率 {w504}", flush=True)

    store["results"] = results
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, ensure_ascii=False, indent=1)
    print(f"\n✓ 已写入 {out_path}", flush=True)


if __name__ == "__main__":
    main()
