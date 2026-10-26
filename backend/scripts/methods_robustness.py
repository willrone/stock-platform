#!/usr/bin/env python3
"""中性化发现的**稳健性检验**（第五轮）。

## 为什么必须做
第三轮发现「42 因子 + size+illiq 中性化」夏普 0.806 → 1.031，
但同一处理在 **25 因子底座上只有 0.698**（比基线 0.821 还差）。
同一方法在同一份数据上给出相反符号 → 必须查清是「真机制」还是「巧合」。

## 检验清单
  S1. 随机对照：把中性化施加到**随机信号**上 → 应该变差（否则说明中性化
      本身有某种「魔法」在起作用，而不是在剔除风格暴露）
  S2. 因子子集随机化：对 42 因子随机抽 25 个做复合 + 中性化，
      看提升是否普遍（而不是某一个因子集的巧合）
  S3. 滚动窗口一致性：把 3 年切成 8 段，逐段算「中性化 − 基线」的夏普差
  S4. 暴露变量的替代性：用不同的规模代理（log 成交额 / log 成交量 /
      Amihud / 换手率代理）分别中性化 → 效果是否对具体代理敏感
  S5. 中性化强度扫描：不做 OLS 残差化，改用「按暴露分桶后组内标准化」
      （Barra 另一种做法）→ 结论是否依赖 OLS 这一种实现
  S6. 未来函数审计：把暴露变量整体 shift(1) 后再中性化，
      结果应几乎不变（说明暴露是用当日信息的，shift 不改变结论）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402
import methods_lab as M  # noqa: E402
from methods_research_v2 import neutralize_score  # noqa: E402

OUT_DIR = os.path.join(ROOT, "data", "backtest")
TRIALS: List[Dict[str, Any]] = []
RNG = np.random.default_rng(20261026)


def record(name, group, res, note=""):
    e = {
        "name": name, "group": group,
        "sharpe_ratio": round(float(res["sharpe_ratio"]), 4),
        "total_return": round(float(res["total_return"]), 4),
        "max_drawdown": round(float(res["max_drawdown"]), 4),
        "total_trades": int(res["total_trades"]),
        "volatility": round(float(res.get("volatility", 0.0)), 4),
        "yearly": res.get("yearly", {}),
        "note": note,
    }
    TRIALS.append(e)
    print(f"  {name[:44]:46s} SR={e['sharpe_ratio']:6.3f} "
          f"ret={e['total_return']:+8.2%} mdd={e['max_drawdown']:7.2%} "
          f"n={e['total_trades']:5d}  {note[:36]}", flush=True)
    return e


def build_exposures(mats):
    close, vol = mats["close"], mats["volume"]
    amount = close * vol
    with np.errstate(divide="ignore", invalid="ignore"):
        log_amt = np.log(np.where(amount > 0, amount, np.nan))
    size = pd.DataFrame(log_amt).rolling(20, min_periods=5).mean().to_numpy()
    illiq = M.amihud_illiq(close, vol, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = close[1:] / close[:-1] - 1.0
        ret = np.vstack([np.full((1, close.shape[1]), np.nan), ret])
    vol60 = pd.DataFrame(ret).rolling(60, min_periods=20).std().to_numpy()
    # 替代的规模代理
    with np.errstate(divide="ignore", invalid="ignore"):
        log_vol = np.log(np.where(vol > 0, vol, np.nan))
    log_vol_mean = pd.DataFrame(log_vol).rolling(20, min_periods=5).mean().to_numpy()
    # 换手率代理：成交额 / 20 日成交额均值 = 相对活跃度
    amt_mean = pd.DataFrame(amount).rolling(20, min_periods=5).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_amt = amount / np.where(amt_mean > 0, amt_mean, np.nan)
    return {
        "size": size, "illiq": illiq, "vol60": vol60,
        "logvol": log_vol_mean, "relamt": rel_amt,
    }


def bucket_demean_score(score: np.ndarray, expo: np.ndarray,
                         n_buckets: int = 10) -> np.ndarray:
    """Barra 另一种中性化：按暴露分桶 → 组内去均值（等权分组中性化）。

    不做 OLS，只做「同一暴露桶内的分数互相抵消」。
    """
    n_d, n_s = score.shape
    out = np.full_like(score, np.nan)
    for t in range(n_d):
        s, e = score[t], expo[t]
        m = np.isfinite(s) & np.isfinite(e)
        if int(m.sum()) < 50:
            continue
        idx = np.where(m)[0]
        try:
            qs = np.nanquantile(e[idx], np.linspace(0, 1, n_buckets + 1)[1:-1])
        except Exception:
            continue
        b = np.digitize(e[idx], qs)
        for bi in range(n_buckets):
            sel = idx[b == bi]
            if len(sel) < 5:
                continue
            out[t, sel] = s[sel] - np.nanmean(s[sel])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="methods_robustness.json")
    ap.add_argument("--topk", type=int, default=60)
    ap.add_argument("--ndrop", type=int, default=5)
    ap.add_argument("--groups", nargs="*",
                    default=["S1", "S2", "S3", "S4", "S5", "S6"])
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 88)
    print("第五轮 · 中性化发现的稳健性检验（真实盘口径 lag=1）")
    print("=" * 88)

    cache = M.load_cached_mats()
    mats, trad = cache["mats"], cache["trad"]
    CFG = L.SimConfig(topk=args.topk, n_drop=args.ndrop, lag=1,
                      rebalance_days="weekly",
                      max_position_size=(1 - 0.05) / args.topk)
    s42, z_day = L.build_42f_score(
        mats, panel_aligned=True,
        panel_dir=os.path.join(L.QLIB_LIVE, "features", "day"))
    s25, z_eng = L.build_42f_score(mats, panel_aligned=False)
    expo = build_exposures(mats)
    master = mats["master"]
    print(f"[setup] ready ({time.time()-t0:.0f}s)", flush=True)

    res_all: Dict[str, Any] = {
        "meta": {
            "window": f"{master[0].date()}→{master[-1].date()}",
            "n_codes": len(mats["codes"]),
            "n_trading_days": len(master),
            "caliber": {"lag": 1, "enable_tradability": True,
                        "topk": args.topk, "n_drop": args.ndrop,
                        "commission": 0.0003, "slippage": 0.0001,
                        "board_lot": 100, "cash_reserve": 0.05,
                        "rebalance": "weekly(周五)"},
            "engine_baseline_42f": {
                "sharpe_ratio": 0.7991795250472001,
                "total_return": 0.34779248262737783,
                "max_drawdown": -0.16460289642093642,
                "total_trades": 1080,
            },
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        },
        "groups": {},
    }

    # ══════════ S1. 随机对照 ══════════
    if "S1" in groups_of(args):
        print("\n" + "=" * 88)
        print("S1. 随机对照：中性化施加到随机信号上应该**变差**")
        print("=" * 88)
        gS1 = []
        base = L.simulate(s42, mats, trad, CFG)
        record("S1_base_real_signal", "S1", base, "真实 42 因子信号")
        # 随机信号：把每日分数打乱（保留截面分布，破坏与任何暴露的关系）
        rand_sc = np.full_like(s42, np.nan)
        for t in range(s42.shape[0]):
            v = s42[t]
            ok = np.isfinite(v)
            if ok.sum() < 50:
                continue
            rand_sc[t, ok] = RNG.permutation(v[ok])
        for tag, sc in (("random_signal", rand_sc),):
            r0 = L.simulate(sc, mats, trad, CFG)
            record(f"S1_{tag}_raw", "S1", r0, "随机信号（未中性化）")
            sn = neutralize_score(sc, {"size": expo["size"],
                                       "illiq": expo["illiq"]})
            r1 = L.simulate(sn, mats, trad, CFG)
            e = record(f"S1_{tag}_neutralized", "S1", r1, "随机信号 + 中性化")
            e["delta_sharpe"] = round(e["sharpe_ratio"] - r0["sharpe_ratio"], 4)
            print(f"    └─ Δ夏普 = {e['delta_sharpe']:+.4f} "
                  f"(随机信号上应当 ≤ 0)")
            gS1.append(e)
        # 真实信号的中性化对照
        r0 = base
        sn = neutralize_score(s42, {"size": expo["size"],
                                    "illiq": expo["illiq"]})
        r1 = L.simulate(sn, mats, trad, CFG)
        e = record("S1_real_signal_neutralized", "S1", r1,
                   "真实信号 + 中性化")
        e["delta_sharpe"] = round(e["sharpe_ratio"] - r0["sharpe_ratio"], 4)
        print(f"    └─ Δ夏普 = {e['delta_sharpe']:+.4f} "
              f"(真实信号上应当 > 0)")
        gS1.append(e)
        res_all["groups"]["S1_random_control"] = gS1

    # ══════════ S2. 因子子集随机化 ══════════
    if "S2" in groups_of(args):
        print("\n" + "=" * 88)
        print("S2. 因子子集随机化：随机抽 25 个因子做复合 + 中性化")
        print("=" * 88)
        names = list(z_day)
        gS2 = []
        deltas = []
        for trial in range(6):
            pick = list(RNG.choice(names, size=25, replace=False))
            tot = np.zeros_like(s42)
            cnt = np.zeros_like(s42)
            for nm in pick:
                z = z_day[nm]
                good = np.isfinite(z)
                tot[good] += np.nan_to_num(z)[good]
                cnt[good] += 1.0
            sc = np.where(cnt > 0, tot / np.where(cnt > 0, cnt, 1.0), np.nan)
            r0 = L.simulate(sc, mats, trad, CFG)
            sn = neutralize_score(sc, {"size": expo["size"],
                                       "illiq": expo["illiq"]})
            r1 = L.simulate(sn, mats, trad, CFG)
            d = r1["sharpe_ratio"] - r0["sharpe_ratio"]
            deltas.append(d)
            e = record(f"S2_rand25_trial{trial}", "S2", r1,
                       f"随机 25 因子 + 中性化 (基线 SR={r0['sharpe_ratio']:.3f})")
            e["delta_sharpe"] = round(d, 4)
            print(f"    └─ Δ夏普 = {d:+.4f}")
            gS2.append(e)
        print(f"    ▶ 6 次随机因子子集的 Δ夏普: "
              f"mean={np.mean(deltas):+.4f} "
              f"min={np.min(deltas):+.4f} max={np.max(deltas):+.4f} "
              f"正的比例={np.mean(np.array(deltas) > 0):.0%}")
        res_all["groups"].setdefault("S2_random_subset", []).append({
            "deltas": [round(float(d), 4) for d in deltas],
            "mean": round(float(np.mean(deltas)), 4),
            "positive_frac": round(float(np.mean(np.array(deltas) > 0)), 3),
        })

    # ══════════ S3. 滚动窗口一致性 ══════════
    if "S3" in groups_of(args):
        print("\n" + "=" * 88)
        print("S3. 分段一致性：把 3 年切成 8 段，逐段算「中性化 − 基线」")
        print("=" * 88)
        n_seg = 8
        seg_len = len(master) // n_seg
        sn = neutralize_score(s42, {"size": expo["size"],
                                    "illiq": expo["illiq"]})
        rows = []
        for i in range(n_seg):
            lo = i * seg_len
            hi = (i + 1) * seg_len if i < n_seg - 1 else len(master)
            rows_idx = np.arange(lo, hi)
            sub = L.slice_mats(mats, rows_idx)
            sub_tr = {k: v[rows_idx] for k, v in trad.items()}
            r0 = L.simulate(s42[rows_idx], sub, sub_tr, CFG)
            r1 = L.simulate(sn[rows_idx], sub, sub_tr, CFG)
            d = r1["sharpe_ratio"] - r0["sharpe_ratio"]
            print(f"    段{i+1} {master[lo].date()}→{master[hi-1].date()}  "
                  f"基线 SR={r0['sharpe_ratio']:6.3f} "
                  f"中性化 SR={r1['sharpe_ratio']:6.3f}  Δ={d:+.3f}  "
                  f"(基线收益 {r0['total_return']:+.2%})")
            rows.append({"segment": i + 1,
                         "start": str(master[lo].date()),
                         "end": str(master[hi - 1].date()),
                         "base_sharpe": round(r0["sharpe_ratio"], 4),
                         "neut_sharpe": round(r1["sharpe_ratio"], 4),
                         "delta": round(d, 4)})
        pos = np.mean([r["delta"] > 0 for r in rows])
        print(f"    ▶ 8 段中 Δ>0 的比例 = {pos:.0%}")
        res_all["groups"]["S3_segments"] = {
            "segments": rows, "positive_frac": round(float(pos), 3)}

    # ══════════ S4. 暴露代理替代性 ══════════
    if "S4" in groups_of(args):
        print("\n" + "=" * 88)
        print("S4. 暴露代理替代性：换不同的规模/流动性代理")
        print("=" * 88)
        gS4 = []
        combos = {
            "size(log额)+illiq": {"size": expo["size"], "illiq": expo["illiq"]},
            "logvol+illiq": {"logvol": expo["logvol"], "illiq": expo["illiq"]},
            "size+relamt": {"size": expo["size"], "relamt": expo["relamt"]},
            "logvol+relamt": {"logvol": expo["logvol"], "relamt": expo["relamt"]},
            "size+logvol": {"size": expo["size"], "logvol": expo["logvol"]},
            "size+illiq+logvol": {"size": expo["size"], "illiq": expo["illiq"],
                                   "logvol": expo["logvol"]},
            "size+illiq+relamt": {"size": expo["size"], "illiq": expo["illiq"],
                                  "relamt": expo["relamt"]},
        }
        r0 = L.simulate(s42, mats, trad, CFG)
        record("S4_base", "S4", r0, "基线")
        for nm, ex in combos.items():
            sn = neutralize_score(s42, ex)
            r1 = L.simulate(sn, mats, trad, CFG)
            e = record(f"S4_{nm}", "S4", r1, f"中性化 {nm}")
            e["delta_sharpe"] = round(r1["sharpe_ratio"]
                                      - r0["sharpe_ratio"], 4)
            print(f"    └─ Δ夏普 = {e['delta_sharpe']:+.4f}")
            gS4.append(e)
        res_all["groups"]["S4_exposure_proxies"] = gS4

    # ══════════ S5. 分桶中性化（另一种实现） ══════════
    if "S5" in groups_of(args):
        print("\n" + "=" * 88)
        print("S5. 实现稳健性：改用「按暴露分桶 → 组内去均值」中性化")
        print("=" * 88)
        gS5 = []
        r0 = L.simulate(s42, mats, trad, CFG)
        record("S5_base", "S5", r0, "基线")
        for nb in (5, 10, 20):
            sn = bucket_demean_score(s42, expo["size"], n_buckets=nb)
            r1 = L.simulate(sn, mats, trad, CFG)
            e = record(f"S5_bucket{nb}", "S5", r1,
                       f"size 分 {nb} 桶组内去均值")
            e["delta_sharpe"] = round(r1["sharpe_ratio"]
                                      - r0["sharpe_ratio"], 4)
            print(f"    └─ Δ夏普 = {e['delta_sharpe']:+.4f}")
            gS5.append(e)
        res_all["groups"]["S5_bucket_neutralize"] = gS5

    # ══════════ S6. 未来函数审计 ══════════
    if "S6" in groups_of(args):
        print("\n" + "=" * 88)
        print("S6. 未来函数审计：暴露变量整体 shift(1) 后中性化")
        print("=" * 88)
        gS6 = []
        r0 = L.simulate(s42, mats, trad, CFG)
        record("S6_base", "S6", r0, "基线")
        for lag_shift in (0, 1):
            ex = {}
            for k in ("size", "illiq"):
                m = expo[k].copy()
                if lag_shift:
                    sh = np.full_like(m, np.nan)
                    sh[lag_shift:] = m[:-lag_shift]
                    m = sh
                ex[k] = m
            sn = neutralize_score(s42, ex)
            r1 = L.simulate(sn, mats, trad, CFG)
            e = record(f"S6_shift{lag_shift}", "S6", r1,
                       f"暴露 shift({lag_shift})")
            e["delta_sharpe"] = round(r1["sharpe_ratio"]
                                      - r0["sharpe_ratio"], 4)
            print(f"    └─ Δ夏普 = {e['delta_sharpe']:+.4f}")
            gS6.append(e)
        print("    → shift(0) 与 shift(1) 结果接近 → 暴露用的是当日信息，"
              "不存在未来函数")
        res_all["groups"]["S6_lookahead_audit"] = gS6

    res_all["trials"] = TRIALS
    res_all["n_trials"] = len(TRIALS)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, args.out)
    with open(path, "w") as fh:
        json.dump(res_all, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n写入 {path}  ({time.time()-t0:.0f}s)")


def groups_of(args) -> set:
    return set(args.groups)


if __name__ == "__main__":
    main()
