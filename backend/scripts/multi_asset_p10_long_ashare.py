#!/usr/bin/env python3
"""P10 · A 股腿长历史重建 —— 打通被误判为"结构性缺失"的第三样本。

## 背景：P6/P7/P8/P9 全部卡在同一句话上

> "honest_composite A 股腿面板从 2021-05-11 起，没有窗口前的 A 股腿历史，
>  → 独立样本外证据结构性缺失"

**这句话是错的。** 它把"Alpha158 面板数据从 2021-05-11 起"和
"A 股腿历史从 2021-05-11 起"混为一谈：

- `honest_composite_strategy.py:199` → `self.use_panel = bool(config.get("use_panel", False))`
  → **生产默认关闭面板**，25 个因子全部由 close/high/low/volume **现场算**（`:343-346`）。
- 2021-05-11 这个边界来自 `data/qlib_data/features/day/` 的两段拼接
  （5241/5513 只 = block1 价格段 + block2 面板段拼接，见
  `docs/backtest/FULL_STRATEGY_AUDIT_2026-10-25.md` D6），
  block1 的价格段只从 2021-05-11 开始。
- 而 `data/qlib_data/features_full/`（5513 只 / 188 列 / 每股票全历史，
  由 `backend/scripts/build_full_history_features.py` 构建）**早就在本地**，
  实测 4330 只（78.5%）在 2021-05-11 之前就有数据，最早回溯到 1990-12-19。

代码清单来自 `universe_from_instruments(QLIB_OFFICIAL)`（PIT 全宇宙），
与 `features/day` 无关 → **唯一差别就是每只股票的数据覆盖**。

## 本脚本做什么

- **Phase A（验证门）**：把 loader 指向 `features_full`，在**现有 main 窗口**
  重跑 A 股腿，必须逐位复现已知基线（A腿夏普 0.8661 / 1190 笔 / 组合 1.4114）。
  不复现 → 立即停，后续阶段全部作废。
- **Phase B（长历史重建）**：在若干早期窗口跑同一套 25f 生产配置，
  得到 A 股腿的长历史夏普曲线 —— 判断"主腿弱"是**结构性**还是**近两年行情特有**。
- **Phase C（组合第三样本）**：A 股腿 + 定稿 ETF 袖套 在早期窗口合成，
  给 P6 的 w 曲线一个真正的独立样本。

## 零生产代码改动

通过 `settings.QLIB_DATA_PATH` 指向 `data/qlib_data_full/`（其
`features/day` 是指向 `qlib_data/features_full` 的符号链接），
`DataLoader.__init__` 在构造时读取该设置 → 运行时改设置即可。

## 已知口径风险（必须随结果一起报告）

1. **ST 标记是静态的**：`load_st_flags` 用 `data/stocks/st_names.json`
   的当前 ST/退市整理名单，**对全历史生效** → 长历史上存在前视排除
   （今天 ST 的股票在 2015 年也被排除）。当前窗口同样有这个问题，
   但长历史上被排除的股票占比更高。→ 必须量化。
2. **行业分类是静态的**：`load_industry_map` 用当前申万一级分类，
   对全历史生效 → 轻微前视。
3. **退市股被过滤**：`_is_data_valid` 的 `max_trailing_gap_days = 7`
   会把"数据顶不到窗口末尾"的股票判为陈旧数据而剔除 → 长历史窗口上
   退市股被系统性剔除 → 幸存者偏差（当前窗口同样存在）。

## 用法

    # Phase A 验证门
    env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
        backend/scripts/multi_asset_p10_long_ashare.py --phase gate

    # Phase B 长历史
    ... --phase long
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BACKEND_DIR)
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, os.path.join(BACKEND_DIR, "scripts"))

# ── 必须在导入任何会构造 DataLoader 的模块之前改设置 ──
from app.core.config import settings  # noqa: E402

FULL_ROOT = os.path.join(ROOT, "data", "qlib_data_full")
FEATURES_FULL = os.path.join(ROOT, "data", "qlib_data", "features_full")

OUT_DIR = os.path.join(ROOT, "data", "backtest")

#: Phase A 的基线门期望值**读取同版上游产物**，不硬编码
#: （硬编码过一次 0.8661 —— 那是 P9 自己按 nav 算的口径，不是引擎权威口径，
#:  正确的权威值是 ALL_real_lag1_v2.json 的 honest_composite.sharpe_ratio = 0.8795668740640868）
V2_PATH = os.path.join(OUT_DIR, "ALL_real_lag1_v2.json")


def load_expected_baseline() -> Dict[str, Any]:
    with open(V2_PATH, encoding="utf-8") as fh:
        v2 = json.load(fh).get("results") or {}
    base = v2.get("honest_composite") or {}
    return {
        "total_return": base.get("total_return"),
        "sharpe_ratio": base.get("sharpe_ratio"),
        "max_drawdown": base.get("max_drawdown"),
        "win_rate": base.get("win_rate"),
        "total_trades": base.get("total_trades"),
    }

#: Phase B 的长历史窗口（左闭右闭，跟随主线的 3 年切分节奏）
LONG_WINDOWS: List[Tuple[str, datetime, datetime]] = [
    ("w2013", datetime(2013, 5, 8), datetime(2016, 5, 8)),
    ("w2016", datetime(2016, 5, 8), datetime(2019, 5, 8)),
    ("w2019", datetime(2019, 5, 8), datetime(2021, 5, 11)),
    ("oos", datetime(2021, 5, 11), datetime(2023, 5, 8)),
    ("main", datetime(2023, 5, 8), datetime(2026, 5, 8)),
]


def enable_full_features() -> None:
    """把 Qlib 预计算目录切到 features_full（零生产代码改动）。

    `DataLoader.__init__` 读的是 `Path(settings.QLIB_DATA_PATH)/"features"/"day"`，
    所以这里造一个 `data/qlib_data_full/features/day` -> `qlib_data/features_full`
    的符号链接布局，运行时只改 settings。链接不存在会自动建。
    """
    if not os.path.isdir(FEATURES_FULL):
        raise SystemExit(f"features_full 不存在: {FEATURES_FULL}")
    link = os.path.join(FULL_ROOT, "features", "day")
    os.makedirs(os.path.dirname(link), exist_ok=True)
    if os.path.islink(link):
        if os.path.realpath(link) != os.path.realpath(FEATURES_FULL):
            os.unlink(link)
            os.symlink(FEATURES_FULL, link)
    elif not os.path.exists(link):
        os.symlink(FEATURES_FULL, link)
    settings.QLIB_DATA_PATH = FULL_ROOT


def sharpe(nav: List[Tuple[str, float]]) -> float:
    if len(nav) < 3:
        return 0.0
    rets = []
    for i in range(1, len(nav)):
        p0, p1 = nav[i - 1][1], nav[i][1]
        if p0 and p0 > 0:
            rets.append(p1 / p0 - 1.0)
    if len(rets) < 3:
        return 0.0
    mu = statistics.fmean(rets)
    sd = statistics.pstdev(rets)
    return mu / sd * math.sqrt(244.0) if sd > 0 else 0.0


import math  # noqa: E402


def maxdd(nav: List[Tuple[str, float]]) -> float:
    peak = None
    mdd = 0.0
    for _, v in nav:
        if peak is None or v > peak:
            peak = v
        if peak and peak > 0:
            dd = v / peak - 1.0
            if dd < mdd:
                mdd = dd
    return mdd


def load_universe() -> List[str]:
    from verify_methods_engine import QLIB_OFFICIAL, universe_from_instruments

    return universe_from_instruments(QLIB_OFFICIAL)


def run_leg(codes: List[str], start: datetime, end: datetime) -> Tuple[Dict[str, Any], List[Tuple[str, float]]]:
    """跑 A 股腿（显式传窗口，不依赖模块全局）。"""
    import multi_asset_leg_experiment as M

    M.START = start
    M.END = end
    rec, nav = M.run_a_share_leg(codes)
    return rec, nav


def phase_gate(codes: List[str]) -> Dict[str, Any]:
    """Phase A：用 features_full 在现有 main 窗口复现已知基线。"""
    enable_full_features()
    start = datetime(2023, 5, 8)
    end = datetime(2026, 5, 8)
    print(f"[p10] 验证门：features_full 跑 main 窗口 {start.date()}~{end.date()}", flush=True)
    rec, nav = run_leg(codes, start, end)
    exp = load_expected_baseline()
    got = {
        "total_return": rec.get("engine_total_return"),
        "sharpe_ratio": rec.get("engine_sharpe"),
        "max_drawdown": rec.get("engine_max_drawdown"),
        "win_rate": rec.get("engine_win_rate"),
        "total_trades": int(rec.get("total_trades") or 0),
        "nav_points": len(nav),
    }
    diffs: Dict[str, Any] = {}
    for k in ("total_return", "sharpe_ratio", "max_drawdown", "win_rate"):
        a, b = got.get(k), exp.get(k)
        if a is None or b is None:
            continue
        diffs[k] = float(a) - float(b)
    diffs["total_trades"] = got["total_trades"] - int(exp.get("total_trades") or 0)
    # 门限：四个引擎指标全部逐位相等（diff == 0）且交易笔数一致
    ok = all(abs(v) == 0.0 for v in diffs.values())
    out = {
        "phase": "gate",
        "features_full": FEATURES_FULL,
        "window": [str(start.date()), str(end.date())],
        "expected": exp,
        "got": got,
        "diff": diffs,
        "passed": bool(ok),
        "rec": rec,
    }
    path = os.path.join(OUT_DIR, "multi_asset_p10_gate.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[p10] 基线复现门 {'✅ 通过' if ok else '❌ 失败'}", flush=True)
    print(f"[p10]   期望 {exp}", flush=True)
    print(f"[p10]   实得 {got}", flush=True)
    print(f"[p10]   逐项 diff {diffs}", flush=True)
    print(f"[p10]   写出 {path}", flush=True)
    return out


def phase_long(codes: List[str], only: str = "") -> Dict[str, Any]:
    """Phase B：在早期窗口跑同一套 25f 生产配置。"""
    enable_full_features()
    out_path = os.path.join(OUT_DIR, "multi_asset_p10_long.json")
    results: Dict[str, Any] = {}
    if only and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            results = (json.load(fh).get("results") or {})
    for label, start, end in LONG_WINDOWS:
        if only and label != only:
            continue
        print(f"[p10] 长历史窗口 {label}: {start.date()}~{end.date()}", flush=True)
        rec, nav = run_leg(codes, start, end)
        results[label] = {
            "window": [str(start.date()), str(end.date())],
            "a_share_sharpe": round(float(rec.get("engine_sharpe") or 0.0), 4),
            "total_return": rec.get("engine_total_return"),
            "max_drawdown": rec.get("engine_max_drawdown"),
            "total_trades": int(rec.get("total_trades") or 0),
            "nav_points": len(nav),
            "nav_head": nav[:2],
            "nav_tail": nav[-2:],
        }
        print(f"[p10]   A腿夏普 {results[label]['a_share_sharpe']} / "
              f"交易 {results[label]['total_trades']} 笔 / 净值点 {results[label]['nav_points']}", flush=True)
    out = {"phase": "long", "features_full": FEATURES_FULL, "results": results}
    path = out_path
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"[p10] 写出 {path}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["gate", "long"], default="gate")
    ap.add_argument("--only", default="", help="只跑指定窗口（其余沿用已有产物）")
    args = ap.parse_args()
    codes = load_universe()
    print(f"[p10] 宇宙 {len(codes)} 只", flush=True)
    if args.phase == "gate":
        out = phase_gate(codes)
        return 0 if out["passed"] else 1
    phase_long(codes, only=args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
