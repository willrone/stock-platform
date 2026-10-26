#!/usr/bin/env python3
"""快速模拟器保真度验收：与真实引擎逐项对比。

用法:
  env -u PYTHONPATH -u PYTHONHOME backend/.venv-py313/bin/python3 \
    backend/scripts/ensemble_lab_selftest.py

对比对象: verify_10y_pit.py --years 3 --lag 1 --only honest_composite
  → sharpe 0.8795668740640868, ret +35.51%, mdd -13.37%, trades 1190
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

BACKEND = "/Users/rone/Projects/stock-platform/backend"
ROOT = "/Users/rone/Projects/stock-platform"
sys.path.insert(0, BACKEND)
sys.path.insert(0, os.path.join(BACKEND, "scripts"))

import ensemble_lab as L  # noqa: E402

ENGINE_REF = {
    "sharpe_ratio": 0.8795668740640868,
    "total_return": 0.35505797455182114,
    "max_drawdown": -0.1337121157482497,
    "total_trades": 1190,
}
TOL = {"sharpe_ratio": 0.06, "total_return": 0.035,
       "max_drawdown": 0.035, "total_trades": 90}


def main() -> None:
    t0 = time.time()
    codes = L.universe_from_instruments()
    print(f"[selftest] universe {len(codes)}", flush=True)
    mats = L.load_matrices(codes)
    print(f"[selftest] loaded in {time.time()-t0:.0f}s", flush=True)

    # 引擎的 honest_composite 在 CWD=仓库根 时 panel_dir 解析失败 →
    # 17 个面板因子全是 NaN → 实际只有 25 个公式因子参与。
    # 这里用 panel_aligned=False 复现"引擎真实行为"（用仓库根的相对路径语义）。
    print("[selftest] building 42f score (panel disabled, matches engine CWD=root)",
          flush=True)
    score, _ = L.build_42f_score(mats, panel_aligned=False)
    print(f"[selftest] score built {time.time()-t0:.0f}s", flush=True)

    print("[selftest] building tradability matrix ...", flush=True)
    trad = L.build_tradability_mats(mats)
    print(f"[selftest] tradability built {time.time()-t0:.0f}s", flush=True)

    cfg = L.SimConfig(topk=60, n_drop=5, lag=1, rebalance_days="weekly")
    res = L.simulate(score, mats, trad, cfg)
    print("\n=== 模拟器 vs 真实引擎 ===")
    print(f"{'metric':16s} {'sim':>12s} {'engine':>12s} {'diff':>10s}  ok")
    all_ok = True
    for k, ref in ENGINE_REF.items():
        got = res[k]
        diff = abs(float(got) - float(ref))
        ok = diff <= TOL[k]
        all_ok = all_ok and ok
        print(f"{k:16s} {got:12.4f} {ref:12.4f} {diff:10.4f}  {'OK' if ok else 'FAIL'}")
    print("\nyearly (sim):", res["yearly"])
    print("yearly (engine): {'2024': 0.160944, '2025': 0.090579, '2026': 0.046826}")

    out = {
        "sim": {k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in res.items()},
        "engine": ENGINE_REF,
        "tolerance": TOL,
        "all_ok": bool(all_ok),
        "wall_s": round(time.time() - t0, 1),
    }
    path = os.path.join(L.OUT_DIR, "ensemble_selftest.json")
    with open(path, "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print(f"\n[selftest] wrote {path}")
    print(f"[selftest] ALL_OK={all_ok}  ({out['wall_s']}s)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
