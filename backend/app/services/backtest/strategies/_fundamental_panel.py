"""财务因子 PIT 面板加载器（2026-10-27）

面板由 `backend/scripts/build_fundamental_panel.py` 生成：
npz 含 `dates` / `codes` / 每个因子一个 float32[ndates, ncodes] 矩阵。
面板本身已是**阶梯函数**（行 p 保存「从 panel_dates[p] 当天起生效的值」）。

加载时必须**按引擎自己的 master（交易日）和 codes（股票顺序）对齐**，
否则矩阵会错位 —— 引擎的 codes 顺序来自入参，不是面板的顺序。

## 对齐语义
对引擎的每个交易日 t，取**可得日 ≤ t 的最后一个面板行**。
（财务值从公布日当天起可用；引擎的 master 可能与面板日期不完全重合。）
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

_CACHE: Dict[str, dict] = {}


def _load_npz(path: str) -> dict:
    if path in _CACHE:
        return _CACHE[path]
    with np.load(path, allow_pickle=False) as z:
        data = {
            "dates": pd.DatetimeIndex(pd.to_datetime(z["dates"])),
            "codes": [str(c) for c in z["codes"]],
        }
        for key in z.files:
            if key in ("dates", "codes"):
                continue
            data[key] = z[key].astype(np.float64)
    _CACHE[path] = data
    return data


def _load_fundamental_panel(
    path: str,
    master: pd.DatetimeIndex,
    codes: List[str],
    factors,
) -> Dict[str, np.ndarray]:
    """返回 {factor: float64[ndates, nstocks]}，已对齐到 master / codes。"""
    data = _load_npz(path)
    p_dates = data["dates"]
    p_codes = data["codes"]
    p_col = {c: i for i, c in enumerate(p_codes)}

    n_d, n_s = len(master), len(codes)

    # 股票列映射：引擎 codes → 面板列
    col_map = [(j, p_col[c]) for j, c in enumerate(codes) if c in p_col]
    if not col_map:
        return {}
    js = np.array([a for a, _ in col_map], dtype=int)
    ps = np.array([b for _, b in col_map], dtype=int)

    # 日期映射：master 日 t → 可得日 ≤ t 的最后一个面板行
    # master 已排序 → searchsorted 有效
    src_idx = master.searchsorted(p_dates, side="right") - 1   # (n_d,)
    ok = src_idx >= 0

    fnames = list(factors.keys()) if isinstance(factors, dict) else list(factors)

    out: Dict[str, np.ndarray] = {}
    for f in fnames:
        mat = data.get(f)
        if mat is None:
            continue
        res = np.full((n_d, n_s), np.nan, dtype=np.float64)
        if ok.any():
            rows = src_idx[ok]
            res[np.flatnonzero(ok)[:, None], js[None, :]] = mat[rows[:, None], ps[None, :]]
        out[f] = res
    return out
