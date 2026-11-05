"""零技巧基准的读取与锚定（建设项 B，2026-11-05）。

基准**电平矩阵**由 `backend/scripts/build_benchmark_index.py` 每日构建一次：
  - `data/paper/benchmark_levels.parquet`：日期 × 代码，起点日 = 1.0，后复权 pct_chg 链
  - `data/paper/benchmark_sleeve.json`：9 只定稿 ETF 等权买入持有电平
本模块只做一件事：把电平**重新锚定**到某条线自己的起点日 t0，
产出 `benchmark_return_pct` / `excess_pp` 需要的比值。

口径（与构建脚本一致，不许在调用方另起炉灶）：
  A  = mean_i [ L_i(t) / L_i(t0) ]      # 买入持有等权；i 跑遍 t0 当天有价的代码
  S  = sleeve_level(t) / sleeve_level(t0)
  母线 = 0.5·A + 0.5·S                  # 与定稿配置 50% A 股 + 50% 袖套一致
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# benchmark_service.py 在 backend/app/services/trading/ 下 → 仓库根要上四级
# （数据目录 `data/paper/` 在**仓库根**，不在 backend/ 下 —— 少一级就会去找
#  backend/data/paper 然后报"缺基准产物"，2026-11-05 实测踩过）
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PAPER_DIR = os.path.join(ROOT, "data", "paper")
LEVELS_FP = os.path.join(PAPER_DIR, "benchmark_levels.parquet")
SLEEVE_FP = os.path.join(PAPER_DIR, "benchmark_sleeve.json")
META_FP = os.path.join(PAPER_DIR, "benchmark_meta.json")

KIND_A = "a"
KIND_SLEEVE = "sleeve"
KIND_BLEND = "blend"

#: 线名 → 基准类型。新增线时必须在这里表态，不许"默认当 A 股"。
LINE_KIND: Dict[str, str] = {
    "22f": KIND_A,
    "honest_42f": KIND_A,
    "honest_42f_neu": KIND_A,
    "rsi_71421": KIND_A,
    "factor_ranking": KIND_A,
    "gbm_voltgt": KIND_A,
    "multi_asset_final_a": KIND_A,
    "multi_asset_final_sleeve": KIND_SLEEVE,
    "multi_asset_final": KIND_BLEND,
}

MIN_COVERAGE = 0.95


def benchmark_path_for(line: str) -> Optional[str]:
    kind = LINE_KIND.get(line)
    if kind is None:
        return None
    return {"a": LEVELS_FP, "sleeve": SLEEVE_FP, "blend": LEVELS_FP}[kind]


def load_benchmark(root: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    """返回 (A 电平矩阵, 袖套电平 dict, meta)。缺文件 → 抛错，不许静默降级成"没有基准"。"""
    base = root or PAPER_DIR
    levels_fp = os.path.join(base, "benchmark_levels.parquet")
    sleeve_fp = os.path.join(base, "benchmark_sleeve.json")
    meta_fp = os.path.join(base, "benchmark_meta.json")
    for fp in (levels_fp, sleeve_fp, meta_fp):
        if not os.path.exists(fp):
            raise FileNotFoundError(
                f"缺基准产物 {os.path.basename(fp)}（先跑 build_benchmark_index.py）")
    levels = pd.read_parquet(levels_fp)
    with open(sleeve_fp, encoding="utf-8") as fh:
        sleeve = json.load(fh)["levels"]
    with open(meta_fp, encoding="utf-8") as fh:
        meta = json.load(fh)
    return levels, sleeve, meta


def _iso(d: Any) -> str:
    return str(d)[:10]


def compute_benchmark(kind: str,
                      a_levels: pd.DataFrame,
                      sleeve_levels: Dict[str, float],
                      t0: str,
                      dates: List[str],
                      min_coverage: float = MIN_COVERAGE
                      ) -> Tuple[Dict[str, Optional[float]], Dict[str, Optional[float]], List[str]]:
    """把基准锚定到 t0。

    返回 (ratios, coverage, problems)：
      ratios   : 日期 → 基准净值倍数（1.0 = 与起点持平），缺基准的日期为 None
      coverage : 日期 → 当日参与均值的代码数 / t0 当天有价的代码数（A 腿口径）
      problems : 人可读的失败原因（覆盖不足、缺基准日等）
    """
    problems: List[str] = []
    ratios: Dict[str, Optional[float]] = {}
    coverage: Dict[str, Optional[float]] = {}

    t0 = _iso(t0)
    need_a = kind in (KIND_A, KIND_BLEND)
    need_s = kind in (KIND_SLEEVE, KIND_BLEND)

    base_a: Optional[pd.Series] = None
    if need_a:
        if a_levels is None or a_levels.empty:
            problems.append("A 基准电平矩阵为空")
        elif pd.Timestamp(t0) not in a_levels.index:
            problems.append(f"基准矩阵没有 {t0} 这一天（起点日缺）")
        else:
            base_a = a_levels.loc[pd.Timestamp(t0)]
            base_a = base_a[base_a.notna() & (base_a > 0)]
            if base_a.empty:
                problems.append(f"{t0} 当天没有任何代码有基准电平")

    base_s: Optional[float] = None
    if need_s:
        base_s = sleeve_levels.get(t0)
        if base_s is None or not (base_s > 0):
            problems.append(f"袖套基准没有 {t0} 的电平")

    for raw in dates:
        d = _iso(raw)
        a_val: Optional[float] = None
        s_val: Optional[float] = None

        if need_a:
            if base_a is None:
                ratios[d] = None
                continue
            ts = pd.Timestamp(d)
            if ts not in a_levels.index:
                problems.append(f"基准矩阵缺 {d}")
                ratios[d] = None
                coverage[d] = None
                continue
            row = a_levels.loc[ts].reindex(base_a.index)
            alive = row.notna() & (row > 0)
            n_alive = int(alive.sum())
            cov = n_alive / max(len(base_a.index), 1)
            coverage[d] = cov
            if cov < min_coverage:
                problems.append(
                    f"{d} 基准覆盖率 {cov:.2%} < {min_coverage:.2%}"
                    f"（基准只在 {n_alive}/{len(base_a.index)} 只幸存者上算，不能用它谈超额）")
            a_val = float((row[alive] / base_a[alive]).mean()) if n_alive else None

        if need_s:
            lv = sleeve_levels.get(d)
            if lv is None or base_s is None or not (lv > 0):
                problems.append(f"袖套基准缺 {d}")
            else:
                s_val = float(lv / base_s)

        # 母线不许在缺一条腿时"降级成 A 基准"——那等于偷偷换口径
        if (need_a and a_val is None) or (need_s and s_val is None):
            ratios[d] = None
            continue
        if kind == KIND_BLEND:
            ratios[d] = 0.5 * a_val + 0.5 * s_val      # type: ignore[operator]
        else:
            ratios[d] = a_val if need_a else s_val     # type: ignore[assignment]
    return ratios, coverage, problems


def attach_benchmark_columns(points: List[Dict[str, Any]],
                            kind: str,
                            a_levels: pd.DataFrame,
                            sleeve_levels: Dict[str, float],
                            meta: Dict[str, Any],
                            min_coverage: float = MIN_COVERAGE
                            ) -> Dict[str, Any]:
    """就地给每个净值点加 `benchmark_return_pct` / `excess_pp` / `benchmark_coverage`。

    锚点 = 本条线**最早的净值点日期**（不是本次 --range 的首日），
    这样同一根线无论什么时候重跑，超额都是"从建仓那天算起"的超额。
    返回诊断字典（供门禁/打印）。
    """
    diag: Dict[str, Any] = {"kind": kind, "problems": [], "missing_benchmark_days": 0,
                            "coverage_low_days": 0}
    if not points:
        return diag
    t0 = _iso(points[0]["date"])
    dates = [_iso(p["date"]) for p in points]
    ratios, coverage, problems = compute_benchmark(
        kind, a_levels, sleeve_levels, t0, dates, min_coverage)
    diag["anchor"] = t0
    diag["problems"] = problems
    for p in points:
        d = _iso(p["date"])
        r = ratios.get(d)
        p["benchmark_return_pct"] = None if r is None else round((r - 1.0) * 100.0, 4)
        p["benchmark_coverage"] = coverage.get(d)
        if r is None or p["return_pct"] is None:
            p["excess_pp"] = None
            diag["missing_benchmark_days"] += 1
        else:
            p["excess_pp"] = round(float(p["return_pct"]) - (r - 1.0) * 100.0, 4)
    diag["coverage_low_days"] = sum(
        1 for v in coverage.values() if v is not None and v < min_coverage)
    diag["basis"] = {"a": meta.get("basis_a"), "sleeve": meta.get("basis_sleeve"),
                     "blend": "0.5 × A 全池等权买入持有 + 0.5 × 9 只定稿 ETF 等权买入持有"}[kind]
    return diag
