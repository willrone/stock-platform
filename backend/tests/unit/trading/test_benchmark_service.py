"""零技巧基准锚定/口径测试（建设项 B，2026-11-05）。

锁的是**口径**，不是小数点：
1. A 基准 = mean_i[L_i(t)/L_i(t0)]（买入持有等权），**不是**日均收益连乘（日频再平衡等权）；
2. 母线 = 0.5·A + 0.5·袖套，缺一条腿时**不许**降级成"A 基准"；
3. 锚点 = 这条线最早的净值点日期，不是本次 `--range` 首日；
4. 覆盖率不足必须留下可审计的失败原因（基准只剩幸存者子集时不许谈超额）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "backend"))

from app.services.trading import benchmark_service as bsvc  # noqa: E402

D0, D1, D2 = "2026-11-02", "2026-11-03", "2026-11-04"


def _levels(frame: dict) -> pd.DataFrame:
    """frame: {code: [L(D0), L(D1), ...]} → 日期 × 代码 电平矩阵。

    短于 3 天的用例 = 该矩阵在更晚的日期上**没有数据**（用 NaN 补齐行，
    等价于"基准矩阵缺那一天"）。
    """
    n = max(len(v) for v in frame.values())
    idx = pd.to_datetime([D0, D1, D2][:n])
    padded = {k: list(v) + [np.nan] * (n - len(v)) for k, v in frame.items()}
    return pd.DataFrame(padded, index=idx).astype("float32")


# ── 1. 买入持有等权，不是日频再平衡等权 ────────────────────────────────────
def test_a_benchmark_is_buy_and_hold_equal_weight():
    # 三只：A 翻倍、B 不变、C 减半 → 买入持有等权末值 = (2 + 1 + 0.5)/3 = 1.16667
    lv = _levels({"A": [1.0, 1.5, 2.0], "B": [1.0, 1.0, 1.0], "C": [1.0, 0.5, 0.5]})
    ratios, cov, problems = bsvc.compute_benchmark(bsvc.KIND_A, lv, {}, D0, [D0, D1, D2])
    assert ratios[D0] == pytest.approx(1.0)
    assert ratios[D1] == pytest.approx((1.5 + 1.0 + 0.5) / 3)
    assert ratios[D2] == pytest.approx((2.0 + 1.0 + 0.5) / 3)
    assert not problems
    assert all(v == pytest.approx(1.0) for v in cov.values())


def test_buy_and_hold_equal_weight_differs_from_daily_rebalanced_equal_weight():
    """如果哪天有人把基准改成"每日均值连乘"，这条测试必须变红。"""
    lv = _levels({"A": [1.0, 1.5, 2.0], "B": [1.0, 1.0, 1.0], "C": [1.0, 0.5, 0.5]})
    ratios, _, _ = bsvc.compute_benchmark(bsvc.KIND_A, lv, {}, D0, [D0, D1, D2])
    daily_rebalanced = (1 + (1.5 + 1.0 + 0.5) / 3 - 1) * (1 + (2 / 1.5 + 1 + 1) / 3 - 1)
    assert ratios[D2] != pytest.approx(daily_rebalanced, rel=1e-3)
    assert ratios[D2] == pytest.approx(7 / 6)


# ── 2. 母线必须两腿齐全，缺腿不许降级 ─────────────────────────────────────
def test_blend_is_half_a_half_sleeve():
    lv = _levels({"A": [1.0, 1.2], "B": [1.0, 0.8]})          # A 基准 D1 = 1.0
    sleeve = {D0: 1.0, D1: 1.10}
    ratios, _, problems = bsvc.compute_benchmark(bsvc.KIND_BLEND, lv, sleeve, D0, [D0, D1])
    assert ratios[D1] == pytest.approx(0.5 * 1.0 + 0.5 * 1.10)
    assert not problems


def test_blend_with_missing_sleeve_day_is_none_not_a_only():
    lv = _levels({"A": [1.0, 1.2]})
    ratios, _, problems = bsvc.compute_benchmark(
        bsvc.KIND_BLEND, lv, {D0: 1.0}, D0, [D0, D1])       # 袖套缺 D1
    assert ratios[D1] is None, "缺一条腿时降级成 A 基准 = 偷偷换口径"
    assert any("袖套基准缺" in p for p in problems)


# ── 3. 锚点 = 这条线最早的净值点（不是本次窗口首日） ──────────────────────
def test_anchor_is_first_nav_point_not_window_start():
    lv = _levels({"A": [1.0, 1.2, 1.5]})
    points = [{"date": D1, "value": 900.0, "return_pct": -10.0},
              {"date": D2, "value": 930.0, "return_pct": -7.0}]
    diag = bsvc.attach_benchmark_columns(points, bsvc.KIND_A, lv, {},
                                         {"basis_a": "x", "basis_sleeve": "y"})
    assert diag["anchor"] == D1
    # 基准从 D1 起算：D2 基准 = 1.5/1.2 = 1.25 → +25%；超额 = -7 - 25 = -32pp
    assert points[1]["benchmark_return_pct"] == pytest.approx(25.0, abs=0.01)
    assert points[1]["excess_pp"] == pytest.approx(-32.0, abs=0.01)


# ── 4. 覆盖率：基准塌成幸存者子集时必须报失败，不许静默算 ──────────────────
def test_coverage_drop_is_reported():
    lv = _levels({"A": [1.0, 1.2], "B": [1.0, 1.1], "C": [1.0, np.nan]})
    ratios, cov, problems = bsvc.compute_benchmark(bsvc.KIND_A, lv, {}, D0, [D0, D1])
    assert cov[D1] == pytest.approx(2 / 3)
    assert problems and "覆盖率" in problems[0]
    # 锚点 = 最早的净值点 D0 → 池子按 D0 定义（3 只）；D1 只剩 2 只 → 必须报
    points = [{"date": D0, "value": 1.0, "return_pct": 0.0},
              {"date": D1, "value": 1.05, "return_pct": 5.0}]
    diag = bsvc.attach_benchmark_columns(points, bsvc.KIND_A, lv, {},
                                         {"basis_a": "x", "basis_sleeve": "y"})
    assert diag["anchor"] == D0
    assert diag["coverage_low_days"] == 1


def test_missing_anchor_day_kills_the_benchmark():
    lv = _levels({"A": [1.0, 1.2]})       # 只有 D0/D1，锚却是 D2
    ratios, _, problems = bsvc.compute_benchmark(bsvc.KIND_A, lv, {}, D2, [D2])
    assert ratios[D2] is None and problems


# ── 5. 每条线都必须表态用哪种基准（新线不许默认当 A 股） ──────────────────
def test_every_daily_line_declares_its_benchmark_kind():
    daily_lines = ["22f", "honest_42f", "honest_42f_neu", "rsi_71421",
                   "factor_ranking", "gbm_voltgt",
                   "multi_asset_final_a", "multi_asset_final_sleeve",
                   "multi_asset_final"]
    missing = [s for s in daily_lines if s not in bsvc.LINE_KIND]
    assert not missing, f"这些线没声明基准口径：{missing}"
    assert bsvc.LINE_KIND["multi_asset_final"] == bsvc.KIND_BLEND
    assert bsvc.LINE_KIND["multi_asset_final_sleeve"] == bsvc.KIND_SLEEVE


# ── 6. 缺基准产物必须抛错，不许静默变成"没有基准" ─────────────────────────
def test_load_benchmark_raises_when_products_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        bsvc.load_benchmark(root=str(tmp_path))


def test_attach_marks_points_when_benchmark_missing_for_that_day():
    lv = _levels({"A": [1.0, 1.2]})
    points = [{"date": D0, "value": 100.0, "return_pct": 0.0},
              {"date": D2, "value": 105.0, "return_pct": 5.0}]      # 矩阵没有 D2
    diag = bsvc.attach_benchmark_columns(points, bsvc.KIND_A, lv, {},
                                         {"basis_a": "x", "basis_sleeve": "y"})
    assert points[1]["benchmark_return_pct"] is None
    assert points[1]["excess_pp"] is None
    assert diag["missing_benchmark_days"] == 1
