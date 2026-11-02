"""Alpha 因子与 qlib 数据的 **index 对齐**门禁（§12.44）。

## 背景：生产训练路径上的真 bug

Alpha158 handler 返回的 index 是 **(datetime, instrument)**，
而 qlib 原始数据经 `_convert_to_qlib_format` 后是 **(instrument, datetime)**。

`prepare_qlib_dataset` 里直接 `pd.concat([qlib_data, alpha_factors], axis=1)`
会按 index 对齐 —— 两边键完全对不上 → pandas 造出 `('000001.SZ', '000001_SZ')`
这类垃圾键 → **159 个 Alpha 因子整列变 NaN**，
而训练流程照样"成功"（只是模型拿不到任何因子信息）。

实测（2026-11-03，1 只股票 / 60 交易日）：
- 修复前：数据集 (111, 364)，其中 60 行日期层是 `'000001_SZ'`（股票代码！），因子全 NaN
- 修复后：数据集 (60, 364)，日期层 100% Timestamp，全 NaN 因子列 = 0

## 本文件守什么

1. `_align_alpha_index` 的层级重排 + instrument 命名规范化
2. **concat 之后因子列不再是 NaN** ← 直接对着原 bug 的表现
3. 幂等：已经对齐的输入不会被改坏
4. 非 MultiIndex / 层级数不同 → 原样返回（不硬改）
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.services.qlib.enhanced_qlib_provider import EnhancedQlibDataProvider

align = EnhancedQlibDataProvider._align_alpha_index


def _target_frame() -> pd.DataFrame:
    """qlib 原始数据：(instrument, datetime)。"""
    idx = pd.MultiIndex.from_tuples(
        [("000001.SZ", pd.Timestamp("2026-02-24")),
         ("000001.SZ", pd.Timestamp("2026-02-25"))],
        names=[None, None],
    )
    return pd.DataFrame({"open": [1.0, 2.0], "close": [1.1, 2.1], "volume": [100, 200]}, index=idx)


def _alpha_frame(level_order: str = "di") -> pd.DataFrame:
    """Alpha 因子：(datetime, instrument) —— qlib handler 的真实顺序。"""
    tuples = [
        (pd.Timestamp("2026-02-24"), "000001_SZ"),
        (pd.Timestamp("2026-02-25"), "000001_SZ"),
    ]
    if level_order == "di":
        tuples = [(d, i) for i, d in tuples]
        names = ["datetime", "instrument"]
    else:
        names = ["instrument", "datetime"]
    idx = pd.MultiIndex.from_tuples(tuples, names=names)
    return pd.DataFrame({"ALPHA_A": [0.5, 0.7], "ALPHA_B": [1.5, 1.7]}, index=idx)


class TestAlignAlphaIndex:
    def test_reorders_levels_to_match_target(self):
        target = _target_frame()
        alpha = _alpha_frame("di")
        out = align(alpha, target)
        assert list(out.index.names) == list(target.index.names)
        assert out.index.get_level_values(0).tolist() == ["000001.SZ", "000001.SZ"]
        assert all(hasattr(v, "year") for v in out.index.get_level_values(1))

    def test_normalizes_instrument_names(self):
        """qlib 内部用 000001_SZ，对外用 000001.SZ。"""
        target = _target_frame()
        out = align(_alpha_frame("di"), target)
        assert set(out.index.get_level_values(0)) == {"000001.SZ"}

    def test_concat_no_longer_produces_nan_factors(self):
        """**对着原 bug 的表现**：拼接后因子列不能是 NaN。"""
        target = _target_frame()
        alpha = _alpha_frame("di")
        merged = pd.concat([target, align(alpha, target)], axis=1)
        # 不能出现垃圾键（日期层是股票代码）
        for key in merged.index:
            assert not isinstance(key[1], str), f"出现垃圾 index 键: {key}"
        # 因子列必须有值
        assert not merged[["ALPHA_A", "ALPHA_B"]].isna().all().any()

    def test_idempotent_when_already_aligned(self):
        target = _target_frame()
        once = align(_alpha_frame("di"), target)
        twice = align(once, target)
        assert list(twice.index) == list(once.index)
        assert twice.index.nlevels == once.index.nlevels

    def test_non_multiindex_returns_unchanged(self):
        target = _target_frame()
        flat = pd.DataFrame({"ALPHA_A": [1.0]}, index=pd.Index([0]))
        out = align(flat, target)
        assert out is flat

    def test_level_count_mismatch_returns_unchanged(self):
        target = _target_frame()
        alpha3 = pd.DataFrame(
            {"ALPHA_A": [1.0]},
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2026-02-24"), "000001_SZ", "x")], names=["d", "i", "x"]
            ),
        )
        out = align(alpha3, target)
        assert out is alpha3

    def test_unknown_instrument_left_alone(self):
        """目标里没有的 instrument 不能被瞎改。"""
        target = _target_frame()
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-02-24"), "999999_SZ")], names=["datetime", "instrument"]
        )
        alpha = pd.DataFrame({"ALPHA_A": [1.0]}, index=idx)
        out = align(alpha, target)
        assert out.index.get_level_values(0).tolist() == ["999999_SZ"]
