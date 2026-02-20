"""
可复现诊断脚本：追踪 EnhancedQlibDataProvider + prepare_training_datasets 全管道

配置与 e2e 一致：
- 10 只股票、2024 全年、alpha158、ratio split、val=0.2、h=5、CSRankNorm=OFF

在每个关键节点打印：
  行数、unique index 数、duplicated index 数、instrument 集合大小、datetime 集合大小

用法:
    cd backend && source venv/bin/activate
    python tests/scripts/diagnose_dataset_pipeline.py
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

# ── 配置 ──────────────────────────────────────────────
STOCK_CODES = [
    "000001.SZ", "000002.SZ", "000063.SZ", "000100.SZ", "000157.SZ",
    "600036.SH", "600050.SH", "600519.SH", "601318.SH", "601398.SH",
]
START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2024, 12, 31)
VALIDATION_SPLIT = 0.2
PREDICTION_HORIZON = 5
EMBARGO_DAYS = 0  # 与 e2e 一致
FEATURE_SET = "alpha158"
SPLIT_METHOD = "ratio"
ENABLE_CS_RANK_NORM = False


def diag(tag: str, df: pd.DataFrame) -> None:
    """在关键节点打印诊断信息"""
    n_rows = len(df)
    n_cols = len(df.columns) if hasattr(df, "columns") else 0

    if isinstance(df.index, pd.MultiIndex):
        n_unique_idx = len(df.index.unique())
        n_dup_idx = n_rows - n_unique_idx
        level0 = df.index.get_level_values(0)
        level1 = df.index.get_level_values(1)
        # 判断哪个 level 是 instrument，哪个是 datetime
        if pd.api.types.is_datetime64_any_dtype(level0.dtype):
            dt_level, inst_level = level0, level1
        else:
            inst_level, dt_level = level0, level1
        n_instruments = inst_level.nunique()
        n_dates = dt_level.nunique()
        expected_rows = n_instruments * n_dates
        idx_names = df.index.names
    else:
        n_unique_idx = len(df.index.unique())
        n_dup_idx = n_rows - n_unique_idx
        n_instruments = "N/A"
        n_dates = "N/A"
        expected_rows = "N/A"
        idx_names = [df.index.name]

    has_label = "label" in df.columns if hasattr(df, "columns") else False
    label_info = ""
    if has_label:
        lbl = df["label"]
        label_info = (
            f"  label: notna={lbl.notna().sum()}, "
            f"range=[{lbl.min():.6f}, {lbl.max():.6f}]"
        )

    print(f"\n{'='*60}")
    print(f"[{tag}]")
    print(f"  rows={n_rows}, cols={n_cols}, index_names={idx_names}")
    print(f"  unique_idx={n_unique_idx}, dup_idx={n_dup_idx}")
    print(f"  instruments={n_instruments}, dates={n_dates}, expected={expected_rows}")
    if n_dup_idx > 0:
        print(f"  ⚠️  DUPLICATED INDEX: {n_dup_idx} rows")
    if isinstance(expected_rows, int) and n_rows != expected_rows:
        ratio = n_rows / expected_rows if expected_rows > 0 else float("inf")
        print(f"  ⚠️  ROW MISMATCH: actual/expected = {ratio:.2f}x")
    if label_info:
        print(label_info)
    # 打印前 3 个 index 值
    print(f"  index[:3] = {list(df.index[:3])}")
    print(f"{'='*60}")


async def run_diagnosis():
    """运行全管道诊断"""
    # ── 导入 ──
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.services.qlib.enhanced_qlib_provider import EnhancedQlibDataProvider
    from app.services.qlib.training.config import QlibModelType, QlibTrainingConfig
    from app.services.qlib.training.dataset_preparation import (
        _extract_sorted_dates,
        _split_by_ratio,
        prepare_training_datasets,
        process_stock_data,
    )

    provider = EnhancedQlibDataProvider()

    # ── Step 1: 初始化 Qlib ──
    print("\n" + "▶" * 60)
    print("Step 1: 初始化 Qlib")
    await provider.initialize_qlib()

    # ── Step 2: 准备基础特征 ──
    print("\n" + "▶" * 60)
    print("Step 2: _prepare_base_features")
    base_features = await provider._prepare_base_features(
        STOCK_CODES, START_DATE, END_DATE,
    )
    diag("base_features (raw from _prepare_base_features)", base_features)

    # ── Step 3: QlibFormatConverter ──
    print("\n" + "▶" * 60)
    print("Step 3: QlibFormatConverter.convert")
    from app.services.qlib.converters import QlibFormatConverter
    converter = QlibFormatConverter()
    qlib_data = converter.convert(
        base_features, date_column="date", instrument_column="stock_code",
    )
    diag("qlib_data (after format conversion)", qlib_data)

    # ── Step 4: 缺失值处理 ──
    from app.services.qlib.data_processing import MissingValueHandler
    handler = MissingValueHandler()
    qlib_data = handler.handle(qlib_data)
    diag("qlib_data (after missing value handling)", qlib_data)

    # ── Step 5: Alpha158 因子计算 ──
    print("\n" + "▶" * 60)
    print("Step 5: Alpha158 因子计算")
    alpha_factors = await provider.alpha_calculator.calculate_alpha_factors(
        qlib_data, STOCK_CODES, (START_DATE, END_DATE), use_cache=False,
    )
    diag("alpha_factors (raw from Alpha158)", alpha_factors)

    # ── Step 5a: 检查 index level 结构 ──
    print("\n" + "▶" * 60)
    print("Step 5a: Index level 对比 (raw, before alignment)")
    print(f"  qlib_data.index.names = {qlib_data.index.names}")
    print(f"  alpha_factors.index.names = {alpha_factors.index.names}")
    from app.services.qlib.enhanced_qlib_provider import _identify_index_levels
    q_inst, q_dt = _identify_index_levels(qlib_data)
    a_inst, a_dt = _identify_index_levels(alpha_factors)
    print(f"  qlib: inst@level{q_inst}, dt@level{q_dt}")
    print(f"  alpha: inst@level{a_inst}, dt@level{a_dt}")
    # 检查 instrument 值交集
    q_instruments = set(qlib_data.index.get_level_values(q_inst).unique())
    a_instruments = set(alpha_factors.index.get_level_values(a_inst).unique())
    overlap = q_instruments & a_instruments
    print(f"  instrument overlap: {len(overlap)}/{min(len(q_instruments), len(a_instruments))}")
    print(f"  qlib instruments[:3]: {list(q_instruments)[:3]}")
    print(f"  alpha instruments[:3]: {list(a_instruments)[:3]}")

    # ── Step 5b: raw concat (without fix) to show the problem ──
    print("\n" + "▶" * 60)
    print("Step 5b: raw pd.concat (NO alignment, showing the bug)")
    before_rows = len(qlib_data)
    combined_raw = pd.concat([qlib_data, alpha_factors], axis=1)
    diag("combined_raw (no alignment)", combined_raw)
    if len(combined_raw) > before_rows:
        print(f"  ⚠️  INFLATION: {before_rows} → {len(combined_raw)} (+{len(combined_raw) - before_rows})")
        print(f"  inflation ratio: {len(combined_raw) / before_rows:.2f}x")

    # ── Step 5d: 用 _add_alpha_factors 走完整路径 ──
    print("\n" + "▶" * 60)
    print("Step 5d: _add_alpha_factors (完整路径，含修复逻辑)")
    # 重新获取 qlib_data（未被 concat 污染）
    qlib_data_clean = converter.convert(
        base_features, date_column="date", instrument_column="stock_code",
    )
    qlib_data_clean = handler.handle(qlib_data_clean)
    qlib_data_with_alpha = await provider._add_alpha_factors(
        qlib_data_clean, STOCK_CODES, START_DATE, END_DATE, use_cache=True,
    )
    diag("qlib_data_with_alpha (after _add_alpha_factors)", qlib_data_with_alpha)

    # ── Step 6: prepare_training_datasets ──
    print("\n" + "▶" * 60)
    print("Step 6: prepare_training_datasets")
    config = QlibTrainingConfig(
        model_type=QlibModelType.LIGHTGBM,
        hyperparameters={},
        prediction_horizon=PREDICTION_HORIZON,
        validation_split=VALIDATION_SPLIT,
        embargo_days=EMBARGO_DAYS,
        feature_set=FEATURE_SET,
        split_method=SPLIT_METHOD,
        enable_cs_rank_norm=ENABLE_CS_RANK_NORM,
        enable_neutralization=False,  # 简化诊断
    )

    # 6a: 先手动走 swaplevel + process_stock_data 看中间状态
    dataset = qlib_data_with_alpha.copy()
    diag("6a: dataset before swaplevel", dataset)

    # 检查是否需要 swaplevel
    if isinstance(dataset.index, pd.MultiIndex) and dataset.index.nlevels == 2:
        level0_dtype = dataset.index.get_level_values(0).dtype
        level1_dtype = dataset.index.get_level_values(1).dtype
        print(f"\n  level0_dtype={level0_dtype}, level1_dtype={level1_dtype}")
        if (
            pd.api.types.is_datetime64_any_dtype(level0_dtype)
            and not pd.api.types.is_datetime64_any_dtype(level1_dtype)
        ):
            print("  → swaplevel needed (datetime, instrument) → (instrument, datetime)")
            dataset = dataset.swaplevel().sort_index()
            diag("6b: dataset after swaplevel", dataset)
        else:
            print("  → swaplevel NOT needed")

    # 6c: process_stock_data (per stock)
    stock_codes_in_data = dataset.index.get_level_values(0).unique()
    print(f"\n  Stocks in data: {list(stock_codes_in_data)}")
    processed_parts = []
    for sc in stock_codes_in_data:
        stock_data = dataset.xs(sc, level=0, drop_level=False)
        processed = process_stock_data(stock_data, sc, PREDICTION_HORIZON)
        processed_parts.append(processed)
    processed_dataset = pd.concat(processed_parts)
    diag("6c: after process_stock_data", processed_dataset)

    # 6d: dropna label
    before_drop = len(processed_dataset)
    processed_dataset = processed_dataset.dropna(subset=["label"])
    diag("6d: after dropna(label)", processed_dataset)
    print(f"  dropped: {before_drop - len(processed_dataset)} rows")

    # 6e: split
    print("\n" + "▶" * 60)
    print("Step 6e: _split_by_ratio")
    print(f"  validation_split={VALIDATION_SPLIT}, embargo_days={EMBARGO_DAYS}")
    dates = _extract_sorted_dates(processed_dataset)
    split_idx = int(len(dates) * (1 - VALIDATION_SPLIT))
    print(f"  total dates={len(dates)}, split_idx={split_idx}")
    print(f"  train dates: {dates[0]} ~ {dates[split_idx-1]}")
    print(f"  val dates: {dates[split_idx]} ~ {dates[-1]}")

    train_data, val_data = _split_by_ratio(
        processed_dataset, VALIDATION_SPLIT, EMBARGO_DAYS,
    )
    diag("6e: train_data", train_data)
    diag("6e: val_data", val_data)

    total = len(train_data) + len(val_data)
    print(f"\n  Split ratio: train={len(train_data)/total:.1%}, val={len(val_data)/total:.1%}")
    # 检查日期不重叠
    train_dates = set(train_data.index.get_level_values(1).unique())
    val_dates = set(val_data.index.get_level_values(1).unique())
    overlap = train_dates & val_dates
    print(f"  Date overlap: {len(overlap)} dates")
    if overlap:
        print(f"  ⚠️  OVERLAP DATES: {sorted(overlap)[:5]}")

    # ── Step 7: 完整 prepare_training_datasets ──
    print("\n" + "▶" * 60)
    print("Step 7: 完整 prepare_training_datasets 调用")
    train_ds, val_ds = await prepare_training_datasets(
        qlib_data_with_alpha, VALIDATION_SPLIT, config,
    )
    print(f"  train_dataset len={len(train_ds)}")
    print(f"  val_dataset len={len(val_ds)}")
    total_ds = len(train_ds) + len(val_ds)
    if total_ds > 0:
        print(f"  ratio: train={len(train_ds)/total_ds:.1%}, val={len(val_ds)/total_ds:.1%}")
    else:
        print("  ⚠️  EMPTY DATASETS")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    n_stocks = len(STOCK_CODES)
    print(f"  Config: {n_stocks} stocks, 2024 full year, alpha158, ratio split, val={VALIDATION_SPLIT}")
    print(f"  Base features: {len(base_features)} rows")
    print(f"  After format conversion: {len(qlib_data)} rows")
    print(f"  Alpha158 factors: {len(alpha_factors)} rows")
    print(f"  After _add_alpha_factors: {len(qlib_data_with_alpha)} rows")
    print(f"  After process_stock_data: {len(processed_dataset)} rows (post dropna)")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")
    if total_ds > 0:
        print(f"  Split: {len(train_ds)/total_ds:.1%} / {len(val_ds)/total_ds:.1%}")


if __name__ == "__main__":
    asyncio.run(run_diagnosis())
