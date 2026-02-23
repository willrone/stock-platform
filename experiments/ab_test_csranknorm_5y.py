#!/usr/bin/env python3
"""
A/B 对比实验：CSRankNorm 5 年滚动验证

使用 expanding window 逐年预测，输出：
  - Overall RankIC / IC（两组）
  - 每年 RankIC / IC（稳定性分析）

两组对比：
  Baseline — 新参数(reg=5) + 原始标签
  CSRankNorm — 新参数(reg=200) + CSRankNorm 标签
"""
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from unified.config import LGBConfig, TrainingConfig
from unified.constants import RANDOM_SEED
from unified.data_loader import prepare_dataset
from unified.features import get_feature_columns

np.random.seed(RANDOM_SEED)

# === 实验常量 ===
DATA_START = "2019-06-01"  # 留半年给特征 warmup（60 日均线等）
DATA_END = "2025-02-01"
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]
EMBARGO_DAYS = 20
VAL_MONTHS = 6  # 验证集长度（月）
N_STOCKS = 350


@dataclass
class YearlyResult:
    """单年度结果"""

    year: int
    ic: float
    rank_ic: float
    n_samples: int


@dataclass
class GroupResult:
    """一组实验的完整结果"""

    name: str
    yearly: List[YearlyResult]
    overall_ic: float
    overall_rank_ic: float
    total_samples: int
    duration: float


def compute_rank_ic(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """计算 RankIC（Spearman 相关系数）"""
    if len(predictions) < 2:
        return 0.0
    corr, _ = spearmanr(predictions, actuals)
    return float(corr) if np.isfinite(corr) else 0.0


def compute_ic(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """计算 IC（Pearson 相关系数）"""
    if len(predictions) < 2:
        return 0.0
    corr = np.corrcoef(predictions, actuals)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    lgb_config: LGBConfig,
    config: TrainingConfig,
):
    """训练 LightGBM 模型"""
    import lightgbm as lgb

    feature_cols = get_feature_columns()
    X_train = train_df[feature_cols]
    y_train = train_df["future_return"]
    X_val = val_df[feature_cols]
    y_val = val_df["future_return"]

    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)

    model = lgb.train(
        lgb_config.to_dict(),
        train_data,
        num_boost_round=config.max_boost_rounds,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds),
            lgb.log_evaluation(500),
        ],
    )
    return model


def split_year_data(
    data: pd.DataFrame, test_year: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    为指定测试年份做 expanding window 分割

    train: 所有 < (test_year - val_months) 的数据
    val: train_end + embargo ~ test_year-01-01
    test: test_year-01-01 + embargo ~ test_year+1-01-01
    """
    embargo = pd.Timedelta(days=EMBARGO_DAYS)
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year + 1}-01-01")

    # 验证集：test_year 前 VAL_MONTHS 个月
    val_start_raw = test_start - pd.DateOffset(months=VAL_MONTHS)
    train_end = val_start_raw - embargo

    train = data[data["date"] < train_end].copy()
    val = data[
        (data["date"] >= val_start_raw) & (data["date"] < test_start)
    ].copy()
    test = data[
        (data["date"] >= test_start + embargo) & (data["date"] < test_end)
    ].copy()

    return train, val, test


def run_group_experiment(
    name: str,
    data: pd.DataFrame,
    lgb_config: LGBConfig,
    config: TrainingConfig,
) -> GroupResult:
    """
    运行一组实验：逐年 expanding window 训练 + 预测

    Returns:
        GroupResult 包含逐年和总体指标
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"实验组: {name}")
    logger.info(f"{'='*60}")

    start_time = time.time()
    feature_cols = get_feature_columns()
    yearly_results: List[YearlyResult] = []
    all_preds = []
    all_actuals = []

    for year in TEST_YEARS:
        train, val, test = split_year_data(data, year)

        if len(train) < 1000 or len(val) < 100 or len(test) < 100:
            logger.warning(
                f"  {year}: 数据不足 "
                f"(train={len(train)}, val={len(val)}, test={len(test)}), 跳过"
            )
            continue

        logger.info(
            f"  {year}: train={len(train)}, val={len(val)}, test={len(test)}"
        )

        model = train_model(train, val, lgb_config, config)
        preds = model.predict(test[feature_cols])
        actuals = test["future_return"].values

        ic = compute_ic(preds, actuals)
        rank_ic = compute_rank_ic(preds, actuals)

        yearly_results.append(YearlyResult(year, ic, rank_ic, len(test)))
        all_preds.append(preds)
        all_actuals.append(actuals)

        logger.info(
            f"  {year}: IC={ic:.4f}, RankIC={rank_ic:.4f}, N={len(test)}"
        )

    # 总体指标
    if all_preds:
        combined_preds = np.concatenate(all_preds)
        combined_actuals = np.concatenate(all_actuals)
        overall_ic = compute_ic(combined_preds, combined_actuals)
        overall_rank_ic = compute_rank_ic(combined_preds, combined_actuals)
        total_samples = len(combined_preds)
    else:
        overall_ic = overall_rank_ic = 0.0
        total_samples = 0

    duration = time.time() - start_time

    return GroupResult(
        name=name,
        yearly=yearly_results,
        overall_ic=overall_ic,
        overall_rank_ic=overall_rank_ic,
        total_samples=total_samples,
        duration=duration,
    )


def print_results(baseline: GroupResult, treatment: GroupResult) -> None:
    """打印对比结果表"""
    logger.info(f"\n{'='*80}")
    logger.info("5 年 A/B 对比结果")
    logger.info(f"{'='*80}")

    # 逐年对比
    header = (
        f"{'年份':<6} │ {'Baseline IC':>11} {'RankIC':>8} {'N':>7}"
        f" │ {'CSRankNorm IC':>13} {'RankIC':>8} {'N':>7}"
        f" │ {'ΔRankIC':>8}"
    )
    logger.info(header)
    logger.info("─" * 80)

    b_yearly = {r.year: r for r in baseline.yearly}
    t_yearly = {r.year: r for r in treatment.yearly}

    for year in TEST_YEARS:
        b = b_yearly.get(year)
        t = t_yearly.get(year)
        if b and t:
            delta = t.rank_ic - b.rank_ic
            row = (
                f"{year:<6} │ {b.ic:>11.4f} {b.rank_ic:>8.4f} {b.n_samples:>7}"
                f" │ {t.ic:>13.4f} {t.rank_ic:>8.4f} {t.n_samples:>7}"
                f" │ {delta:>+8.4f}"
            )
            logger.info(row)
        elif b:
            logger.info(f"{year:<6} │ {b.ic:>11.4f} {b.rank_ic:>8.4f} {b.n_samples:>7} │ {'N/A':>13} {'N/A':>8} {'N/A':>7} │ {'N/A':>8}")
        elif t:
            logger.info(f"{year:<6} │ {'N/A':>11} {'N/A':>8} {'N/A':>7} │ {t.ic:>13.4f} {t.rank_ic:>8.4f} {t.n_samples:>7} │ {'N/A':>8}")

    logger.info("─" * 80)

    # 总体
    delta_overall = treatment.overall_rank_ic - baseline.overall_rank_ic
    logger.info(
        f"{'Overall':<6} │ {baseline.overall_ic:>11.4f} {baseline.overall_rank_ic:>8.4f}"
        f" {baseline.total_samples:>7}"
        f" │ {treatment.overall_ic:>13.4f} {treatment.overall_rank_ic:>8.4f}"
        f" {treatment.total_samples:>7}"
        f" │ {delta_overall:>+8.4f}"
    )
    logger.info(f"{'='*80}")
    logger.info(
        f"耗时: Baseline={baseline.duration:.0f}s, "
        f"CSRankNorm={treatment.duration:.0f}s"
    )

    # 注意事项
    logger.info("\n注意:")
    logger.info(
        "  - IC (Pearson) 与 RankIC (Spearman) 均在原始预测值 vs 实际值上计算"
    )
    logger.info(
        "  - CSRankNorm 组的标签经过排名标准化，预测值尺度不同于 Baseline"
    )
    logger.info(
        "  - 因此两组的 MSE 不可比；IC/RankIC 是排序相关性指标，可跨尺度比较"
    )


def main():
    """运行 5 年 A/B 对比实验"""
    logger.info("CSRankNorm 5 年 Expanding Window A/B 实验")
    logger.info(f"数据范围: {DATA_START} ~ {DATA_END}")
    logger.info(f"测试年份: {TEST_YEARS}")
    logger.info(f"股票数: {N_STOCKS}")

    # --- 准备 Baseline 数据（无 CSRankNorm）---
    logger.info("\n>>> 准备 Baseline 数据...")
    baseline_config = TrainingConfig(
        start_date=DATA_START,
        end_date=DATA_END,
        n_stocks=N_STOCKS,
        label_transform=None,
    )
    baseline_data = prepare_dataset(baseline_config)

    # --- 准备 CSRankNorm 数据 ---
    logger.info("\n>>> 准备 CSRankNorm 数据...")
    csrn_config = TrainingConfig(
        start_date=DATA_START,
        end_date=DATA_END,
        n_stocks=N_STOCKS,
        label_transform="csranknorm",
    )
    csrn_data = prepare_dataset(csrn_config)

    # --- Baseline: 新参数 + reg=5（适配原始标签尺度）---
    baseline_lgb = LGBConfig(seed=RANDOM_SEED)
    baseline_lgb.reg_alpha = 5.0
    baseline_lgb.reg_lambda = 5.0

    baseline_result = run_group_experiment(
        "Baseline(reg=5)", baseline_data, baseline_lgb, baseline_config,
    )

    # --- CSRankNorm: 新参数 + reg=200（适配 N(0,1) 标签）---
    csrn_lgb = LGBConfig(seed=RANDOM_SEED)

    csrn_result = run_group_experiment(
        "CSRankNorm(reg=200)", csrn_data, csrn_lgb, csrn_config,
    )

    # --- 输出对比 ---
    print_results(baseline_result, csrn_result)


if __name__ == "__main__":
    main()
