#!/usr/bin/env python3
"""
A/B 对比实验：P0(CSRankNorm) + P1(参数修复)

三组对比：
  Baseline — 旧参数 + 原始标签
  B1       — 新参数（Qlib 默认）+ 原始标签
  B2       — 新参数 + CSRankNorm 标签

输出指标：IC / RankIC / MSE
"""
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from unified.config import LGBConfig, TrainingConfig
from unified.constants import RANDOM_SEED
from unified.data_loader import prepare_dataset, split_with_embargo
from unified.features import get_feature_columns
from unified.trainer import train_lightgbm

np.random.seed(RANDOM_SEED)


# === Baseline 旧参数 ===
@dataclass
class BaselineLGBConfig(LGBConfig):
    """旧版 LightGBM 参数（修复前）"""

    objective: str = "huber"
    metric: str = "mae"
    num_leaves: int = 45
    max_depth: int = 6
    learning_rate: float = 0.015
    feature_fraction: float = 0.65
    bagging_fraction: float = 0.65
    min_child_samples: int = 100
    reg_alpha: float = 0.6
    reg_lambda: float = 0.6


@dataclass
class ExperimentResult:
    """单组实验结果"""

    name: str
    ic: float
    rank_ic: float
    mse: float
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


def run_single_experiment(
    name: str,
    config: TrainingConfig,
    lgb_config: LGBConfig,
) -> ExperimentResult:
    """
    运行单组实验：数据准备 → 训练 → 评估

    Returns:
        ExperimentResult 包含 IC/RankIC/MSE
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"实验: {name}")
    logger.info(f"{'='*60}")

    start = time.time()

    data = prepare_dataset(config)
    train, val, test = split_with_embargo(data, config)

    feature_cols = get_feature_columns()
    model, _ = _train_with_config(train, val, config, lgb_config)

    predictions = model.predict(test[feature_cols])
    actuals = test["future_return"].values

    ic = compute_ic(predictions, actuals)
    rank_ic = compute_rank_ic(predictions, actuals)
    mse = float(np.mean((predictions - actuals) ** 2))
    duration = time.time() - start

    logger.info(
        f"结果: IC={ic:.4f}, RankIC={rank_ic:.4f}, "
        f"MSE={mse:.6f}, 耗时={duration:.1f}s",
    )
    return ExperimentResult(name, ic, rank_ic, mse, duration)


def _train_with_config(
    train: pd.DataFrame,
    val: pd.DataFrame,
    config: TrainingConfig,
    lgb_config: LGBConfig,
):
    """用指定的 LGBConfig 训练模型"""
    import lightgbm as lgb

    feature_cols = get_feature_columns()
    X_train = train[feature_cols]
    y_train = train["future_return"]
    X_val = val[feature_cols]
    y_val = val["future_return"]

    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)

    model = lgb.train(
        lgb_config.to_dict(),
        train_data,
        num_boost_round=config.max_boost_rounds,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(config.early_stopping_rounds),
            lgb.log_evaluation(200),
        ],
    )
    return model, {}


def print_comparison_table(results: List[ExperimentResult]) -> None:
    """打印对比表"""
    logger.info(f"\n{'='*70}")
    logger.info("A/B 对比结果")
    logger.info(f"{'='*70}")
    header = f"{'实验':<25} {'IC':>8} {'RankIC':>8} {'MSE':>12} {'耗时(s)':>8}"
    logger.info(header)
    logger.info("-" * 70)
    for r in results:
        row = (
            f"{r.name:<25} {r.ic:>8.4f} {r.rank_ic:>8.4f} "
            f"{r.mse:>12.6f} {r.duration:>8.1f}"
        )
        logger.info(row)
    logger.info("-" * 70)

    if len(results) >= 3:
        baseline = results[0]
        for r in results[1:]:
            ic_delta = r.ic - baseline.ic
            ric_delta = r.rank_ic - baseline.rank_ic
            logger.info(
                f"  {r.name} vs Baseline: "
                f"ΔIC={ic_delta:+.4f}, ΔRankIC={ric_delta:+.4f}",
            )


def main():
    """运行 A/B 对比实验"""
    logger.info("P0+P1 A/B 对比实验")
    logger.info("Baseline: 旧参数 + 原始标签")
    logger.info("B1: 新参数(Qlib默认) + 原始标签")
    logger.info("B2: 新参数 + CSRankNorm 标签")

    results: List[ExperimentResult] = []

    # Baseline: 旧参数 + 无 CSRankNorm
    baseline_config = TrainingConfig(label_transform=None)
    baseline_lgb = BaselineLGBConfig(seed=RANDOM_SEED)
    results.append(run_single_experiment(
        "Baseline(旧参数)", baseline_config, baseline_lgb,
    ))

    # B1: 新参数 + 无 CSRankNorm
    b1_config = TrainingConfig(label_transform=None)
    b1_lgb = LGBConfig(seed=RANDOM_SEED)
    results.append(run_single_experiment(
        "B1(新参数)", b1_config, b1_lgb,
    ))

    # B2: 新参数 + CSRankNorm
    b2_config = TrainingConfig(label_transform="csranknorm")
    b2_lgb = LGBConfig(seed=RANDOM_SEED)
    results.append(run_single_experiment(
        "B2(新参数+CSRankNorm)", b2_config, b2_lgb,
    ))

    print_comparison_table(results)


if __name__ == "__main__":
    main()
