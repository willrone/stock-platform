"""
评估与回测模块

回归模型评估指标 + 基于预测收益率排序的回测
"""
from typing import Dict

import numpy as np
import pandas as pd
from loguru import logger

from .constants import DEFAULT_TOP_N, DEFAULT_TRANSACTION_COST, TRADING_DAYS_PER_YEAR


def evaluate_on_test(
    predictions: np.ndarray, actuals: pd.Series, label: str = "Ensemble"
) -> Dict[str, float]:
    """
    在测试集上评估回归模型

    Returns:
        包含 mse, r2, ic 的指标字典
    """
    actuals_arr = actuals.values
    mse = float(np.mean((predictions - actuals_arr) ** 2))
    r2 = _compute_r2(predictions, actuals_arr)
    ic = _compute_ic(predictions, actuals_arr)

    logger.info(f"{label} 测试指标: MSE={mse:.6f}, R²={r2:.6f}, IC={ic:.4f}")
    return {"mse": mse, "r2": r2, "ic": ic}


def backtest_by_ranking(test_df: pd.DataFrame) -> Dict[str, float]:
    """
    基于预测收益率排序的回测

    选股逻辑：每日按预测收益率排序，选 Top N 等权持有

    Args:
        test_df: 包含 date, pred, actual_return 列的 DataFrame
    """
    top_n = DEFAULT_TOP_N
    cost = DEFAULT_TRANSACTION_COST

    daily_returns = _compute_daily_portfolio_returns(test_df, top_n, cost)

    if daily_returns.empty:
        logger.warning("回测无有效交易日")
        return _empty_backtest_result()

    return _compute_backtest_metrics(daily_returns)


def _compute_daily_portfolio_returns(
    test_df: pd.DataFrame, top_n: int, cost: float
) -> pd.Series:
    """计算每日组合收益率：选 Top N 股票等权"""
    daily_groups = []
    for date_str, group in test_df.groupby("date_str"):
        top_stocks = group.nlargest(top_n, "pred")
        daily_groups.append(top_stocks)

    if not daily_groups:
        return pd.Series(dtype=float)

    daily_top = pd.concat(daily_groups, ignore_index=True)
    daily_returns = (
        daily_top.groupby("date_str")["actual_return"].mean() - cost
    )
    return daily_returns


def _compute_backtest_metrics(daily_returns: pd.Series) -> Dict[str, float]:
    """从日收益率序列计算回测指标"""
    total_days = len(daily_returns)
    win_rate = float((daily_returns > 0).sum() / total_days)
    avg_daily = float(daily_returns.mean())

    cumulative = (1 + daily_returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1)
    annual_return = _annualize_return(total_return, total_days)

    max_drawdown = _compute_max_drawdown(cumulative)
    sharpe = _compute_sharpe(daily_returns)

    logger.info(
        f"回测结果: 夏普={sharpe:.2f}, 年化={annual_return*100:.1f}%, "
        f"最大回撤={max_drawdown*100:.1f}%, 胜率={win_rate*100:.1f}%"
    )
    return {
        "sharpe": sharpe,
        "annual_return": annual_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "total_return": total_return,
    }


def _annualize_return(total_return: float, total_days: int) -> float:
    """年化收益率"""
    if total_days <= 0:
        return 0.0
    return float(
        (1 + total_return) ** (TRADING_DAYS_PER_YEAR / total_days) - 1
    )


def _compute_max_drawdown(cumulative: pd.Series) -> float:
    """计算最大回撤"""
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def _compute_sharpe(daily_returns: pd.Series) -> float:
    """计算年化夏普比率"""
    std = daily_returns.std()
    if std <= 0:
        return 0.0
    return float(
        daily_returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def _compute_r2(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """计算 R²"""
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def _compute_ic(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """计算信息系数"""
    if len(predictions) < 2:
        return 0.0
    corr = np.corrcoef(predictions, actuals)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def _empty_backtest_result() -> Dict[str, float]:
    """空回测结果"""
    return {
        "sharpe": 0.0,
        "annual_return": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "total_return": 0.0,
    }
