"""
滚动训练模块 (Walk-Forward Rolling Trainer)

核心思路：
  - 滑动窗口训练：每隔 step 个月，用最近 train_months 的数据训练模型
  - 每个窗口独立训练 → 独立预测 → 拼接所有窗口的测试期做回测
  - 测试期永远是模型没见过的未来数据，杜绝过拟合

时间线示意（默认 train=18m, val=3m, test=3m, step=3m）：
  Window 0: Train [2020-01~2021-06] | Val [2021-07~2021-09] | Test [2021-10~2021-12]
  Window 1: Train [2020-04~2021-09] | Val [2021-10~2021-12] | Test [2022-01~2022-03]
  ...

用法：
  python -m experiments.unified.rolling_trainer [--n_stocks 300] [--train_months 18] [--step_months 3]
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from loguru import logger

from .config import LGBConfig, TrainingConfig
from .constants import (
    DATA_DIR,
    EMBARGO_DAYS,
    MODEL_DIR,
    RANDOM_SEED,
    TRADING_DAYS_PER_YEAR,
)
from .features import (
    compute_all_features,
    compute_cross_sectional_features,
    compute_regression_label,
    get_feature_columns,
    neutralize_features,
)
from .label_transform import cs_rank_norm


# ─── 配置 ───────────────────────────────────────────────

@dataclass
class RollingConfig:
    """滚动训练配置"""
    # 数据
    data_dir: Path = DATA_DIR
    n_stocks: int = 300
    start_date: str = "2020-01-01"   # 整体数据起始
    end_date: str = "2025-01-01"     # 整体数据截止

    # 滚动窗口
    train_months: int = 18           # 训练窗口长度
    val_months: int = 3              # 验证窗口长度
    test_months: int = 3             # 测试窗口长度
    step_months: int = 3             # 滚动步长
    embargo_days: int = EMBARGO_DAYS # 训练/验证之间的 embargo

    # 模型
    seed: int = RANDOM_SEED
    label_transform: str = "csranknorm"
    enable_neutralization: bool = True

    # 输出
    output_dir: Path = MODEL_DIR / "rolling"


# ─── 数据加载 ─────────────────────────────────────────────

def load_stock_universe(config: RollingConfig) -> pd.DataFrame:
    """
    加载股票池：按成交量排序取 Top N，计算特征和标签

    与 data_loader.load_stock_files 类似，但接受 RollingConfig
    """
    stock_files = list(config.data_dir.glob("*.parquet"))
    logger.info(f"发现 {len(stock_files)} 个股票文件")

    # 筛选有效股票（交易日 >= 200）
    valid = []
    for f in stock_files:
        try:
            df = pd.read_parquet(f)
            df = df[(df["date"] >= config.start_date) & (df["date"] < config.end_date)]
            if len(df) >= 200:
                valid.append((f, df["volume"].mean()))
        except Exception:
            continue

    valid.sort(key=lambda x: x[1], reverse=True)
    selected = valid[: config.n_stocks]
    logger.info(f"筛选出 {len(valid)} 只有效股票，选择 Top {len(selected)}")

    # 加载并计算特征
    all_data = []
    for i, (f, _) in enumerate(selected):
        df = pd.read_parquet(f)
        df = df[(df["date"] >= config.start_date) & (df["date"] < config.end_date)]
        df = compute_all_features(df)
        df = compute_regression_label(df)
        all_data.append(df)
        if (i + 1) % 50 == 0:
            logger.info(f"  已加载 {i + 1}/{len(selected)} 只股票")

    data = pd.concat(all_data, ignore_index=True)
    logger.info(f"总数据量: {len(data)} 条")

    # 截面特征
    logger.info("计算截面特征...")
    data = compute_cross_sectional_features(data)

    if config.enable_neutralization:
        logger.info("执行市场中性化...")
        data = neutralize_features(data)

    # 清洗
    feature_cols = get_feature_columns()
    data = data.dropna(subset=feature_cols + ["future_return"])
    logger.info(f"清洗后数据量: {len(data)} 条")

    return data


# ─── 窗口生成 ─────────────────────────────────────────────

@dataclass
class RollingWindow:
    """单个滚动窗口的时间范围"""
    index: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


def generate_windows(config: RollingConfig) -> List[RollingWindow]:
    """
    生成所有滚动窗口

    从 start_date 开始，每隔 step_months 滑动一次，
    直到 test_end 超过 end_date
    """
    windows = []
    # 第一个窗口的 train_start
    base = datetime.strptime(config.start_date, "%Y-%m-%d")
    end_limit = datetime.strptime(config.end_date, "%Y-%m-%d")

    idx = 0
    while True:
        train_start = base + relativedelta(months=config.step_months * idx)
        train_end = train_start + relativedelta(months=config.train_months)
        val_start = train_end
        val_end = val_start + relativedelta(months=config.val_months)
        test_start = val_end
        test_end = test_start + relativedelta(months=config.test_months)

        if test_start >= end_limit:
            break

        # 如果 test_end 超过数据截止日，截断
        if test_end > end_limit:
            test_end = end_limit

        windows.append(RollingWindow(
            index=idx,
            train_start=train_start.strftime("%Y-%m-%d"),
            train_end=train_end.strftime("%Y-%m-%d"),
            val_start=val_start.strftime("%Y-%m-%d"),
            val_end=val_end.strftime("%Y-%m-%d"),
            test_start=test_start.strftime("%Y-%m-%d"),
            test_end=test_end.strftime("%Y-%m-%d"),
        ))
        idx += 1

    logger.info(f"生成 {len(windows)} 个滚动窗口")
    for w in windows:
        logger.info(
            f"  Window {w.index}: "
            f"Train [{w.train_start}~{w.train_end}) | "
            f"Val [{w.val_start}~{w.val_end}) | "
            f"Test [{w.test_start}~{w.test_end})"
        )
    return windows


# ─── 单窗口训练 ─────────────────────────────────────────────

@dataclass
class WindowResult:
    """单窗口训练结果"""
    window: RollingWindow
    train_size: int
    val_size: int
    test_size: int
    val_metrics: Dict[str, float]
    test_predictions: Optional[pd.DataFrame] = None  # date, stock, pred, actual


def train_single_window(
    data: pd.DataFrame,
    window: RollingWindow,
    config: RollingConfig,
) -> WindowResult:
    """
    训练单个滚动窗口

    1. 按时间切分 train/val/test
    2. 对训练+验证集做 CSRankNorm（独立于测试集）
    3. 训练 LightGBM
    4. 在测试集上预测
    """
    feature_cols = get_feature_columns()

    # 切分数据
    train_mask = (data["date"] >= window.train_start) & (data["date"] < window.train_end)
    val_mask = (data["date"] >= window.val_start) & (data["date"] < window.val_end)
    test_mask = (data["date"] >= window.test_start) & (data["date"] < window.test_end)

    train_df = data[train_mask].copy()
    val_df = data[val_mask].copy()
    test_df = data[test_mask].copy()

    logger.info(
        f"Window {window.index}: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    if len(train_df) < 1000 or len(val_df) < 100 or len(test_df) < 100:
        logger.warning(f"Window {window.index}: 数据量不足，跳过")
        return WindowResult(
            window=window,
            train_size=len(train_df),
            val_size=len(val_df),
            test_size=len(test_df),
            val_metrics={},
        )

    # 标签变换（CSRankNorm）—— 只在 train+val 上 fit
    if config.label_transform == "csranknorm":
        train_df = cs_rank_norm(train_df, label_col="future_return")
        val_df = cs_rank_norm(val_df, label_col="future_return")
        # 测试集也需要变换（用于评估 IC 等指标时对齐尺度）
        test_df = cs_rank_norm(test_df, label_col="future_return")

    # 保存测试集的原始收益率（用于回测）
    test_actual = data[test_mask]["future_return"].values

    # 训练 LightGBM
    lgb_config = LGBConfig(seed=config.seed)
    X_train, y_train = train_df[feature_cols], train_df["future_return"]
    X_val, y_val = val_df[feature_cols], val_df["future_return"]

    train_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_val, y_val, reference=train_data)

    model = lgb.train(
        lgb_config.to_dict(),
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(500),
        ],
    )

    # 验证集指标
    val_pred = model.predict(X_val)
    val_ic = _rank_ic(val_pred, y_val.values)
    val_mse = float(np.mean((val_pred - y_val.values) ** 2))

    # 测试集预测
    X_test = test_df[feature_cols]
    test_pred = model.predict(X_test)
    test_ic = _rank_ic(test_pred, test_df["future_return"].values)

    logger.info(
        f"Window {window.index}: "
        f"val_IC={val_ic:.4f}, val_MSE={val_mse:.6f}, "
        f"test_IC={test_ic:.4f}, best_iter={model.best_iteration}"
    )

    # 构建测试集预测 DataFrame
    test_predictions = pd.DataFrame({
        "date": test_df["date"].values,
        "stock_code": test_df["stock_code"].values if "stock_code" in test_df.columns else range(len(test_df)),
        "pred": test_pred,
        "actual_return": test_actual,  # 原始收益率
    })

    return WindowResult(
        window=window,
        train_size=len(train_df),
        val_size=len(val_df),
        test_size=len(test_df),
        val_metrics={"ic": val_ic, "mse": val_mse},
        test_predictions=test_predictions,
    )


# ─── 回测 ─────────────────────────────────────────────────

def backtest_rolling_predictions(
    all_predictions: pd.DataFrame,
    top_n: int = 10,
    cost: float = 0.001,
) -> Dict[str, float]:
    """
    对拼接后的所有窗口预测做回测

    每日选 pred 排名 Top N 的股票等权持有，计算组合收益
    """
    daily_returns = []
    prev_holdings = set()

    for date, group in all_predictions.groupby("date"):
        top_stocks = group.nlargest(top_n, "pred")
        current_holdings = set(top_stocks["stock_code"].values)

        # 换手成本
        if prev_holdings:
            turnover = len(current_holdings - prev_holdings) / top_n
        else:
            turnover = 1.0

        daily_ret = top_stocks["actual_return"].mean() - cost * turnover
        daily_returns.append({"date": date, "return": daily_ret})
        prev_holdings = current_holdings

    if not daily_returns:
        return _empty_metrics()

    ret_df = pd.DataFrame(daily_returns).sort_values("date")
    returns = ret_df["return"]

    cumulative = (1 + returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1)
    n_days = len(returns)
    annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / max(n_days, 1)) - 1

    # 最大回撤
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    # 夏普
    std = returns.std()
    sharpe = float(returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)) if std > 0 else 0.0

    # 胜率
    win_rate = float((returns > 0).sum() / n_days)

    # Calmar
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # 日均换手率
    avg_turnover = _compute_avg_turnover(all_predictions, top_n)

    metrics = {
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "calmar": calmar,
        "total_return": total_return,
        "n_trading_days": n_days,
        "avg_daily_turnover": avg_turnover,
    }

    logger.info(
        f"滚动回测结果: "
        f"年化={annual_return*100:.2f}%, 夏普={sharpe:.3f}, "
        f"回撤={max_drawdown*100:.2f}%, 胜率={win_rate*100:.1f}%, "
        f"Calmar={calmar:.3f}, 交易日={n_days}"
    )
    return metrics


def _compute_avg_turnover(predictions: pd.DataFrame, top_n: int) -> float:
    """计算平均日换手率"""
    dates = sorted(predictions["date"].unique())
    if len(dates) < 2:
        return 0.0

    turnovers = []
    prev_holdings = None
    for date in dates:
        group = predictions[predictions["date"] == date]
        top = set(group.nlargest(top_n, "pred")["stock_code"].values)
        if prev_holdings is not None:
            turnover = len(top - prev_holdings) / top_n
            turnovers.append(turnover)
        prev_holdings = top

    return float(np.mean(turnovers)) if turnovers else 0.0


# ─── IC 分析 ──────────────────────────────────────────────

def analyze_ic_series(all_predictions: pd.DataFrame) -> Dict[str, float]:
    """
    分析每日 Rank IC 序列

    返回 mean IC, IC std, ICIR, IC>0 比例
    """
    daily_ics = []
    for date, group in all_predictions.groupby("date"):
        if len(group) < 10:
            continue
        ic = _rank_ic(group["pred"].values, group["actual_return"].values)
        if np.isfinite(ic):
            daily_ics.append(ic)

    if not daily_ics:
        return {"mean_ic": 0, "ic_std": 0, "icir": 0, "ic_positive_ratio": 0}

    ics = np.array(daily_ics)
    mean_ic = float(ics.mean())
    ic_std = float(ics.std())
    icir = mean_ic / ic_std if ic_std > 0 else 0.0
    ic_pos = float((ics > 0).sum() / len(ics))

    logger.info(
        f"IC 分析: mean={mean_ic:.4f}, std={ic_std:.4f}, "
        f"ICIR={icir:.4f}, IC>0={ic_pos*100:.1f}%"
    )
    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "icir": icir,
        "ic_positive_ratio": ic_pos,
    }


# ─── 工具函数 ─────────────────────────────────────────────

def _rank_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Spearman Rank IC"""
    from scipy.stats import spearmanr
    if len(pred) < 5:
        return 0.0
    corr, _ = spearmanr(pred, actual)
    return float(corr) if np.isfinite(corr) else 0.0


def _empty_metrics() -> Dict[str, float]:
    return {
        "annual_return": 0, "sharpe": 0, "max_drawdown": 0,
        "win_rate": 0, "calmar": 0, "total_return": 0,
        "n_trading_days": 0, "avg_daily_turnover": 0,
    }


# ─── 主流程 ──────────────────────────────────────────────

def run_rolling_training(config: RollingConfig) -> Dict:
    """
    执行完整的滚动训练流程

    1. 加载股票池数据
    2. 生成滚动窗口
    3. 逐窗口训练 + 预测
    4. 拼接预测 → 回测
    5. IC 分析
    6. 保存结果
    """
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("开始滚动训练 (Walk-Forward)")
    logger.info(f"配置: {config}")
    logger.info("=" * 60)

    # 1. 加载数据
    data = load_stock_universe(config)

    # 2. 生成窗口
    windows = generate_windows(config)
    if not windows:
        logger.error("无有效窗口，退出")
        return {}

    # 3. 逐窗口训练
    results: List[WindowResult] = []
    all_test_preds = []

    for window in windows:
        logger.info(f"\n{'='*40} Window {window.index} {'='*40}")
        result = train_single_window(data, window, config)
        results.append(result)

        if result.test_predictions is not None and len(result.test_predictions) > 0:
            all_test_preds.append(result.test_predictions)

    # 4. 拼接预测 → 回测
    if not all_test_preds:
        logger.error("所有窗口均无有效预测，退出")
        return {}

    all_predictions = pd.concat(all_test_preds, ignore_index=True)
    logger.info(f"\n拼接后总预测: {len(all_predictions)} 条, "
                f"覆盖 {all_predictions['date'].nunique()} 个交易日")

    # 多组 top_n 回测
    backtest_results = {}
    for top_n in [5, 10, 20]:
        logger.info(f"\n--- Top {top_n} 回测 ---")
        metrics = backtest_rolling_predictions(all_predictions, top_n=top_n)
        backtest_results[f"top_{top_n}"] = metrics

    # 5. IC 分析
    ic_analysis = analyze_ic_series(all_predictions)

    # 6. 汇总
    elapsed = time.time() - t0
    summary = {
        "config": {
            "n_stocks": config.n_stocks,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "train_months": config.train_months,
            "val_months": config.val_months,
            "test_months": config.test_months,
            "step_months": config.step_months,
            "label_transform": config.label_transform,
            "n_windows": len(windows),
        },
        "window_results": [
            {
                "index": r.window.index,
                "train_period": f"{r.window.train_start}~{r.window.train_end}",
                "test_period": f"{r.window.test_start}~{r.window.test_end}",
                "train_size": r.train_size,
                "test_size": r.test_size,
                "val_ic": r.val_metrics.get("ic", None),
            }
            for r in results
        ],
        "ic_analysis": ic_analysis,
        "backtest": backtest_results,
        "elapsed_seconds": round(elapsed, 1),
    }

    # 保存
    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.output_dir / "rolling_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
    logger.info(f"\n报告已保存: {report_path}")

    # 保存预测数据（供后续分析）
    pred_path = config.output_dir / "rolling_predictions.parquet"
    all_predictions.to_parquet(pred_path, index=False)
    logger.info(f"预测数据已保存: {pred_path}")

    # 打印最终汇总
    _print_summary(summary)

    return summary


def _print_summary(summary: Dict) -> None:
    """打印最终汇总"""
    logger.info("\n" + "=" * 60)
    logger.info("滚动训练完成 - 最终汇总")
    logger.info("=" * 60)

    ic = summary["ic_analysis"]
    logger.info(f"IC: mean={ic['mean_ic']:.4f}, ICIR={ic['icir']:.4f}, IC>0={ic['ic_positive_ratio']*100:.1f}%")

    for key, bt in summary["backtest"].items():
        logger.info(
            f"{key}: 年化={bt['annual_return']*100:.2f}%, "
            f"夏普={bt['sharpe']:.3f}, 回撤={bt['max_drawdown']*100:.2f}%, "
            f"胜率={bt['win_rate']*100:.1f}%, Calmar={bt['calmar']:.3f}"
        )

    logger.info(f"耗时: {summary['elapsed_seconds']:.0f}s")


# ─── CLI ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Walk-Forward 滚动训练")
    parser.add_argument("--n_stocks", type=int, default=300, help="股票池大小")
    parser.add_argument("--start_date", type=str, default="2020-01-01")
    parser.add_argument("--end_date", type=str, default="2025-01-01")
    parser.add_argument("--train_months", type=int, default=18)
    parser.add_argument("--val_months", type=int, default=3)
    parser.add_argument("--test_months", type=int, default=3)
    parser.add_argument("--step_months", type=int, default=3)
    parser.add_argument("--label_transform", type=str, default="csranknorm")
    parser.add_argument("--no_neutralize", action="store_true")
    args = parser.parse_args()

    config = RollingConfig(
        n_stocks=args.n_stocks,
        start_date=args.start_date,
        end_date=args.end_date,
        train_months=args.train_months,
        val_months=args.val_months,
        test_months=args.test_months,
        step_months=args.step_months,
        label_transform=args.label_transform,
        enable_neutralization=not args.no_neutralize,
    )

    run_rolling_training(config)


if __name__ == "__main__":
    main()
