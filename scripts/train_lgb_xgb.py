#!/usr/bin/env python3
"""
LightGBM + XGBoost 双模型训练脚本（优化版）

独立运行，不依赖 Willrone 后端服务。
使用本地 Parquet 数据训练模型，输出到 data/models/。

用法:
    cd /Users/ronghui/Projects/willrone
    backend/venv/bin/python scripts/train_lgb_xgb.py
"""

import glob
import json
import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 常量
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "parquet" / "stock_data"
MODEL_DIR = PROJECT_ROOT / "data" / "models"

# 时间窗口（需要 60 天 lookback，所以从 2017-01-01 开始加载）
DATA_START = "2017-01-01"
TRAIN_START = "2018-01-01"
TRAIN_END = "2023-12-31"
VALID_START = "2024-01-01"
VALID_END = "2024-06-30"
TEST_START = "2024-07-01"
TEST_END = "2025-12-31"

LABEL_HORIZON = 5
MIN_HISTORY_DAYS = 60
MIN_DAILY_VOLUME = 50_000
MAX_STOCKS = 800  # 限制股票数量，选流动性最好的

FEATURE_NAMES: List[str] = [
    "return_1d", "return_2d", "return_3d", "return_5d", "return_10d", "return_20d",
    "momentum_short", "momentum_long", "momentum_reversal",
    "momentum_strength_5", "momentum_strength_10", "momentum_strength_20",
    "relative_strength_5d", "relative_strength_20d", "relative_momentum",
    "return_1d_rank", "return_5d_rank", "return_20d_rank", "volume_rank", "volatility_20_rank",
    "market_up_ratio",
    "ma_ratio_5", "ma_ratio_10", "ma_ratio_20", "ma_ratio_60",
    "ma_slope_5", "ma_slope_10", "ma_slope_20", "ma_alignment",
    "volatility_5", "volatility_20", "volatility_60", "vol_regime", "volatility_skew",
    "vol_ratio", "vol_ma_ratio", "vol_std", "vol_price_diverge",
    "rsi_6", "rsi_14", "rsi_diff",
    "macd", "macd_signal", "macd_hist", "macd_hist_slope",
    "bb_position", "bb_width",
    "body", "wick_upper", "wick_lower", "range_pct", "consecutive_up", "consecutive_down",
    "price_pos_20", "price_pos_60",
    "dist_high_20", "dist_low_20", "dist_high_60", "dist_low_60",
    "atr_pct", "di_diff", "adx",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# 数据加载（优化：只加载 2017+ 数据，选 Top 流动性股票）
# ============================================================

def load_and_filter_stocks() -> pd.DataFrame:
    """加载数据，过滤日期和流动性，选 Top N 股票。"""
    files = sorted(glob.glob(str(DATA_DIR / "*.parquet")))
    logger.info("发现 %d 个 parquet 文件", len(files))

    # 第一遍：快速扫描，计算每只股票的平均成交量
    stock_volumes: Dict[str, float] = {}
    for f in files:
        stock_code = os.path.basename(f).replace(".parquet", "")
        df = pd.read_parquet(f, columns=["date", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        recent = df[df["date"] >= DATA_START]
        if len(recent) >= MIN_HISTORY_DAYS:
            stock_volumes[stock_code] = recent["volume"].mean()

    logger.info("满足最低历史天数的股票: %d", len(stock_volumes))

    # 过滤最低成交量，然后选 Top N
    qualified = {k: v for k, v in stock_volumes.items() if v >= MIN_DAILY_VOLUME}
    logger.info("满足最低成交量的股票: %d", len(qualified))

    # 按成交量排序，选 Top N
    sorted_stocks = sorted(qualified.items(), key=lambda x: x[1], reverse=True)
    selected = [s[0] for s in sorted_stocks[:MAX_STOCKS]]
    logger.info("选择 Top %d 流动性股票", len(selected))

    # 第二遍：加载选中股票的完整数据
    frames: List[pd.DataFrame] = []
    for code in selected:
        f = DATA_DIR / f"{code}.parquet"
        df = pd.read_parquet(f)
        df["date"] = pd.to_datetime(df["date"])
        # 只保留 2017+ 数据
        df = df[df["date"] >= DATA_START].copy()
        df["stock_code"] = code
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(["stock_code", "date"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    logger.info("加载完成: %d 只股票, %d 行", len(selected), len(combined))
    return combined


# ============================================================
# 特征工程
# ============================================================

def compute_ts_features(df: pd.DataFrame) -> pd.DataFrame:
    """为单只股票计算时序特征 + 标签。"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    open_ = df["open"]
    returns = close.pct_change()

    # 收益率
    for p in [1, 2, 3, 5, 10, 20]:
        df[f"return_{p}d"] = close.pct_change(p)

    # 动量
    df["momentum_short"] = df["return_5d"] - df["return_10d"]
    df["momentum_long"] = df["return_10d"] - df["return_20d"]
    df["momentum_reversal"] = -df["return_1d"]
    for p in [5, 10, 20]:
        df[f"momentum_strength_{p}"] = (returns > 0).rolling(p).sum() / p

    # 移动平均
    for w in [5, 10, 20, 60]:
        ma = close.rolling(w).mean()
        df[f"ma_ratio_{w}"] = close / ma - 1
        df[f"ma_slope_{w}"] = ma.pct_change(5)
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    df["ma_alignment"] = (ma5 > ma10).astype(int) + (ma10 > ma20).astype(int)

    # 波动率
    for w in [5, 20, 60]:
        df[f"volatility_{w}"] = returns.rolling(w).std()
    df["vol_regime"] = df["volatility_5"] / (df["volatility_20"] + 1e-10)
    df["volatility_skew"] = returns.rolling(20).skew()

    # 成交量
    vol_ma20 = volume.rolling(20).mean()
    vol_ma5 = volume.rolling(5).mean()
    df["vol_ratio"] = volume / (vol_ma20 + 1)
    df["vol_ma_ratio"] = vol_ma5 / (vol_ma20 + 1)
    df["vol_std"] = volume.rolling(20).std() / (vol_ma20 + 1)
    price_up = (close > close.shift(1)).astype(int)
    vol_up = (volume > volume.shift(1)).astype(int)
    df["vol_price_diverge"] = (price_up != vol_up).astype(int)

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss_s = -delta.where(delta < 0, 0)
    for p in [6, 14]:
        avg_gain = gain.ewm(span=p, adjust=False).mean()
        avg_loss = loss_s.ewm(span=p, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        df[f"rsi_{p}"] = 100 - (100 / (1 + rs))
    df["rsi_diff"] = df["rsi_6"] - df["rsi_14"]

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_hist_slope"] = df["macd_hist"].diff(3)

    # 布林带
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_position"] = (close - bb_mid) / (2 * bb_std + 1e-10)
    df["bb_width"] = 4 * bb_std / (bb_mid + 1e-10)

    # 价格形态
    df["body"] = (close - open_) / (open_ + 1e-10)
    df["wick_upper"] = (high - np.maximum(close, open_)) / (high - low + 1e-10)
    df["wick_lower"] = (np.minimum(close, open_) - low) / (high - low + 1e-10)
    df["range_pct"] = (high - low) / (close + 1e-10)
    df["consecutive_up"] = (close > close.shift(1)).rolling(5).sum()
    df["consecutive_down"] = (close < close.shift(1)).rolling(5).sum()

    # 价格位置
    for w in [20, 60]:
        high_n = high.rolling(w).max()
        low_n = low.rolling(w).min()
        df[f"price_pos_{w}"] = (close - low_n) / (high_n - low_n + 1e-10)
        df[f"dist_high_{w}"] = (high_n - close) / (close + 1e-10)
        df[f"dist_low_{w}"] = (close - low_n) / (close + 1e-10)

    # ATR / ADX
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    df["atr_pct"] = atr14 / (close + 1e-10)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 1e-10)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 1e-10)
    df["di_diff"] = plus_di - minus_di
    df["adx"] = (plus_di - minus_di).abs().rolling(14).mean()

    # 标签
    df["label"] = close.shift(-LABEL_HORIZON) / close - 1

    return df


def compute_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """计算截面特征。"""
    result = df.copy()

    for col in ["return_1d", "return_5d", "return_20d"]:
        if col in result.columns:
            result[f"{col}_rank"] = result.groupby("date")[col].rank(pct=True)

    if "volume" in result.columns:
        result["volume_rank"] = result.groupby("date")["volume"].rank(pct=True)

    if "volatility_20" in result.columns:
        result["volatility_20_rank"] = result.groupby("date")["volatility_20"].rank(pct=True)

    if "return_1d" in result.columns:
        result["market_up_ratio"] = result.groupby("date")["return_1d"].transform(
            lambda x: (x > 0).sum() / max(len(x), 1)
        )

    if "return_5d" in result.columns:
        mkt_5d = result.groupby("date")["return_5d"].transform("mean")
        result["relative_strength_5d"] = result["return_5d"] - mkt_5d

    if "return_20d" in result.columns:
        mkt_20d = result.groupby("date")["return_20d"].transform("mean")
        result["relative_strength_20d"] = result["return_20d"] - mkt_20d

    if "relative_strength_5d" in result.columns:
        result["relative_momentum"] = (
            result["relative_strength_5d"] - result["relative_strength_20d"]
        )

    return result


def build_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """构建完整特征矩阵。"""
    logger.info("计算时序特征...")
    groups = []
    stock_codes = raw_df["stock_code"].unique()
    total = len(stock_codes)

    for i, code in enumerate(stock_codes):
        if (i + 1) % 200 == 0:
            logger.info("  时序特征进度: %d/%d", i + 1, total)
        stock_df = raw_df[raw_df["stock_code"] == code].copy()
        stock_df = stock_df.sort_values("date").reset_index(drop=True)
        feat_df = compute_ts_features(stock_df)
        groups.append(feat_df)

    combined = pd.concat(groups, ignore_index=True)
    logger.info("时序特征完成: %d 行", len(combined))

    logger.info("计算截面特征...")
    combined = compute_cross_sectional(combined)
    logger.info("截面特征完成")

    return combined


# ============================================================
# 数据分割
# ============================================================

@dataclass
class DataSplit:
    """数据分割结果。"""
    X_train: np.ndarray
    y_train: np.ndarray
    X_valid: np.ndarray
    y_valid: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_meta: pd.DataFrame


def split_data(df: pd.DataFrame) -> DataSplit:
    """按时间分割数据。"""
    df = df.dropna(subset=["label"])

    # 确保所有特征列存在
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0.0

    train_df = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)]
    valid_df = df[(df["date"] >= VALID_START) & (df["date"] <= VALID_END)]
    test_df = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)]

    logger.info(
        "数据分割: train=%d, valid=%d, test=%d",
        len(train_df), len(valid_df), len(test_df),
    )

    X_train = train_df[FEATURE_NAMES].fillna(0).values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.float32)
    X_valid = valid_df[FEATURE_NAMES].fillna(0).values.astype(np.float32)
    y_valid = valid_df["label"].values.astype(np.float32)
    X_test = test_df[FEATURE_NAMES].fillna(0).values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.float32)
    test_meta = test_df[["stock_code", "date", "close"]].copy()

    return DataSplit(
        X_train=X_train, y_train=y_train,
        X_valid=X_valid, y_valid=y_valid,
        X_test=X_test, y_test=y_test,
        test_meta=test_meta,
    )


# ============================================================
# 模型训练
# ============================================================

def train_lightgbm(data: DataSplit) -> lgb.LGBMRegressor:
    """训练 LightGBM。"""
    logger.info("训练 LightGBM...")
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        objective="regression",
        metric="mse",
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=128,
        max_depth=7,
        min_child_samples=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=10.0,
        reg_lambda=10.0,
        verbose=-1,
    )
    model.fit(
        data.X_train, data.y_train,
        eval_set=[(data.X_valid, data.y_valid)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100),
        ],
    )
    logger.info("LightGBM 完成, best_iteration=%d", model.best_iteration_)
    return model


def train_xgboost(data: DataSplit) -> xgb.XGBRegressor:
    """训练 XGBoost。"""
    logger.info("训练 XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=1000,
        objective="reg:squarederror",
        eval_metric="rmse",
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=10.0,
        reg_lambda=10.0,
        early_stopping_rounds=50,
    )
    model.fit(
        data.X_train, data.y_train,
        eval_set=[(data.X_valid, data.y_valid)],
        verbose=100,
    )
    logger.info("XGBoost 完成, best_iteration=%d", model.best_iteration)
    return model


# ============================================================
# 评估 & 回测
# ============================================================

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """回归评估指标。"""
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    ic = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"mse": round(mse, 8), "r2": round(r2, 6), "ic": round(ic, 6)}


def simulate_topk_backtest(data: DataSplit, predictions: np.ndarray) -> dict:
    """TopK 选股回测模拟。"""
    meta = data.test_meta.copy()
    meta["pred"] = predictions
    meta["actual_return"] = data.y_test

    top_k = 5
    daily_returns = []

    for date, group in meta.groupby("date"):
        if len(group) < top_k:
            continue
        top_stocks = group.nlargest(top_k, "pred")
        avg_return = top_stocks["actual_return"].mean()
        daily_returns.append(avg_return)

    if not daily_returns:
        return {"sharpe_ratio": 0, "total_return": 0, "max_drawdown": 0}

    period_returns = np.array(daily_returns)
    cumulative = np.cumprod(1 + period_returns)
    total_return = float(cumulative[-1] - 1)

    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    max_drawdown = float(drawdown.min())

    # 年化夏普（每个 period ~5 天，一年 ~48 个 period）
    periods_per_year = 48
    mean_ret = float(np.mean(period_returns))
    std_ret = float(np.std(period_returns))
    sharpe = (mean_ret / (std_ret + 1e-10)) * np.sqrt(periods_per_year)
    win_rate = float(np.mean(period_returns > 0))

    return {
        "sharpe_ratio": round(float(sharpe), 4),
        "total_return": round(total_return, 4),
        "max_drawdown": round(max_drawdown, 4),
        "win_rate": round(win_rate, 4),
        "num_periods": len(period_returns),
        "mean_period_return": round(mean_ret, 6),
    }


# ============================================================
# 保存
# ============================================================

def save_models(lgb_model: lgb.LGBMRegressor, xgb_model: xgb.XGBRegressor) -> None:
    """保存模型。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "lgb_model.pkl", "wb") as f:
        pickle.dump(lgb_model, f)
    with open(MODEL_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    logger.info("模型已保存到 %s", MODEL_DIR)


def save_feature_importance(lgb_model: lgb.LGBMRegressor) -> None:
    """保存特征重要性。"""
    fi_df = pd.DataFrame({
        "feature": FEATURE_NAMES,
        "importance": lgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
    logger.info("Top 15 特征:")
    for _, row in fi_df.head(15).iterrows():
        logger.info("  %s: %d", row["feature"], row["importance"])


def save_report(report: dict) -> None:
    """保存训练报告。"""
    with open(MODEL_DIR / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("训练报告已保存")


# ============================================================
# 主流程
# ============================================================

def main() -> None:
    """主训练流程。"""
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("LightGBM + XGBoost 训练开始")
    logger.info("=" * 60)

    # 1. 加载 & 过滤
    raw_df = load_and_filter_stocks()

    # 2. 特征工程
    feature_df = build_features(raw_df)

    # 3. 分割
    data = split_data(feature_df)

    # 4. 训练
    lgb_model = train_lightgbm(data)
    xgb_model = train_xgboost(data)

    # 5. 预测 & 评估
    lgb_pred = lgb_model.predict(data.X_test)
    xgb_pred = xgb_model.predict(data.X_test)
    ens_pred = 0.5 * lgb_pred + 0.5 * xgb_pred

    lgb_m = evaluate_predictions(data.y_test, lgb_pred)
    xgb_m = evaluate_predictions(data.y_test, xgb_pred)
    ens_m = evaluate_predictions(data.y_test, ens_pred)
    logger.info("LGB 测试: %s", lgb_m)
    logger.info("XGB 测试: %s", xgb_m)
    logger.info("ENS 测试: %s", ens_m)

    # 6. 回测模拟
    lgb_bt = simulate_topk_backtest(data, lgb_pred)
    xgb_bt = simulate_topk_backtest(data, xgb_pred)
    ens_bt = simulate_topk_backtest(data, ens_pred)
    logger.info("LGB 回测: %s", lgb_bt)
    logger.info("XGB 回测: %s", xgb_bt)
    logger.info("ENS 回测: %s", ens_bt)

    # 7. 保存
    save_models(lgb_model, xgb_model)
    save_feature_importance(lgb_model)

    elapsed = round(time.time() - t0, 1)
    report = {
        "training_time_seconds": elapsed,
        "data_info": {
            "total_stocks": len(raw_df["stock_code"].unique()),
            "train_samples": len(data.y_train),
            "valid_samples": len(data.y_valid),
            "test_samples": len(data.y_test),
            "feature_count": len(FEATURE_NAMES),
        },
        "lgb_best_iteration": lgb_model.best_iteration_,
        "xgb_best_iteration": xgb_model.best_iteration,
        "test_metrics": {
            "lightgbm": lgb_m,
            "xgboost": xgb_m,
            "ensemble": ens_m,
        },
        "backtest_simulation": {
            "lightgbm": lgb_bt,
            "xgboost": xgb_bt,
            "ensemble": ens_bt,
        },
    }
    save_report(report)

    logger.info("=" * 60)
    logger.info("训练完成! 耗时: %.1f 秒", elapsed)
    logger.info("Ensemble 夏普率: %.4f", ens_bt["sharpe_ratio"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
