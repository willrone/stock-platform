# LightGBM + XGBoost 训练方案

## 1. 目标

训练 LightGBM + XGBoost 双模型集成，在 Willrone 回测中实现夏普率 > 1。

## 2. 数据

- **来源**: `/data/parquet/stock_data/` 下 5479 只 A 股 OHLCV 日线数据
- **训练集**: 2018-01-01 ~ 2023-12-31（6年）
- **验证集**: 2024-01-01 ~ 2024-06-30（6个月）
- **测试集**: 2024-07-01 ~ 2025-12-31（18个月，用于回测验证）
- **筛选**: 仅使用日均成交量 > 500万、上市满1年的股票，排除 ST 股

## 3. 标签设计

**回归标签**: 未来5日收益率（与 Qlib Alpha158 一致）

```
label = close[t+5] / close[t] - 1
```

**理由**:
- 回归比分类保留更多信息
- 5日收益率平滑了日内噪声
- 与 `ml_ensemble_strategy.py` 的 `prediction_horizon=5` 一致

## 4. 特征工程（62个特征）

与 `ml_ensemble_strategy.py._get_feature_names()` 完全对齐：

### 4.1 时序特征（53个）

| 类别 | 特征 | 数量 |
|------|------|------|
| 收益率 | return_1d/2d/3d/5d/10d/20d | 6 |
| 动量 | momentum_short/long/reversal, momentum_strength_5/10/20 | 6 |
| 移动平均 | ma_ratio_5/10/20/60, ma_slope_5/10/20, ma_alignment | 8 |
| 波动率 | volatility_5/20/60, vol_regime, volatility_skew | 5 |
| 成交量 | vol_ratio, vol_ma_ratio, vol_std, vol_price_diverge | 4 |
| RSI | rsi_6, rsi_14, rsi_diff | 3 |
| MACD | macd, macd_signal, macd_hist, macd_hist_slope | 4 |
| 布林带 | bb_position, bb_width | 2 |
| 价格形态 | body, wick_upper/lower, range_pct, consecutive_up/down | 6 |
| 价格位置 | price_pos_20/60, dist_high/low_20/60 | 6 |
| ATR/趋势 | atr_pct, di_diff, adx | 3 |

### 4.2 截面特征（9个）

| 类别 | 特征 | 数量 |
|------|------|------|
| 相对强度 | relative_strength_5d/20d, relative_momentum | 3 |
| 排名 | return_1d/5d/20d_rank, volume_rank, volatility_20_rank | 5 |
| 市场状态 | market_up_ratio | 1 |

## 5. 训练参数

### LightGBM
```python
lgb_params = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 128,
    'max_depth': 7,
    'min_child_samples': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 10.0,      # L1 正则化（较强，防过拟合）
    'reg_lambda': 10.0,     # L2 正则化
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
    'verbose': -1,
}
```

### XGBoost
```python
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 10.0,
    'reg_lambda': 10.0,
    'n_estimators': 1000,
    'early_stopping_rounds': 50,
}
```

### 防过拟合策略
1. **强正则化**: reg_alpha=10, reg_lambda=10
2. **大 min_child_samples**: 200（避免学习噪声）
3. **时序交叉验证**: 滚动窗口，训练集不能看到未来数据
4. **Embargo**: 训练集和验证集之间留 5 天间隔
5. **特征标准化**: 截面 Z-Score 标准化

## 6. 交叉验证方式

**滚动窗口时序交叉验证（Expanding Window）**:

```
Fold 1: Train [2018-01 ~ 2020-12] → Valid [2021-01 ~ 2021-06]
Fold 2: Train [2018-01 ~ 2021-06] → Valid [2021-07 ~ 2021-12]
Fold 3: Train [2018-01 ~ 2021-12] → Valid [2022-01 ~ 2022-06]
Fold 4: Train [2018-01 ~ 2022-06] → Valid [2022-07 ~ 2022-12]
Fold 5: Train [2018-01 ~ 2022-12] → Valid [2023-01 ~ 2023-06]
Fold 6: Train [2018-01 ~ 2023-06] → Valid [2023-07 ~ 2023-12]
```

最终模型: Train [2018-01 ~ 2023-12] → Valid [2024-01 ~ 2024-06]

## 7. 选股策略（回测时）

1. 每日对所有股票预测未来5日收益率
2. 选择预测收益率 Top N（默认 N=5）的股票
3. 等权重分配仓位
4. 每5个交易日调仓一次
5. 风控：止损 2%、波动率缩放、市场过滤

## 8. 输出

- `data/models/lgb_model.pkl` - LightGBM 模型
- `data/models/xgb_model.pkl` - XGBoost 模型
- `data/models/training_report.json` - 训练报告
- `data/models/feature_importance.csv` - 特征重要性
