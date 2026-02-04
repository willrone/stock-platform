# Reversal Neutral 策略系列

**反转因子市场中性策略** - A 股量化交易策略

## 策略原理

利用 A 股市场的**反转效应**：
- 短期超跌的股票倾向于反弹
- 短期超涨的股票倾向于回调

通过**市场中性**设计消除 beta 暴露：
- 做多 Top N 超跌股票
- 做空 Bottom N 超涨股票
- 只赚取选股 alpha，熊市也能盈利

## 版本对比

| 版本 | 配置 | 收益 | 夏普 | 回撤 | 换手率 | 特点 |
|------|------|------|------|------|--------|------|
| V1 | Top10 每天 | +548% | 3.21 | -37% | 54% | 基础版 |
| **V2** | **Top10 每5天** | **+783%** | **3.97** | -45% | **16%** | **推荐** |
| V3 | Top15 每5天 | +348% | 3.34 | -45% | 16% | 稳健版 |

**推荐使用 V2**：最高收益、最高夏普、低换手率

## 核心因子

| 因子 | 说明 | 重要性 |
|------|------|--------|
| reversal_5d/10d/20d | 短期反转（过去收益取负） | ⭐⭐⭐ |
| drawdown_60d | 超跌程度（距离 60 日高点） | ⭐⭐⭐ |
| runup_20d | 超涨程度（距离低点涨幅） | ⭐⭐ |
| volatility_20d | 20 日波动率 | ⭐⭐⭐ |
| macd | MACD 反转信号 | ⭐⭐ |
| vol_ratio | 成交量比率 | ⭐ |

## 回测表现（V2）

- **回测期**：2023-07 ~ 2024-12
- **累计收益**：+783%
- **年化夏普**：3.97
- **最大回撤**：-45%
- **2023 年**：+22%（熊市盈利）
- **2024 年**：+624%

## 使用方法

```python
from strategies.reversal_neutral_v2.signal_generator import ReversalNeutralV2

# 初始化
generator = ReversalNeutralV2()

# 加载数据
generator.load_data('/path/to/stock_data')

# 训练
generator.train(train_end_date='2024-01-01')

# 生成信号
signal = generator.generate_signals()
print(f"做多: {signal['long']}")
print(f"做空: {signal['short']}")
print(f"下次调仓: {signal['next_rebalance']}")

# 回测
result = generator.backtest('2024-01-01', '2024-12-31')
print(f"收益: {result['cum_return']*100:.1f}%")
```

## 注意事项

1. **做空限制**：A 股做空需要融券，实际操作受限
2. **交易成本**：默认 0.3% 单边，包含手续费和滑点
3. **数据要求**：需要日线 OHLCV 数据
4. **再训练**：建议每 3-6 个月重新训练模型

## 文件结构

```
strategies/
├── reversal_neutral_v1/    # 基础版（每天调仓）
├── reversal_neutral_v2/    # 推荐版（每5天调仓）
└── reversal_neutral_v3/    # 稳健版（Top15）
```

## 版本历史

- **2024-02-04**：发布 V1/V2/V3 三个版本
