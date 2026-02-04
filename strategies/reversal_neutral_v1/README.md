# Reversal Neutral V1

**反转因子市场中性策略** - 第一个迭代版本

## 策略概述

基于 A 股市场的反转效应设计的市场中性策略：
- **做多**：超跌股票（预期反弹）
- **做空**：超涨股票（预期回调）
- **市场中性**：消除 beta 暴露，只赚取选股 alpha

## 回测表现

| 指标 | 数值 |
|------|------|
| 回测期 | 2023-07 ~ 2024-12 |
| 累计收益 | **+285%** |
| 夏普比率 | **2.35** |
| 最大回撤 | -41.7% |
| 平均换手 | 56.6% |
| 交易成本 | 0.3%/单边 |

### 分年表现
- 2023 年：+21%（熊市也能盈利）
- 2024 年：+396%

## 核心因子

| 因子 | 说明 |
|------|------|
| reversal_5d/10d/20d | 短期反转（过去收益取负） |
| drawdown_60d | 超跌程度（距离 60 日高点） |
| runup_10d/20d | 超涨程度（距离低点涨幅，取负） |
| macd | MACD 反转信号 |
| volatility_20d | 20 日波动率 |
| vol_ratio | 成交量比率 |

## 使用方法

### Python API

```python
from signal_generator import ReversalNeutralV1

# 初始化
generator = ReversalNeutralV1(config={'top_n': 10})

# 加载数据
generator.load_data('/path/to/stock_data')

# 训练模型
generator.train(train_end_date='2024-01-01')

# 生成信号
signal = generator.generate_signals()
print(f"做多: {signal['long']}")
print(f"做空: {signal['short']}")

# 回测
result = generator.backtest('2024-01-01', '2024-12-31')
print(f"收益: {result['cum_return']*100:.1f}%")

# 保存/加载模型
generator.save('/path/to/save')
generator.load('/path/to/load')
```

### 命令行

```bash
# 训练并保存模型
python signal_generator.py \
    --data-dir /path/to/stock_data \
    --action train \
    --date 2024-01-01 \
    --save-dir ./model

# 生成信号
python signal_generator.py \
    --data-dir /path/to/stock_data \
    --action signal \
    --load-dir ./model

# 回测
python signal_generator.py \
    --data-dir /path/to/stock_data \
    --action backtest \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| top_n | 10 | 多头/空头各选股数量 |
| train_months | 6 | 训练窗口（月） |
| holding_days | 5 | 持仓周期（天） |
| cost_per_trade | 0.003 | 单边交易成本 |

## 文件结构

```
reversal_neutral_v1/
├── signal_generator.py  # 信号生成器主类
├── model.pkl           # 训练好的模型
├── config.json         # 配置文件
├── metadata.json       # 元数据
└── README.md           # 本文档
```

## 注意事项

1. **数据要求**：需要 parquet 格式的股票日线数据，包含 `date`, `open`, `high`, `low`, `close`, `volume` 字段
2. **做空限制**：A 股做空需要融券，实际操作可能受限
3. **交易成本**：默认 0.3% 单边，包含手续费和滑点
4. **再平衡频率**：每日调仓，换手率较高

## 版本历史

- **v1.0.0** (2024-02-04)：首个版本，基于反转因子的市场中性策略

## 待优化方向

1. 降低换手率（当前 56%）
2. 加入更多因子（基本面、资金流）
3. 动态调整 Top N
4. 加入止损机制
