# Reversal Neutral V2

**反转因子市场中性策略 - 低换手版**

## V2 改进

| 指标 | V1 | V2 | 改进 |
|------|----|----|------|
| 调仓频率 | 每天 | **每5天** | - |
| 换手率 | 54% | **16%** | **-70%** ✅ |
| 夏普比率 | 2.50 | 2.24 | 略降 |
| 累计收益 | +285% | +238% | 略降 |

**核心改进**：大幅降低换手率，减少交易成本

## 回测表现

| 指标 | 数值 |
|------|------|
| 回测期 | 2023-07 ~ 2024-12 |
| 累计收益 | **+238%** |
| 夏普比率 | **2.24** |
| 最大回撤 | -51.4% |
| 换手率 | **16.5%** |

## 使用方法

```python
from signal_generator import ReversalNeutralV2

generator = ReversalNeutralV2(config={
    'top_n': 10,
    'rebalance_days': 5  # 每5天调仓
})

generator.load_data('/path/to/stock_data')
generator.train()

# 生成信号（自动判断是否需要调仓）
signal = generator.generate_signals()
print(f"是否调仓: {signal['rebalanced']}")
print(f"下次调仓: {signal['next_rebalance']}")
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| top_n | 10 | 多空各选股数量 |
| rebalance_days | 5 | 调仓频率（天） |
| train_months | 6 | 训练窗口 |
| cost_per_trade | 0.003 | 单边交易成本 |

## 版本历史

- **v2.0.0** (2024-02-04)：降低调仓频率，换手率从 54% 降到 16%
- **v1.0.0** (2024-02-04)：首个版本
