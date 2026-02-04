# Reversal Neutral V4

**反转因子市场中性策略 - 趋势增强版**

## V4 突破

| 指标 | V2 | V4 | 改进 |
|------|----|----|------|
| 累计收益 | +783% | **+1285%** | **+64%** ✅ |
| 夏普比率 | 3.97 | **4.47** | **+13%** ✅ |
| 最大回撤 | -45.3% | **-41.5%** | **-8%** ✅ |

## 核心创新

**趋势跟随仓位管理**：
- 上涨趋势（近 10 天平均收益 > 1%）→ 加仓至 1.2x
- 下跌趋势（近 10 天平均收益 < -1%）→ 减仓至 0.8x
- 震荡市场 → 保持 1.0x

**原理**：
- 反转策略在趋势市场中表现更好
- 上涨趋势时市场情绪好，反转效应更强
- 下跌趋势时减仓控制风险

## 回测表现

- **回测期**：2023-07 ~ 2024-12
- **累计收益**：+1285%
- **年化夏普**：4.47
- **最大回撤**：-41.5%
- **2023 年**：+20%
- **2024 年**：+1058%

## 使用方法

```python
from strategies.reversal_neutral_v4.signal_generator import ReversalNeutralV4

generator = ReversalNeutralV4()
generator.load_data('/path/to/stock_data')
generator.train()

# 生成信号（自动判断趋势）
signal = generator.generate_signals()
print(f"做多: {signal['long']}")
print(f"做空: {signal['short']}")
print(f"仓位系数: {signal['position_factor']:.2f}")  # 0.8 / 1.0 / 1.2
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| trend_window | 10 | 趋势判断窗口（天） |
| trend_up_factor | 1.2 | 上涨趋势加仓倍数 |
| trend_down_factor | 0.8 | 下跌趋势减仓倍数 |
| trend_threshold | 0.01 | 趋势判断阈值（1%） |

## 版本历史

- **v4.0.0** (2024-02-04)：加入趋势跟随仓位管理，收���提升 64%
