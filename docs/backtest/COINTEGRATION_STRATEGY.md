# 协整策略（当前实现）的原理说明（超细版）

本文档描述 **当前代码中的 `CointegrationStrategy` 实际实现逻辑**，用于打印与研究。该实现本质上是 **单标的均值回归 + 半衰期过滤**，并非经典“配对协整交易”。

对应代码位置：
- `backend/app/services/backtest/strategies/strategies.py`
- 类：`CointegrationStrategy`

---

## 1. 策略定位与目标（逐条拆解）

当前实现试图在单只股票上捕捉“价格回归均值”的行为，拆开来看就是：

- **先判断价格是否偏离“自身均值”**（偏离用 z-score 表示）。
- **再判断这种偏离是否“容易回归”**（用收益率估计的半衰期/回归强度过滤）。
- **若偏离开始回穿阈值且回归性为真**，产生买卖信号。

因此，它更接近“单标的均值回归”而非严格意义的“协整配对”。

---

## 2. 输入数据与基本约束（逐条拆解）

输入数据：
- `data: pd.DataFrame`，需要包含 `close` 列。
- `data.index` 为交易日期（`datetime`）。

在信号生成时：
- 若 `current_idx < lookback_period`，直接返回空信号（滚动均值/方差不可靠）。
- 在回测主循环中还要求历史数据长度 >= 20（见 `backtest_executor.py`）。

换句话说：**需要足够长的历史窗口**，才能计算出“均值、波动、z-score、半衰期”。

---

## 3. 参数与默认值（逐个解释）

来自策略初始化 `__init__`：

- `lookback_period`：滚动均值/波动窗口（默认 60）
- `half_life`：半衰期默认值（默认 20）
- `entry_threshold`：入场阈值（默认 2.0）
- `exit_threshold`：出场阈值（默认 0.5）**注意：当前实现未使用**

解释拆开说：

- `lookback_period` 越大：均值与波动更平滑，信号更少但更稳；越小：更敏感、噪声也更多。
- `half_life` 是回归速度的“保底值”，当估计失败时会回落到它。
- `entry_threshold` 决定“偏离程度”，比如 2.0 就是“2 个标准差”的偏离。
- `exit_threshold` 在当前版本没有生效，所以**真正的出场逻辑还是基于 entry 回穿**。

---

## 4. 指标计算逻辑（calculate_indicators，逐行拆解）

### 4.1 价格与收益率

```text
close_prices = data['close']
returns = close_prices.pct_change().dropna()
```

含义拆开：
- `close_prices`：价格时间序列。
- `returns`：收益率序列（用 `(p_t / p_{t-1} - 1)` 计算）。
- `dropna()`：去掉第一个空值。

### 4.2 半衰期估计（_estimate_half_life）

逻辑：
- 取最近最多 252 个收益率样本。
- 用 OLS 回归：`returns_t = alpha + beta * returns_{t-1}`。
- 若 `beta >= 0`，认为无回归性，返回默认 `half_life`。
- 否则按公式：

```text
half_life = -ln(2) / beta
```

并裁剪到 `[1, 252]`。

更细一点解释：
- 如果 `beta < 0`，代表收益率有“反向关系”（上一期高收益 → 下一期低收益），即**回归性**。
- `beta` 越负，`half_life` 越短，意味着回归更快。
- 如果 `beta >= 0`，说明收益率更像“趋势/随机游走”，不强调回归，于是退回默认 `half_life`。

> 说明：这里的半衰期是基于“收益率自身的回归性”，不是协整价差的 OU 半衰期。

### 4.3 均值与波动、z-score

```text
sma = close_prices.rolling(window=lookback_period).mean()
std = close_prices.rolling(window=lookback_period).std()
zscore = (close_prices - sma) / (std + 0.001)
```

拆开：
- `sma`：滚动均值（过去 `lookback_period` 天的平均价）。
- `std`：滚动标准差（过去 `lookback_period` 天的波动）。
- `zscore`：价格偏离均值的“标准化距离”。
- `+0.001`：防止 `std=0` 时除零。

### 4.4 均值回归强度

```text
mean_reversion_strength = -ln(2) / half_life
```

拆开：
- `half_life` 越短，`mean_reversion_strength` 绝对值越大（回归越快）。
- 由于 `half_life` 来自 `beta < 0` 的情况，`mean_reversion_strength` 通常为负。
- 代码里用 `mean_reversion_strength < 0` 作为“允许交易”的过滤条件。

---

## 5. 信号生成逻辑（generate_signals，细化流程）

关键变量（当前时点 `current_date`）：

- `current_zscore`
- `prev_zscore`
- `mean_reversion_strength`

### 5.1 买入信号（BUY）

触发条件：

```text
prev_zscore <= -entry_threshold
AND current_zscore > -entry_threshold
AND mean_reversion_strength < 0
```

逐句解释：
- `prev_zscore <= -entry_threshold`：上一期“显著低于均值”。
- `current_zscore > -entry_threshold`：当前期已经开始向均值回穿。
- `mean_reversion_strength < 0`：收益率有回归性，允许交易。

直观含义：**价格超跌后回升**，且“回归特征”被确认。

### 5.2 卖出信号（SELL）

触发条件：

```text
prev_zscore >= entry_threshold
AND current_zscore < entry_threshold
AND mean_reversion_strength < 0
```

逐句解释：
- `prev_zscore >= entry_threshold`：上一期“显著高于均值”。
- `current_zscore < entry_threshold`：当前期已经向均值回穿。
- `mean_reversion_strength < 0`：收益率回归性存在。

直观含义：**价格超涨后回落**，且“回归特征”被确认。

### 5.3 信号强度

```text
strength = min(1.0, abs(current_zscore) / entry_threshold)
```

拆开：
- `abs(current_zscore)` 越大，说明偏离越大。
- 用阈值归一化，保证强度不超过 1。

### 5.4 信号元信息（metadata）

信号会附带：
- `zscore`
- `half_life`
- `mean_reversion_strength`
- `sma`

并在 `reason` 中包含 z-score 和半衰期。

---

## 6. 回测流程中如何被调用（逐步拆解）

在回测执行器 `backtest_executor.py` 中：

- 每个交易日遍历每只股票
- 取 “截至当前日的历史数据”
- 调用 `strategy.generate_signals(historical_data, current_date)`

因此该策略是 **逐股独立计算**，不存在跨股票配对或协整组合。

---

## 7. 与标准“协整配对”差异（逐条拆解）

当前实现与标准协整策略主要差异：

1. **单标的**：不包含第二序列，缺少配对与 hedge ratio。
2. **无协整检验**：未使用 Engle-Granger / Johansen / ADF 等检验。
3. **信号基于价格偏离**：使用单标的 `price - SMA` 的 z-score。
4. **半衰期是收益率 AR(1)**：不是对价差/残差的 OU 回归估计。

结论：当前实现是“单标的均值回归”，并非严格意义的“协整交易”。

---

## 8. 读者要点（打印版速读）

- 这是“单标的均值回归 + 半衰期过滤”的策略。
- `entry_threshold` 用于“回穿入场”，不是突破顺势。
- `exit_threshold` 目前未被使用。
- 半衰期是基于收益率回归估计，更多是过滤器。
- 该策略适合震荡/回归阶段，趋势行情容易失效。

---

## 9. 图示（统计图风格）

### 9.1 价格与滚动均值（SMA）

![Price vs SMA](assets/price_sma.svg)

### 9.2 Z-score 阈值与买入回穿

![Z-score Buy Cross](assets/zscore_buy.svg)

### 9.3 Z-score 阈值与卖出回穿

![Z-score Sell Cross](assets/zscore_sell.svg)

### 9.4 半衰期与 beta 的关系（收益率 AR(1)）

![Half-life Filter](assets/half_life_filter.svg)

