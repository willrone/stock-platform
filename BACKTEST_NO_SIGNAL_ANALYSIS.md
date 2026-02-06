# 回测任务无交易信号问题分析报告

**任务ID**: `e4f9608f-89d2-495b-b1a0-af9a1e5142ab`  
**任务名称**: mixed  
**股票数量**: 1000只  
**执行时间**: 2026-02-05 13:08:23 - 13:18:40 (约11分钟)  
**状态**: cancelled  
**问题**: 回测完成但没有产生任何交易信号

---

## 问题根因

### 1. 策略配置不当

任务使用了**portfolio组合策略**，包含三个子策略：

```python
{
    "strategies": [
        {"name": "bollinger", "weight": 1},      # 布林带策略
        {"name": "rsi", "weight": 1},            # RSI策略
        {"name": "cointegration", "weight": 2}   # 协整策略 ⚠️
    ],
    "integration_method": "weighted_voting"
}
```

**问题所在**：协整策略（Cointegration）权重最高（2），但它**不适用���单股票回测**！

### 2. 协整策略的工作原理

协整策略是一种**统计套利策略**，用于配对交易（Pairs Trading）：

- **需要两只或多只股票**的价格序列
- 通过计算股票间的**协整关系**来判断价差是否偏离均衡
- 当价差偏离时做多低估股票、做空高估股票
- 等待价差回归均值时平仓获利

**关键代码**：
```python
class StatisticalArbitrageStrategy(BaseStrategy):
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.pairs = config.get("pairs", [])  # ← 需要配对列表！
```

### 3. 当前回测架构的限制

Willrone回测系统采用**单股票独立处理**架构：

```python
# 数据加载：逐只股票加载
for stock_code in stock_codes:
    data = load_stock_data(stock_code)
    
# 策略执行：每只股票独立生成信号
for stock_code, data in stock_data.items():
    signals = strategy.generate_signals(data, current_date)
```

在这种架构下，协整策略：
1. 无法获取配对股票的数据
2. 只能基于单只股票的价格计算Z-score
3. 均值回归强度默认为0

**信号生成条件**：
```python
# 协整策略只在均值回归强度为负时生成信号
buy_mask &= (mean_rev < 0)
sell_mask &= (mean_rev < 0)

# 但在单股票场景下
mean_reversion_strength = pd.Series(0.0, index=close_prices.index)
# ↑ 默认为0，导致所有信号被过滤！
```

### 4. 加权投票机制的影响

Portfolio策略使用`weighted_voting`整合子策略信号：

```
总权重 = 布林带(1) + RSI(1) + 协整(2) = 4
协整占比 = 2/4 = 50%
```

即使布林带和RSI产生信号，协整策略的**0信号**会严重拉低整体得分，导致：
- 信号强度不足
- 无法通过一致性阈值（consistency_threshold=0.6）
- 最终不产生交易信号

---

## 验证证据

### 日志分析

1. **数据加载正常**：
```
[INFO] 并行加载 1000 只股票数据，使用 10 个线程
[INFO] 从Parquet加载股票数据: 000977.SZ, 数据量: 992
```

2. **策略配置确认**：
```
[INFO] 策略配置 (strategy_config): {
    'strategies': [
        {'name': 'bollinger', 'weight': 1, ...},
        {'name': 'rsi', 'weight': 1, ...},
        {'name': 'cointegration', 'weight': 2, ...}  # ← 问题策略
    ]
}
```

3. **结果为空**：
```
[INFO] 成功获取组合快照: task_id=..., count=0  # ← 无交易
[INFO] 成功获取信号记录: ..., count=0           # ← 无信号
```

### 数据库查询

```sql
SELECT result FROM tasks WHERE task_id = 'e4f9608f-89d2-495b-b1a0-af9a1e5142ab';
-- result: NULL (无回测结果)
```

---

## 解决方案

### 方案1：移除协整策略（推荐）

**适用场景**：单股票回测

```python
{
    "strategy_name": "portfolio",
    "strategy_config": {
        "strategies": [
            {
                "name": "bollinger",
                "weight": 0.5,
                "config": {
                    "period": 20,
                    "std_dev": 2,
                    "entry_threshold": 0.02
                }
            },
            {
                "name": "rsi",
                "weight": 0.5,
                "config": {
                    "rsi_period": 14,
                    "oversold_threshold": 30,
                    "overbought_threshold": 70,
                    "trend_ma_period": 50,
                    "enable_trend_alignment": true
                }
            }
        ],
        "integration_method": "weighted_voting"
    }
}
```

**优点**：
- ✅ 立即可用，无需修改代码
- ✅ 策略逻辑清晰，适合单股票场景
- ✅ 布林带+RSI组合已被广泛验证

### 方案2：使用单策略回测

**适用场景**：快速验证单个策略效果

```python
{
    "strategy_name": "rsi",  # 或 "bollinger"
    "strategy_config": {
        "rsi_period": 14,
        "oversold_threshold": 30,
        "overbought_threshold": 70
    }
}
```

### 方案3：改造协整策略为均值回归策略

**适用场景**：需要保留均值回归逻辑

修改协整策略，使其在单股票场景下：
1. 不依赖配对股票
2. 基于单股票价格的均值回归特性
3. 使用Ornstein-Uhlenbeck过程估计

**需要修改代码**：
```python
class CointegrationStrategy(StatisticalArbitrageStrategy):
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        # 修改：不再要求配对，改为单股票均值回归
        close_prices = data["close"]
        returns = close_prices.pct_change()
        
        # 估计均值回归速度（OU过程）
        half_life = self._estimate_half_life(returns)
        
        # 只有当half_life有效时才计算均值回归强度
        if half_life > 0 and half_life < 252:  # 半衰期在1年内
            mean_reversion_strength = -np.log(2) / half_life
        else:
            mean_reversion_strength = 0.0  # 无效时设为0
        
        # ... 其余逻辑
```

### 方案4：升级回测架构支持配对交易

**适用场景**：需要真正的配对交易策略

**需要大量改造**：
1. 修改数据加载器，支持多股票联合加载
2. 修改策略接口，支持多股票输入
3. 修改信号生成逻辑，支持配对信号
4. 修改回测引擎，支持多股票联合回测

**工作量大，不推荐作为短期方案**

---

## 修复步骤

### 立即修复（方案1）

1. **运行修复脚本**：
```bash
cd ~/Documents/GitHub/willrone
python3 fix_portfolio_strategy.py
```

2. **重新创建回测任务**：
```bash
curl -X POST http://localhost:8000/api/v1/tasks/backtest \
  -H "Content-Type: application/json" \
  -d @fixed_config.json
```

3. **验证结果**：
- 检查任务是否产生交易信号
- 查看回测指标（收益率、夏普比率等）
- 对比基准任务（task_id: 814287d1-202c-4109-a746-c932206bd840）

### 长期优化

1. **添加策略验证**：
   - 在任务创建时检查策略配置
   - 协整策略必须提供`pairs`配置
   - 单股票回测禁用协整策略

2. **改进错误提示**：
   - 当协整策略无法生成信号时，记录警告日志
   - 在任务结果中说明原因

3. **文档完善**：
   - 在策略文档中明确说明协整策略的使用场景
   - 提供portfolio策略的最佳实践配置

---

## 总结

### 问题本质
协整策略是**配对交易策略**，需要多只股票的联合数据，但当前回测系统采用**单股票独立处理**架构，导致协整策略无法生成有效信号。

### 影响范围
- 所有使用portfolio策略且包含cointegration的回测任务
- 预估影响：可能有多个类似任务存在相同问题

### 推荐方案
**移除协整策略**，使用布林带+RSI组合，这是最快速、最可靠的解决方案。

### 预期效果
修复后，1000只股票的回测任务应该能够：
- 产生正常的交易信号
- 生成完整的回测报告
- 性能与基准任务（500股票6分钟）相当

---

**报告生成时间**: 2026-02-05 21:22  
**分析人员**: OpenClaw Subagent  
**任务标签**: backtest-debug
