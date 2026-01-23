# 量化交易回测平台使用指南

## 目录
1. [平台概述](#平台概述)
2. [回测流程详解](#回测流程详解)
3. [策略详解](#策略详解)
4. [使用方法](#使用方法)
5. [实战案例](#实战案例)

---

## 平台概述

### 平台简介
- **定位**: 专业的量化交易回测平台
- **技术栈**: 
  - 后端: FastAPI + SQLAlchemy + WebSocket
  - 前端: Next.js + React + TypeScript
  - 数据库: PostgreSQL
- **核心功能**: 策略回测、实时进度监控、结果分析

### 平台优势
- ✅ **多策略支持**: 涵盖技术分析、统计套利、因子投资三大类策略
- ✅ **实时监控**: WebSocket实时推送回测进度
- ✅ **组合策略**: 支持多策略组合，灵活配置权重
- ✅ **性能优化**: 支持并行计算，提升回测速度
- ✅ **数据完整**: 支持多种技术指标和因子计算

---

## 回测流程详解

### 整体流程图

```
用户提交回测任务
    ↓
创建任务记录 (状态: CREATED, 进度: 0%)
    ↓
加入后台执行队列
    ↓
┌─────────────────────────────────────┐
│  阶段1: 初始化 (0-10%)              │
│  - 生成回测ID                        │
│  - 启动进度监控                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  阶段2: 数据加载 (10-25%)           │
│  - 加载股票历史数据                  │
│  - 验证数据完整性                    │
│  - 支持并行加载 (股票数>3)          │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  阶段3: 策略设置 (25-30%)           │
│  - 创建交易策略实例                  │
│  - 初始化组合管理器                  │
│  - 配置初始资金、手续费、滑点       │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  阶段4: 回测执行 (30-90%)           │
│  - 获取交易日历                      │
│  - 逐日回测循环:                     │
│    • 获取当前价格                    │
│    • 生成交易信号                    │
│    • 验证并执行信号                  │
│    • 更新持仓和资金                  │
│    • 记录交易历史                    │
│  - 实时更新进度                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  阶段5: 结果计算 (90-95%)            │
│  - 计算收益率指标                    │
│  - 计算风险指标                      │
│  - 生成交易报告                      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  阶段6: 完成 (95-100%)               │
│  - 保存回测结果                      │
│  - 更新任务状态为COMPLETED          │
│  - 推送完成通知                      │
└─────────────────────────────────────┘
```

### 详细流程说明

#### 1. 任务创建阶段
**前端操作**:
- 填写任务名称
- 选择任务类型: "backtest"
- 选择股票代码列表
- 配置回测参数

**后端处理**:
```python
# 创建任务记录
task = {
    "task_id": "uuid",
    "task_name": "回测任务名称",
    "task_type": "backtest",
    "status": "CREATED",
    "progress": 0,
    "config": {
        "strategy_name": "moving_average",
        "stock_codes": ["000001", "000002"],
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "initial_cash": 100000,
        "commission_rate": 0.0003,
        "slippage_rate": 0.0001
    }
}
```

#### 2. 数据加载阶段
**数据来源**:
- 从 `backend/data/` 目录加载CSV文件
- 使用 `StockDataLoader` 统一加载器
- 支持并行加载 (股票数>3时自动启用)

**数据验证**:
- 检查必需列: `open`, `high`, `low`, `close`, `volume`
- 验证数据量是否充足 (至少20个交易日)
- 检查数据完整性

#### 3. 策略设置阶段
**策略创建**:
```python
# 使用策略工厂创建策略
strategy = StrategyFactory.create_strategy(
    strategy_name="moving_average",
    config={
        "short_window": 5,
        "long_window": 20
    }
)
```

**组合管理器初始化**:
- 初始资金: 默认100,000元
- 手续费率: 默认0.03% (0.0003)
- 滑点率: 默认0.01% (0.0001)

#### 4. 回测执行阶段 (核心)
**逐日回测循环**:

```python
for current_date in trading_dates:
    # 1. 获取当前价格
    current_prices = {
        stock_code: data.loc[current_date, 'close']
        for stock_code, data in stock_data.items()
    }
    
    # 2. 生成交易信号
    for stock_code, data in stock_data.items():
        historical_data = data[data.index <= current_date]
        signals = strategy.generate_signals(
            historical_data, 
            current_date
        )
    
    # 3. 验证并执行信号
    for signal in all_signals:
        if strategy.validate_signal(signal, ...):
            trade = portfolio_manager.execute_signal(
                signal, 
                current_prices[signal.stock_code]
            )
    
    # 4. 更新持仓和资金
    portfolio_manager.update_positions(current_prices)
    
    # 5. 记录交易历史
    backtest_records.append({
        "date": current_date,
        "signals": all_signals,
        "trades": executed_trades,
        "portfolio_value": portfolio_manager.total_value
    })
    
    # 6. 更新进度
    progress = 30 + (i / len(trading_dates)) * 60
    update_progress(progress)
```

#### 5. 结果计算阶段
**计算指标**:
- **收益率指标**: 总收益率、年化收益率、夏普比率
- **风险指标**: 最大回撤、波动率、VaR
- **交易统计**: 交易次数、胜率、平均持仓时间

#### 6. 完成阶段
- 保存回测结果到数据库
- 更新任务状态为 `COMPLETED`
- 通过WebSocket推送完成通知

---

## 策略详解

### 策略分类

#### 1. 基础技术分析策略

##### 1.1 移动平均策略 (MovingAverage)
**策略原理**:
- 使用短期和长期移动平均线的交叉来生成交易信号
- 金叉 (短期上穿长期): 买入信号
- 死叉 (短期下穿长期): 卖出信号

**关键参数**:
```python
{
    "short_window": 5,      # 短期均线周期 (默认5)
    "long_window": 20,     # 长期均线周期 (默认20)
    "signal_threshold": 0.02  # 信号阈值 (默认0.02)
}
```

**适用场景**:
- 趋势明显的市场
- 中长期持仓
- 适合波动较大的股票

**信号生成逻辑**:
```python
# 买入信号: 短期均线上穿长期均线
if prev_ma_diff <= 0 and current_ma_diff > 0:
    if abs(current_ma_diff) > signal_threshold:
        generate_buy_signal()

# 卖出信号: 短期均线下穿长期均线
elif prev_ma_diff >= 0 and current_ma_diff < 0:
    if abs(current_ma_diff) > signal_threshold:
        generate_sell_signal()
```

##### 1.2 RSI策略 (RSI)
**策略原理**:
- RSI (相对强弱指数) 衡量价格动量
- RSI < 30: 超卖区域，可能反弹
- RSI > 70: 超买区域，可能回调
- 结合趋势对齐和背离检测，提高信号质量

**关键参数**:
```python
{
    "rsi_period": 14,              # RSI计算周期 (默认14)
    "oversold_threshold": 30,      # 超卖阈值 (默认30)
    "overbought_threshold": 70,    # 超买阈值 (默认70)
    "trend_ma_period": 50,         # 趋势判断均线周期 (默认50)
    "enable_trend_alignment": True, # 启用趋势对齐 (默认True)
    "enable_divergence": True,     # 启用背离检测 (默认True)
    "enable_crossover": True        # 启用穿越信号 (默认True)
}
```

**核心特性**:
1. **趋势对齐**: 
   - 上升趋势中，等待RSI回调到40-50再买入
   - 下降趋势中，等待RSI反弹到50-60再卖出
2. **背离检测**:
   - 看涨背离: 价格创新低，RSI未创新低 → 买入信号
   - 看跌背离: 价格创新高，RSI未创新高 → 卖出信号
3. **穿越信号**:
   - RSI从超卖区域向上穿越 → 买入信号
   - RSI从超买区域向下穿越 → 卖出信号

**适用场景**:
- 震荡市场
- 短期交易
- 适合波动适中的股票

##### 1.3 MACD策略 (MACD)
**策略原理**:
- MACD (指数平滑移动平均线) 衡量趋势变化
- MACD金叉: MACD线上穿信号线 → 买入信号
- MACD死叉: MACD线下穿信号线 → 卖出信号

**关键参数**:
```python
{
    "fast_period": 12,    # 快速EMA周期 (默认12)
    "slow_period": 26,    # 慢速EMA周期 (默认26)
    "signal_period": 9    # 信号线周期 (默认9)
}
```

**信号生成逻辑**:
```python
# MACD金叉 (买入)
if prev_hist <= 0 and current_hist > 0:
    generate_buy_signal()

# MACD死叉 (卖出)
elif prev_hist >= 0 and current_hist < 0:
    generate_sell_signal()
```

**适用场景**:
- 趋势跟踪
- 中长期持仓
- 适合趋势明显的股票

#### 2. 高级技术分析策略

##### 2.1 布林带策略 (BollingerBands)
**策略原理**:
- 布林带由中轨(均线)、上轨、下轨组成
- 价格突破下轨: 可能反弹 → 买入信号
- 价格突破上轨: 可能回调 → 卖出信号

**关键参数**:
```python
{
    "period": 20,           # 均线周期 (默认20)
    "std_dev": 2,          # 标准差倍数 (默认2)
    "entry_threshold": 0.02 # 入场阈值 (默认0.02)
}
```

**信号生成逻辑**:
```python
# 买入信号: 价格从下轨下方突破到下轨上方
if prev_percent_b <= 0 and percent_b > 0:
    generate_buy_signal()

# 卖出信号: 价格从上轨上方突破到上轨下方
elif prev_percent_b >= 1 and percent_b < 1:
    generate_sell_signal()
```

**适用场景**:
- 震荡市场
- 均值回归交易
- 适合波动较大的股票

##### 2.2 随机指标策略 (Stochastic)
**策略原理**:
- 随机指标 (KDJ) 衡量价格相对位置
- K < 20 且 K上穿D: 超卖反弹 → 买入信号
- K > 80 且 K下穿D: 超买回调 → 卖出信号

**关键参数**:
```python
{
    "k_period": 14,      # K值周期 (默认14)
    "d_period": 3,       # D值周期 (默认3)
    "oversold": 20,      # 超卖阈值 (默认20)
    "overbought": 80     # 超买阈值 (默认80)
}
```

**适用场景**:
- 短期交易
- 震荡市场
- 适合波动适中的股票

##### 2.3 CCI策略 (Commodity Channel Index)
**策略原理**:
- CCI衡量价格偏离均值的程度
- CCI < -100: 超卖 → 买入信号
- CCI > 100: 超买 → 卖出信号

**关键参数**:
```python
{
    "period": 20,        # 计算周期 (默认20)
    "oversold": -100,    # 超卖阈值 (默认-100)
    "overbought": 100    # 超买阈值 (默认100)
}
```

**适用场景**:
- 短期交易
- 震荡市场
- 适合波动较大的股票

#### 3. 统计套利策略

##### 3.1 配对交易策略 (PairsTrading)
**策略原理**:
- 寻找相关性高的股票对
- 当价差偏离均值时，做多弱势股，做空强势股
- 等待价差回归均值时平仓

**关键参数**:
```python
{
    "correlation_threshold": 0.8,  # 相关性阈值 (默认0.8)
    "lookback_period": 20,         # 回看周期 (默认20)
    "zscore_threshold": 2.0,      # Z-score阈值 (默认2.0)
    "entry_threshold": 2.0,        # 入场阈值 (默认2.0)
    "exit_threshold": 0.5          # 出场阈值 (默认0.5)
}
```

**信号生成逻辑**:
```python
# 计算价差的Z-score
zscore = (spread - spread.mean()) / spread.std()

# 买入信号: Z-score从-2以下回归到-2以上
if prev_zscore <= -zscore_threshold and current_zscore > -zscore_threshold:
    generate_buy_signal()

# 卖出信号: Z-score从2以上回归到2以下
elif prev_zscore >= zscore_threshold and current_zscore < zscore_threshold:
    generate_sell_signal()
```

**适用场景**:
- 需要选择相关性高的股票对
- 适合波动较小的市场
- 适合长期持仓

##### 3.2 均值回归策略 (MeanReversion)
**策略原理**:
- 价格偏离均值时，预期回归
- 使用Z-score衡量偏离程度
- Z-score < -2: 买入信号
- Z-score > 2: 卖出信号

**关键参数**:
```python
{
    "lookback_period": 20,    # 回看周期 (默认20)
    "zscore_threshold": 2.0,  # Z-score阈值 (默认2.0)
    "position_size": 0.1      # 仓位大小 (默认0.1)
}
```

**适用场景**:
- 震荡市场
- 短期交易
- 适合波动较大的股票

##### 3.3 协整策略 (Cointegration)
**策略原理**:
- 寻找具有长期均衡关系的股票对
- 使用协整检验确定配对关系
- 当价差偏离均衡时交易

**关键参数**:
```python
{
    "pvalue_threshold": 0.05,  # 协整检验p值阈值 (默认0.05)
    "lookback_period": 60,    # 回看周期 (默认60)
    "zscore_threshold": 2.0   # Z-score阈值 (默认2.0)
}
```

**适用场景**:
- 需要选择协整关系强的股票对
- 适合长期持仓
- 适合波动较小的市场

#### 4. 因子投资策略

##### 4.1 价值因子策略 (ValueFactor)
**策略原理**:
- 基于估值指标选择股票
- 低估值股票可能被低估，未来可能上涨
- 使用PE、PB、PS等估值指标

**关键参数**:
```python
{
    "factor_name": "pe_ratio",  # 因子名称
    "quantile": 0.2,            # 分位数 (默认0.2，选择最低20%)
    "rebalance_period": 20      # 调仓周期 (默认20天)
}
```

**适用场景**:
- 长期投资
- 价值投资理念
- 适合基本面稳定的股票

##### 4.2 动量因子策略 (MomentumFactor)
**策略原理**:
- 基于价格动量选择股票
- 过去表现好的股票，未来可能继续表现好
- 使用收益率、相对强度等指标

**关键参数**:
```python
{
    "lookback_period": 20,   # 回看周期 (默认20天)
    "quantile": 0.8,         # 分位数 (默认0.8，选择最高20%)
    "rebalance_period": 20   # 调仓周期 (默认20天)
}
```

**适用场景**:
- 趋势明显的市场
- 中期持仓
- 适合波动较大的股票

##### 4.3 低波动因子策略 (LowVolatility)
**策略原理**:
- 选择波动率低的股票
- 低波动股票风险较小，长期收益稳定
- 使用历史波动率指标

**关键参数**:
```python
{
    "lookback_period": 20,   # 回看周期 (默认20天)
    "quantile": 0.2,         # 分位数 (默认0.2，选择最低20%)
    "rebalance_period": 20   # 调仓周期 (默认20天)
}
```

**适用场景**:
- 风险厌恶型投资者
- 长期投资
- 适合波动较小的股票

##### 4.4 多因子组合策略 (MultiFactor)
**策略原理**:
- 结合多个因子进行选股
- 综合价值、动量、质量等多个维度
- 使用因子加权或等权重组合

**关键参数**:
```python
{
    "factors": [
        {"name": "value", "weight": 0.3},
        {"name": "momentum", "weight": 0.3},
        {"name": "quality", "weight": 0.4}
    ],
    "rebalance_period": 20  # 调仓周期 (默认20天)
}
```

**适用场景**:
- 追求稳健收益
- 长期投资
- 适合多股票组合

### 组合策略

#### 策略组合原理
- 将多个策略组合使用
- 通过权重分配控制各策略影响
- 使用信号整合方法合并信号

#### 信号整合方法

**1. 加权投票 (weighted_voting)**
```python
# 根据策略权重加权求和
final_signal = sum(strategy_signal * weight for strategy, weight in strategies.items())
```

**2. 多数投票 (majority_voting)**
```python
# 多数策略同意的信号
final_signal = majority(signals)
```

**3. 加权平均 (weighted_average)**
```python
# 加权平均信号强度
final_signal = weighted_average(signals, weights)
```

#### 组合策略配置示例
```python
{
    "strategy_name": "portfolio",
    "strategy_config": {
        "strategies": [
            {
                "name": "rsi",
                "weight": 0.4,
                "config": {
                    "rsi_period": 14,
                    "oversold_threshold": 30,
                    "overbought_threshold": 70
                }
            },
            {
                "name": "macd",
                "weight": 0.3,
                "config": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9
                }
            },
            {
                "name": "bollinger",
                "weight": 0.3,
                "config": {
                    "period": 20,
                    "std_dev": 2
                }
            }
        ],
        "integration_method": "weighted_voting"
    }
}
```

---

## 使用方法

### 1. 创建回测任务

#### 步骤1: 访问任务创建页面
- 导航到: `/tasks/create`
- 选择任务类型: **回测**

#### 步骤2: 配置基本参数
```typescript
{
    task_name: "我的回测任务",      // 任务名称
    description: "测试移动平均策略", // 任务描述
    start_date: "2024-01-01",      // 回测开始日期
    end_date: "2024-12-31",        // 回测结束日期
    initial_cash: 100000,           // 初始资金 (默认100,000元)
    commission_rate: 0.0003,       // 手续费率 (默认0.03%)
    slippage_rate: 0.0001          // 滑点率 (默认0.01%)
}
```

#### 步骤3: 选择股票
- 点击"选择股票"按钮
- 从股票列表中选择要回测的股票
- 支持多选 (建议1-10只股票)

#### 步骤4: 选择策略

**单策略模式**:
```typescript
{
    strategy_type: "single",
    strategy_name: "moving_average",  // 策略名称
    strategy_config: {
        short_window: 5,              // 短期均线周期
        long_window: 20,              // 长期均线周期
        signal_threshold: 0.02        // 信号阈值
    }
}
```

**组合策略模式**:
```typescript
{
    strategy_type: "portfolio",
    strategy_config: {
        strategies: [
            {
                name: "rsi",
                weight: 0.4,
                config: { rsi_period: 14 }
            },
            {
                name: "macd",
                weight: 0.3,
                config: { fast_period: 12 }
            }
        ],
        integration_method: "weighted_voting"
    }
}
```

#### 步骤5: 提交任务
- 点击"创建任务"按钮
- 系统自动创建任务并开始执行
- 跳转到任务详情页面查看进度

### 2. 监控回测进度

#### 实时进度监控
- **WebSocket连接**: 自动建立WebSocket连接
- **进度更新**: 实时显示回测进度 (0-100%)
- **阶段信息**: 显示当前执行阶段

**进度阶段**:
```
初始化 (0-10%)      → 生成回测ID，启动监控
数据加载 (10-25%)   → 加载股票历史数据
策略设置 (25-30%)   → 创建策略实例
回测执行 (30-90%)   → 逐日回测循环
结果计算 (90-95%)   → 计算收益率和风险指标
完成 (95-100%)      → 保存结果，任务完成
```

#### 进度详情
- **已处理交易日**: 显示已处理的交易日数/总交易日数
- **已生成信号**: 显示已生成的交易信号数
- **已执行交易**: 显示已执行的交易数
- **当前持仓**: 显示当前持仓股票和数量

### 3. 查看回测结果

#### 结果概览
- **总收益率**: 回测期间的总收益率
- **年化收益率**: 年化后的收益率
- **夏普比率**: 风险调整后的收益指标
- **最大回撤**: 最大亏损幅度
- **交易次数**: 总交易次数
- **胜率**: 盈利交易占比

#### 详细分析

**1. 收益曲线**
- 显示回测期间的资产净值曲线
- 对比基准指数 (可选)
- 标注重要交易点

**2. 持仓分析**
- 显示持仓股票分布
- 显示持仓时间分布
- 显示持仓收益贡献

**3. 交易历史**
- 显示所有交易记录
- 包括买入/卖出时间、价格、数量
- 显示每笔交易的盈亏

**4. 风险分析**
- 波动率分析
- 最大回撤分析
- VaR (风险价值) 分析

**5. 信号历史**
- 显示所有生成的交易信号
- 包括信号类型、强度、原因
- 显示信号执行情况

### 4. 策略参数优化

#### 使用优化功能
- 导航到: `/optimization`
- 选择要优化的策略
- 设置参数范围
- 选择优化目标 (如: 最大化夏普比率)

#### 优化参数示例
```typescript
{
    strategy_name: "moving_average",
    parameters: {
        short_window: {
            min: 5,
            max: 20,
            step: 1
        },
        long_window: {
            min: 20,
            max: 60,
            step: 5
        }
    },
    optimization_target: "sharpe_ratio"  // 优化目标
}
```

### 5. 保存和加载策略配置

#### 保存策略配置
- 在策略配置页面点击"保存配置"
- 输入配置名称
- 配置将保存到数据库

#### 加载策略配置
- 在策略配置页面点击"加载配置"
- 从已保存的配置列表中选择
- 自动填充策略参数

---

## 实战案例

### 案例1: 移动平均策略回测

#### 场景
- **股票**: 000001 (平安银行)
- **时间范围**: 2024-01-01 至 2024-12-31
- **初始资金**: 100,000元
- **策略**: 移动平均策略 (短期5日，长期20日)

#### 配置
```json
{
    "strategy_name": "moving_average",
    "strategy_config": {
        "short_window": 5,
        "long_window": 20,
        "signal_threshold": 0.02
    },
    "stock_codes": ["000001"],
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_cash": 100000
}
```

#### 预期结果
- 在趋势明显的时期产生交易信号
- 金叉时买入，死叉时卖出
- 适合中长期持仓

### 案例2: RSI策略回测

#### 场景
- **股票**: 000002 (万科A)
- **时间范围**: 2024-01-01 至 2024-12-31
- **初始资金**: 100,000元
- **策略**: RSI策略 (周期14，超卖30，超买70)

#### 配置
```json
{
    "strategy_name": "rsi",
    "strategy_config": {
        "rsi_period": 14,
        "oversold_threshold": 30,
        "overbought_threshold": 70,
        "enable_trend_alignment": true,
        "enable_divergence": true
    },
    "stock_codes": ["000002"],
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_cash": 100000
}
```

#### 预期结果
- 在震荡市场产生较多交易信号
- 结合趋势对齐，提高信号质量
- 适合短期交易

### 案例3: 组合策略回测

#### 场景
- **股票**: 000001, 000002, 000858
- **时间范围**: 2024-01-01 至 2024-12-31
- **初始资金**: 100,000元
- **策略**: RSI + MACD + 布林带组合

#### 配置
```json
{
    "strategy_name": "portfolio",
    "strategy_config": {
        "strategies": [
            {
                "name": "rsi",
                "weight": 0.4,
                "config": {
                    "rsi_period": 14,
                    "oversold_threshold": 30,
                    "overbought_threshold": 70
                }
            },
            {
                "name": "macd",
                "weight": 0.3,
                "config": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9
                }
            },
            {
                "name": "bollinger",
                "weight": 0.3,
                "config": {
                    "period": 20,
                    "std_dev": 2
                }
            }
        ],
        "integration_method": "weighted_voting"
    },
    "stock_codes": ["000001", "000002", "000858"],
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_cash": 100000
}
```

#### 预期结果
- 综合多个策略的信号
- 降低单一策略的风险
- 提高策略稳定性

---

## 最佳实践

### 1. 策略选择建议

**趋势市场**:
- 推荐: 移动平均策略、MACD策略
- 特点: 适合中长期持仓

**震荡市场**:
- 推荐: RSI策略、布林带策略、均值回归策略
- 特点: 适合短期交易

**多股票组合**:
- 推荐: 因子投资策略、组合策略
- 特点: 分散风险，提高稳定性

### 2. 参数调优建议

**移动平均策略**:
- 短期均线: 5-10日 (适合短期交易)
- 长期均线: 20-60日 (适合中长期持仓)
- 根据股票波动率调整阈值

**RSI策略**:
- RSI周期: 14日 (标准设置)
- 超卖/超买阈值: 根据市场调整 (30/70 或 20/80)
- 启用趋势对齐和背离检测

**MACD策略**:
- 快速/慢速周期: 12/26 (标准设置)
- 信号线周期: 9 (标准设置)
- 适合趋势明显的股票

### 3. 风险控制建议

**仓位管理**:
- 单股持仓不超过30%
- 总持仓不超过100%
- 设置止损和止盈

**交易成本**:
- 考虑手续费和滑点
- 避免频繁交易
- 选择流动性好的股票

**回测验证**:
- 使用足够长的历史数据 (至少1年)
- 在不同市场环境下测试
- 对比多个策略的表现

### 4. 常见问题

**Q: 为什么回测结果和实盘差异大?**
A: 
- 回测使用历史数据，实盘面临未来不确定性
- 回测假设可以按收盘价成交，实盘可能有滑点
- 回测没有考虑市场冲击和流动性问题

**Q: 如何选择合适的策略?**
A:
- 根据市场环境选择 (趋势 vs 震荡)
- 根据投资周期选择 (短期 vs 长期)
- 根据风险偏好选择 (激进 vs 保守)

**Q: 组合策略如何分配权重?**
A:
- 根据策略历史表现分配
- 根据策略相关性分配 (低相关性策略可以增加权重)
- 使用优化工具自动寻找最优权重

---

## 总结

### 平台核心价值
1. **多策略支持**: 涵盖技术分析、统计套利、因子投资
2. **实时监控**: WebSocket实时推送进度和结果
3. **灵活配置**: 支持单策略和组合策略
4. **完整分析**: 提供详细的收益和风险分析

### 使用建议
1. **从简单开始**: 先使用基础策略 (如移动平均)
2. **逐步优化**: 根据回测结果调整参数
3. **组合使用**: 尝试组合多个策略降低风险
4. **持续学习**: 理解策略原理，不断改进

### 下一步
- 尝试不同的策略组合
- 优化策略参数
- 分析回测结果，改进策略
- 准备实盘交易 (谨慎!)

---

**祝您回测顺利，交易成功！** 🚀
