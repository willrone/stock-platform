# 回测引擎模块

该模块包含所有与策略回测和执行相关的服务，提供完整的量化交易策略验证解决方案。

## 主要组件

### 回测引擎核心
- **BacktestEngine**: 回测引擎核心（注意：实际在 backtest_engine.py 中定义）
- **BacktestConfig**: 回测配置
- **TradingSignal**: 交易信号
- **Trade**: 交易记录
- **Position**: 持仓信息

### 交易策略
- **BaseStrategy**: 策略基类，所有策略的基础接口
- **MovingAverageStrategy**: 移动平均策略
- **RSIStrategy**: RSI 策略
- **MACDStrategy**: MACD 策略
- **StrategyFactory**: 策略工厂，用于创建不同类型的策略

### 组合管理
- **PortfolioManager**: 组合管理器，处理资金管理和风险控制

### 回测执行
- **BacktestExecutor**: 回测执行器，负责执行回测流程
- **DataLoader**: 数据加载器，加载回测所需的历史数据

### 枚举类型
- **SignalType**: 信号类型（买入/卖出）
- **OrderType**: 订单类型（市价单/限价单）

## 使用示例

```python
# 导入回测服务
from app.services.backtest import BacktestExecutor, MovingAverageStrategy, BacktestConfig

# 创建回测配置
config = BacktestConfig(
    initial_cash=100000.0,
    start_date="2023-01-01",
    end_date="2023-12-31",
    commission=0.001
)

# 创建策略
strategy = MovingAverageStrategy(short_window=5, long_window=20)

# 创建回测执行器
executor = BacktestExecutor()

# 执行回测
result = await executor.run_backtest(
    strategy=strategy,
    stock_codes=["000001.SZ", "000002.SZ"],
    config=config
)

# 分析结果
print(f"总收益率: {result.total_return:.2%}")
print(f"夏普比率: {result.sharpe_ratio:.2f}")
print(f"最大回撤: {result.max_drawdown:.2%}")
```

## 支持的策略类型

### 技术指标策略
- **移动平均策略**: 基于短期和长期移动平均线的交叉
- **RSI 策略**: 基于相对强弱指数的超买超卖策略
- **MACD 策略**: 基于 MACD 指标的趋势跟踪策略

### 自定义策略
可以通过继承 `BaseStrategy` 类来实现自定义策略：

```python
class CustomStrategy(BaseStrategy):
    def generate_signals(self, data):
        # 实现自定义信号生成逻辑
        pass
    
    def calculate_position_size(self, signal, portfolio_value):
        # 实现自定义仓位管理逻辑
        pass
```

## 回测指标

回测结果包含以下关键指标：

- **收益指标**: 总收益率、年化收益率、累计收益
- **风险指标**: 波动率、最大回撤、VaR
- **风险调整收益**: 夏普比率、索提诺比率、卡尔马比率
- **交易统计**: 交易次数、胜率、平均盈亏

## 配置选项

回测引擎支持以下配置：

- **资金管理**: 初始资金、仓位限制、杠杆设置
- **交易成本**: 手续费、滑点、印花税
- **风险控制**: 止损止盈、最大持仓数量
- **数据设置**: 数据频率、复权方式

## 依赖关系

该模块依赖于：
- 数据模块（历史数据）
- 预测模块（技术指标）
- 基础设施模块（缓存、日志）

## 注意事项

1. 回测结果不代表未来表现
2. 需要考虑交易成本和滑点的影响
3. 建议使用样本外数据进行策略验证
4. 注意避免过度拟合和前瞻偏差
5. 策略参数应通过交叉验证进行优化