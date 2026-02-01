# 多策略组合回测系统开发计划

## 📋 项目概述

**目标**：实现多策略组合回测系统，通过信号融合提升策略稳健性和收益潜力。

**核心价值**：
- 解决单策略效果不佳的问题
- 通过策略组合降低单一策略风险
- 提升信号质量和交易决策的可靠性

**技术原则**：
- 保持与现有架构完全兼容
- 不引入第三方回测框架
- 参考业界最佳实践（Backtrader、QuantConnect等）
- 渐进式开发，快速验证假设

---

## 🎯 阶段规划

### 阶段0：准备与设计（1-2天）

#### 任务清单
- [ ] 技术方案评审
- [ ] API设计评审
- [ ] 数据模型设计
- [ ] 开发环境准备

#### 交付物
- 技术设计文档
- API接口设计
- 数据模型定义

---

### 阶段1：核心功能实现（P0 - 1-2周）

**目标**：实现基础的多策略组合和信号融合功能，快速验证假设。

#### 1.1 信号整合器（SignalIntegrator）- 3-4天

**文件**：`backend/app/services/backtest/utils/signal_integrator.py`

**功能**：
- 加权投票算法
- 一致性增强机制
- 信号冲突解决
- 信号强度归一化

**技术方案**：
```python
class SignalIntegrator:
    """信号整合器 - 参考QuantConnect的信号融合算法"""
    
    def __init__(self, method: str = "weighted_voting"):
        self.method = method
    
    def integrate(self, signals: List[TradingSignal], 
                  weights: Dict[str, float]) -> List[TradingSignal]:
        """
        整合多个策略的信号
        
        算法：
        1. 按股票分组信号
        2. 计算加权投票得分
        3. 应用一致性增强
        4. 解决冲突信号
        5. 生成最终信号
        """
        pass
```

**验收标准**：
- [ ] 支持加权投票算法
- [ ] 支持一致性增强（2+策略同向时增强信号）
- [ ] 正确处理信号冲突（买入vs卖出）
- [ ] 单元测试覆盖率 > 80%

**风险评估**：
- 低风险：算法相对简单，主要是数学计算
- 注意点：信号强度归一化需要仔细设计

---

#### 1.2 策略组合类（StrategyPortfolio）- 3-4天

**文件**：`backend/app/services/backtest/core/strategy_portfolio.py`

**功能**：
- 管理多个策略实例
- 继承BaseStrategy，保持兼容
- 收集各策略信号并整合
- 策略权重管理

**技术方案**：
```python
class StrategyPortfolio(BaseStrategy):
    """策略组合类 - 参考Backtrader的Cerebro设计思想"""
    
    def __init__(self, strategies: List[BaseStrategy], 
                 weights: Dict[str, float] = None,
                 integration_method: str = "weighted_voting"):
        super().__init__("StrategyPortfolio", {})
        self.strategies = strategies
        self.weights = weights or self._default_weights()
        self.integrator = SignalIntegrator(integration_method)
    
    def generate_signals(self, data, current_date):
        """生成组合信号"""
        # 1. 收集所有策略的信号
        # 2. 使用SignalIntegrator整合
        # 3. 返回最终信号
        pass
```

**验收标准**：
- [ ] 完全兼容BaseStrategy接口
- [ ] 支持动态添加/移除策略
- [ ] 支持自定义权重配置
- [ ] 信号包含来源策略信息（metadata）
- [ ] 单元测试覆盖率 > 80%

**风险评估**：
- 中风险：需要确保与现有BacktestExecutor兼容
- 注意点：策略实例的生命周期管理

---

#### 1.3 扩展策略工厂 - 2-3天

**文件**：`backend/app/services/backtest/strategies/strategy_factory.py`

**功能**：
- 支持创建策略组合
- 保持向后兼容
- 支持组合策略配置解析

**技术方案**：
```python
class StrategyFactory:
    @classmethod
    def create_strategy(cls, strategy_name: str, config: Dict[str, Any]) -> BaseStrategy:
        """创建策略实例（支持单策略和组合策略）"""
        # 检测是否为组合策略
        if strategy_name == "portfolio" or "strategies" in config:
            return cls._create_portfolio_strategy(config)
        else:
            # 原有逻辑
            return cls._create_single_strategy(strategy_name, config)
    
    @classmethod
    def _create_portfolio_strategy(cls, config: Dict[str, Any]) -> StrategyPortfolio:
        """创建策略组合"""
        strategies = []
        weights = {}
        
        strategy_configs = config.get("strategies", [])
        for strat_config in strategy_configs:
            name = strat_config["name"]
            weight = strat_config.get("weight", 1.0)
            strat_config_dict = strat_config.get("config", {})
            
            strategy = cls.create_strategy(name, strat_config_dict)
            strategies.append(strategy)
            weights[strategy.name] = weight
        
        # 归一化权重
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        return StrategyPortfolio(strategies, weights)
```

**验收标准**：
- [ ] 支持组合策略配置格式
- [ ] 保持单策略创建功能不变
- [ ] 支持权重自动归一化
- [ ] 配置验证和错误处理
- [ ] 单元测试覆盖率 > 80%

**风险评估**：
- 低风险：主要是扩展现有工厂，不破坏原有功能

---

#### 1.4 扩展回测API - 2-3天

**文件**：`backend/app/api/v1/backtest.py`

**功能**：
- 支持组合策略配置
- 更新策略列表API
- 保持向后兼容

**API设计**：
```python
# 原有API保持不变
@router.post("", response_model=StandardResponse)
async def run_backtest(request: BacktestRequest):
    """运行回测（支持单策略和组合策略）"""
    # 自动检测是否为组合策略
    if isinstance(request.strategy_config.get("strategies"), list):
        # 组合策略逻辑
        pass
    else:
        # 原有单策略逻辑
        pass

# 新增：获取策略组合模板
@router.get("/portfolio-templates", response_model=StandardResponse)
async def get_portfolio_templates():
    """获取预设的策略组合模板"""
    pass
```

**配置格式示例**：
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
          "oversold_threshold": 30
        }
      },
      {
        "name": "macd",
        "weight": 0.3,
        "config": {
          "fast_period": 12,
          "slow_period": 26
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

**验收标准**：
- [ ] API支持组合策略配置
- [ ] 保持原有API完全兼容
- [ ] 配置验证和错误提示
- [ ] API文档更新
- [ ] 集成测试通过

**风险评估**：
- 低风险：主要是扩展，不破坏现有功能

---

#### 1.5 单元测试和集成测试 - 2-3天

**测试文件**：
- `backend/tests/test_signal_integrator.py`
- `backend/tests/test_strategy_portfolio.py`
- `backend/tests/test_strategy_factory_portfolio.py`
- `backend/tests/integration/test_backtest_portfolio.py`

**测试覆盖**：
- [ ] SignalIntegrator所有算法
- [ ] StrategyPortfolio信号生成
- [ ] 策略工厂组合策略创建
- [ ] API端到端测试
- [ ] 边界情况和错误处理

**验收标准**：
- [ ] 单元测试覆盖率 > 80%
- [ ] 集成测试覆盖主要流程
- [ ] 所有测试通过

---

#### 1.6 前端组合策略配置组件 - 4-5天

**文件**：
- `frontend/src/components/backtest/PortfolioStrategyConfig.tsx`（新建）
- `frontend/src/app/tasks/create/page.tsx`（修改）
- `frontend/src/services/dataService.ts`（修改）

**功能**：
- 策略组合选择器（单选：单策略/组合策略）
- 组合策略配置界面
  - 添加/删除策略
  - 配置每个策略的权重
  - 配置每个策略的参数
  - 权重归一化显示
  - 权重约束验证（max_weight, gross_leverage）
- 策略组合模板选择
- 组合策略预览

**UI设计**：
```tsx
// 策略组合配置组件
<PortfolioStrategyConfig
  strategies={availableStrategies}
  portfolioConfig={portfolioConfig}
  onChange={(config) => setPortfolioConfig(config)}
  constraints={{
    maxWeight: 0.5,
    grossLeverage: 1.0,
    minStrategies: 1,
    maxStrategies: 10
  }}
/>
```

**技术方案**：
1. **策略选择切换**：
   - 在回测配置中添加"策略类型"选择（单策略/组合策略）
   - 根据选择显示不同的配置界面

2. **组合策略配置界面**：
   - 策略列表：显示已添加的策略
   - 策略选择器：从可用策略列表添加
   - 权重配置：滑块或输入框，实时显示归一化后的权重
   - 参数配置：每个策略的参数配置（复用现有StrategyConfigForm）
   - 权重约束提示：显示当前权重是否符合约束

3. **数据模型**：
```typescript
interface PortfolioStrategyConfig {
  strategy_type: 'single' | 'portfolio';
  // 单策略配置（保持兼容）
  strategy_name?: string;
  strategy_config?: Record<string, any>;
  // 组合策略配置
  portfolio_config?: {
    strategies: Array<{
      name: string;
      weight: number;
      config: Record<string, any>;
    }>;
    integration_method?: 'weighted_voting' | 'consensus' | 'ml_fusion';
    constraints?: {
      max_weight?: number;
      gross_leverage?: number;
    };
  };
}
```

4. **API调用修改**：
```typescript
// dataService.ts
static async runBacktest(request: BacktestRequest): Promise<BacktestResult> {
  // 自动检测是否为组合策略
  if (request.strategy_config?.strategies) {
    // 组合策略格式
    request.strategy_name = 'portfolio';
  }
  return apiRequest.post<BacktestResult>('/backtest', request);
}
```

**验收标准**：
- [ ] 能够切换单策略/组合策略模式
- [ ] 能够添加/删除策略
- [ ] 权重配置正确，自动归一化
- [ ] 权重约束验证正确
- [ ] 策略参数配置正确
- [ ] 配置数据格式符合后端API要求
- [ ] UI响应流畅，用户体验良好

**风险评估**：
- 中风险：需要修改现有创建任务页面，可能影响现有功能
- 应对措施：充分测试单策略模式，确保向后兼容

---

#### 1.7 前端组合策略结果展示 - 3-4天

**文件**：
- `frontend/src/components/backtest/PortfolioStrategyResults.tsx`（新建）
- `frontend/src/app/tasks/[id]/page.tsx`（修改）
- `frontend/src/components/charts/BacktestChart.tsx`（修改）

**功能**：
- 组合策略回测结果展示
- 策略贡献度可视化
- 策略权重分布图表
- 策略信号来源标识
- 与单策略结果对比

**UI设计**：
1. **策略贡献度图表**：
   - 饼图：各策略收益贡献占比
   - 柱状图：各策略的夏普比率、胜率等指标对比
   - 时间序列图：各策略在不同时期的贡献

2. **策略权重可视化**：
   - 权重分布图
   - 权重变化时间序列（如果支持动态权重）

3. **信号来源标识**：
   - 在交易记录中显示信号来源策略
   - 在信号列表中显示策略来源

**技术方案**：
```typescript
// 组合策略结果数据模型
interface PortfolioBacktestResult extends BacktestResult {
  portfolio_info?: {
    strategies: Array<{
      name: string;
      weight: number;
      contribution: {
        return_contribution: number;
        sharpe_contribution: number;
        trade_count: number;
      };
    }>;
    correlation_matrix?: number[][];
  };
}
```

**验收标准**：
- [ ] 组合策略结果正确展示
- [ ] 策略贡献度图表正确显示
- [ ] 策略权重可视化正确
- [ ] 信号来源标识清晰
- [ ] 与单策略结果展示兼容

**风险评估**：
- 低风险：主要是新增展示功能，不影响现有功能

---

### 阶段2：功能增强和优化（P1 - 2-3周）

#### 2.1 策略贡献度分析 - 3-4天

**文件**：`backend/app/services/backtest/analysis/portfolio_contribution_analyzer.py`

**功能**：
- 分析每个策略对整体收益的贡献
- 计算策略相关性
- 识别冗余策略
- 生成贡献度报告

**技术方案**：
```python
class PortfolioContributionAnalyzer:
    """策略组合贡献度分析器"""
    
    def analyze_contribution(self, portfolio_history, strategy_signals):
        """
        分析各策略贡献度
        
        方法：
        1. 追踪每个信号的来源策略
        2. 计算策略信号的收益贡献
        3. 计算策略间相关性
        4. 识别冗余策略
        """
        pass
```

**验收标准**：
- [ ] 准确计算策略收益贡献
- [ ] 计算策略相关性矩阵
- [ ] 识别冗余策略
- [ ] 生成可视化报告

---

#### 2.2 动态权重调整 - 4-5天

**文件**：`backend/app/services/backtest/core/dynamic_weight_manager.py`

**功能**：
- 根据策略近期表现动态调整权重
- 支持多种权重调整策略
- 防止权重剧烈波动

**技术方案**：
```python
class DynamicWeightManager:
    """动态权重管理器"""
    
    def __init__(self, adjustment_method: str = "performance_based"):
        self.method = adjustment_method
    
    def adjust_weights(self, current_weights: Dict[str, float],
                      strategy_performance: Dict[str, float],
                      lookback_period: int = 30) -> Dict[str, float]:
        """
        根据策略表现调整权重
        
        方法：
        1. 计算策略近期夏普比率/收益
        2. 根据表现调整权重
        3. 应用平滑机制防止剧烈波动
        4. 归一化权重
        """
        pass
```

**验收标准**：
- [ ] 支持基于表现的权重调整
- [ ] 支持平滑机制
- [ ] 权重调整可配置
- [ ] 回测验证权重调整效果

---

#### 2.3 市场状态感知 - 5-6天

**文件**：`backend/app/services/backtest/core/market_regime_detector.py`

**功能**：
- 检测市场状态（趋势/震荡/波动）
- 根据市场状态选择策略组合
- 动态调整策略权重

**技术方案**：
```python
class MarketRegimeDetector:
    """市场状态检测器"""
    
    def detect_regime(self, data: pd.DataFrame) -> str:
        """
        检测市场状态
        
        方法：
        1. 计算趋势指标（ADX、均线斜率）
        2. 计算波动率指标（ATR、波动率）
        3. 计算震荡指标（RSI、布林带位置）
        4. 综合判断市场状态
        """
        pass
    
    def get_recommended_strategies(self, regime: str) -> List[str]:
        """根据市场状态推荐策略"""
        pass
```

**验收标准**：
- [ ] 准确识别市场状态
- [ ] 策略推荐合理
- [ ] 回测验证效果提升

---

#### 2.4 性能优化 - 3-4天

**优化点**：
- 策略信号生成并行化
- 指标计算共享（避免重复计算）
- 信号整合向量化

**验收标准**：
- [ ] 多策略回测性能提升 > 30%
- [ ] 内存使用优化
- [ ] 性能测试报告

---

### 阶段3：高级功能和优化（P2 - 持续优化）

#### 3.1 机器学习融合（可选）
- 使用XGBoost/LightGBM融合多策略信号
- 训练信号融合模型
- 在线学习更新

#### 3.2 风险平价权重（可选）
- 基于风险贡献分配权重
- 实现风险平价模型

#### 3.3 策略组合优化器（可选）
- 自动寻找最优策略组合
- 使用Optuna优化组合权重

---

## 📊 时间估算

| 阶段 | 任务 | 时间估算 | 累计时间 |
|------|------|---------|---------|
| **阶段0** | 准备与设计 | 1-2天 | 1-2天 |
| **阶段1** | 核心功能实现 | 17-22天 | 18-24天 |
| **阶段2** | 功能增强 | 15-19天 | 33-43天 |
| **阶段3** | 高级功能 | 持续优化 | - |

**阶段1详细时间**：
- 1.1 信号整合器：3-4天
- 1.2 策略组合类：3-4天
- 1.3 扩展策略工厂：2-3天
- 1.4 扩展回测API：2-3天
- 1.5 单元测试和集成测试：2-3天
- 1.6 前端组合策略配置组件：4-5天
- 1.7 前端组合策略结果展示：3-4天

**总计（阶段0+1+2）**：约 **6-8周**（按每周5个工作日计算）

---

## 🎯 里程碑

### 里程碑1：MVP完成（阶段1结束）
- ✅ 基础组合策略功能可用
- ✅ 信号融合算法实现
- ✅ API支持组合策略
- ✅ 前端配置界面完成
- ✅ 前端结果展示完成
- ✅ 基础测试通过（包括工程验收测试）

**验证标准**：
- 能够通过前端创建并运行组合策略回测
- 组合策略效果优于单策略（至少一个测试用例）
- 所有工程验收标准通过（权重约束、一致性等）

---

### 里程碑2：功能完整（阶段2结束）
- ✅ 策略贡献度分析可用
- ✅ 动态权重调整可用
- ✅ 市场状态感知可用
- ✅ 性能优化完成

**验证标准**：
- 组合策略在多个测试用例中表现优于单策略
- 贡献度分析报告准确
- 性能满足要求

---

## ⚠️ 风险评估与应对

### 技术风险

| 风险 | 影响 | 概率 | 应对措施 |
|------|------|------|---------|
| 信号融合算法效果不佳 | 高 | 中 | 先实现简单算法验证，再优化 |
| 与现有架构不兼容 | 高 | 低 | 严格遵循BaseStrategy接口，充分测试 |
| 性能问题 | 中 | 中 | 阶段2包含性能优化，必要时向量化 |
| 权重调整不稳定 | 中 | 中 | 实现平滑机制，设置权重边界 |

### 业务风险

| 风险 | 影响 | 概率 | 应对措施 |
|------|------|---------|---------|
| 组合策略效果不如预期 | 高 | 中 | 阶段1快速验证，及时调整方向 |
| 用户学习成本高 | 中 | 低 | 提供预设模板，完善文档 |

---

## 📝 文档要求

### 开发文档
- [ ] 技术设计文档
- [ ] API接口文档
- [ ] 代码注释（关键算法）

### 用户文档
- [ ] 组合策略使用指南
- [ ] 配置示例
- [ ] 常见问题解答

### 测试文档
- [ ] 测试用例说明
- [ ] 测试覆盖率报告

---

## ✅ 验收标准

### 功能验收
- [ ] 能够创建并运行组合策略回测
- [ ] 信号融合算法正确工作
- [ ] 策略贡献度分析准确
- [ ] 动态权重调整有效
- [ ] API文档完整

### 工程验收（确定性测试）

#### 1. 组合权重约束验证
- [ ] **权重归一化**：所有策略权重之和必须等于1.0（误差 < 0.001）
- [ ] **最大权重限制**：单个策略权重不超过 `max_weight`（默认0.5，可配置）
- [ ] **总杠杆限制**：`gross_leverage` 不超过配置值（默认1.0，可配置）
- [ ] **权重非负**：所有权重 >= 0

**测试用例**：
```python
# 测试权重归一化
weights = {'rsi': 0.4, 'macd': 0.3, 'bollinger': 0.3}
assert abs(sum(weights.values()) - 1.0) < 0.001

# 测试最大权重限制
assert all(w <= max_weight for w in weights.values())

# 测试总杠杆
assert sum(abs(w) for w in weights.values()) <= gross_leverage
```

#### 2. 策略权重 meta-weight 生效验证
- [ ] **策略级权重**：每个策略的 `weight` 配置正确应用到信号融合
- [ ] **权重传递**：策略权重正确传递到 `SignalIntegrator`
- [ ] **权重持久化**：权重配置在回测过程中保持不变

**测试用例**：
```python
# 测试策略权重应用
portfolio = StrategyPortfolio(
    strategies=[rsi_strategy, macd_strategy],
    weights={'rsi': 0.7, 'macd': 0.3}
)
signals = portfolio.generate_signals(data, date)
# 验证RSI信号权重为0.7，MACD信号权重为0.3
```

#### 3. 调仓频率正确性验证
- [ ] **调仓频率配置**：`rebalance_frequency` 配置正确生效
- [ ] **调仓时机**：按照配置的频率（daily/weekly/monthly）执行调仓
- [ ] **信号生成频率**：信号生成频率与调仓频率一致

**测试用例**：
```python
# 测试daily调仓
config = BacktestConfig(rebalance_frequency='daily')
# 验证每个交易日都生成信号

# 测试weekly调仓
config = BacktestConfig(rebalance_frequency='weekly')
# 验证每周第一个交易日生成信号
```

#### 4. 缺失处理正确性验证
- [ ] **策略信号缺失**：当某个策略无法生成信号时，不影响其他策略
- [ ] **数据缺失**：当某个策略所需数据缺失时，优雅降级
- [ ] **权重重新归一化**：当部分策略失效时，剩余策略权重自动归一化

**测试用例**：
```python
# 测试策略信号缺失
strategies = [rsi_strategy, macd_strategy]
# 模拟macd_strategy无法生成信号
# 验证rsi_strategy信号仍然正常，权重自动调整
```

#### 5. 与单策略结果一致性验证
- [ ] **单策略等价性**：当组合只包含1个策略且权重=1时，结果与单策略完全一致
- [ ] **数值精度**：收益、夏普比率等指标误差 < 0.0001
- [ ] **交易记录一致性**：交易记录（时间、价格、数量）完全一致

**测试用例**：
```python
# 测试单策略等价性
single_strategy_result = run_backtest('rsi', ...)
portfolio_result = run_backtest('portfolio', {
    'strategies': [{'name': 'rsi', 'weight': 1.0}]
})

# 验证结果一致性
assert abs(single_strategy_result.total_return - 
           portfolio_result.total_return) < 0.0001
assert abs(single_strategy_result.sharpe_ratio - 
           portfolio_result.sharpe_ratio) < 0.0001
# 验证交易记录一致
assert len(single_strategy_result.trades) == len(portfolio_result.trades)
```

### 质量验收
- [ ] 单元测试覆盖率 > 80%
- [ ] 集成测试通过
- [ ] 代码审查通过
- [ ] 性能满足要求（多策略回测时间 < 单策略回测时间 * 策略数量 * 1.5）

### 兼容性验收
- [ ] 原有单策略功能完全不受影响
- [ ] 原有API完全兼容
- [ ] 数据库结构兼容（如涉及）

---

## 🚀 快速启动建议

### 第一步：快速验证（1-2天）
1. 实现最简单的加权投票算法
2. 手动创建2-3个策略的组合
3. 运行回测验证效果

### 第二步：完善框架（阶段1）
如果验证通过，继续完善框架和API

### 第三步：持续优化（阶段2+）
根据实际使用情况持续优化

---

## 📅 建议时间表

**第1周**：
- 阶段0：准备与设计（1-2天）
- 阶段1.1-1.2：信号整合器和策略组合类（6-8天）

**第2周**：
- 阶段1.3-1.4：扩展工厂和API（4-6天）
- 阶段1.5：后端测试（2-3天）

**第3周**：
- 阶段1.6：前端组合策略配置组件（4-5天）

**第4周**：
- 阶段1.7：前端组合策略结果展示（3-4天）
- 阶段1：前后端联调测试（2-3天）

**第5-6周**：
- 阶段2：功能增强和优化

**第7周及以后**：
- 阶段3：高级功能（按需）

---

## 📌 注意事项

1. **保持兼容性**：所有改动必须保持向后兼容
2. **渐进式开发**：先实现核心功能，快速验证假设
3. **充分测试**：每个阶段都要有充分的测试
4. **文档同步**：代码和文档同步更新
5. **性能监控**：关注性能指标，及时优化

---

## 🔄 迭代计划

如果阶段1验证效果不佳：
1. 分析原因（算法问题？策略选择问题？）
2. 调整方案
3. 重新验证

如果阶段1验证效果良好：
1. 继续阶段2
2. 根据实际使用情况优化
3. 考虑阶段3的高级功能

---

**文档版本**：v1.0  
**创建日期**：2025-01-XX  
**最后更新**：2025-01-XX
