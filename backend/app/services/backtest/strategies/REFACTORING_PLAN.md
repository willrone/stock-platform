# Strategies.py 重构计划

## 目标
将 strategies.py (1507行) 拆分为多个策略文件，每个策略一个文件，提高可维护性。

## 约束条件（重要！）
1. **所有方法必须完整实现** - 不允许使用 NotImplementedError
2. **保持功能完整** - 不能删除或简化任何现有功能
3. **向后兼容** - 保持原有 API 接口不变
4. **完整迁移** - 每个方法的完整实现都要迁移，包括所有逻辑

## 文件信息
- 原文件：strategies.py (1507行)
- 备份文件：strategies.py.backup (将创建)

## 策略分类

### 技术分析策略 (Technical Analysis)
1. **BollingerBandStrategy** - 布林带策略
   - calculate_indicators
   - precompute_all_signals
   - generate_signals

2. **StochasticStrategy** - 随机指标策略
   - calculate_indicators
   - precompute_all_signals
   - generate_signals

3. **CCIStrategy** - CCI策略
   - calculate_indicators
   - precompute_all_signals
   - generate_signals

### 统计套利策略 (Statistical Arbitrage)
4. **StatisticalArbitrageStrategy** - 统计套利基类
   - calculate_spread_zscore
   - validate_pair_correlation

5. **PairsTradingStrategy** - 配对交易策略
   - calculate_indicators
   - find_pairs
   - generate_signals

6. **MeanReversionStrategy** - 均值回归策略
   - calculate_indicators
   - generate_signals

7. **CointegrationStrategy** - 协整策略
   - calculate_indicators
   - precompute_all_signals
   - generate_signals
   - _estimate_half_life

### 因子投资策略 (Factor Investment)
8. **FactorStrategy** - 因子投资基类
   - normalize_factors
   - apply_neutralization

9. **ValueFactorStrategy** - 价值因子策略
   - calculate_indicators
   - generate_signals

10. **MomentumFactorStrategy** - 动量因子策略
    - calculate_indicators
    - generate_signals

11. **LowVolatilityStrategy** - 低波动因子策略
    - calculate_indicators
    - generate_signals

12. **MultiFactorStrategy** - 多因子组合策略
    - calculate_indicators
    - generate_signals

### 工厂类
13. **AdvancedStrategyFactory** - 高级策略工厂
    - create_strategy
    - get_available_strategies
    - register_strategy

## 拆分方案

### 目录结构
```
backend/app/services/backtest/strategies/
├── __init__.py                          # 统一导出
├── strategies.py.backup                 # 备份文件
├── base/
│   ├── __init__.py
│   ├── statistical_arbitrage_base.py   # StatisticalArbitrageStrategy
│   └── factor_base.py                   # FactorStrategy
├── technical/
│   ├── __init__.py
│   ├── bollinger_band.py               # BollingerBandStrategy
│   ├── stochastic.py                   # StochasticStrategy
│   └── cci.py                          # CCIStrategy
├── statistical_arbitrage/
│   ├── __init__.py
│   ├── pairs_trading.py                # PairsTradingStrategy
│   ├── mean_reversion.py               # MeanReversionStrategy
│   └── cointegration.py                # CointegrationStrategy
├── factor/
│   ├── __init__.py
│   ├── value_factor.py                 # ValueFactorStrategy
│   ├── momentum_factor.py              # MomentumFactorStrategy
│   ├── low_volatility.py               # LowVolatilityStrategy
│   └── multi_factor.py                 # MultiFactorStrategy
└── factory.py                          # AdvancedStrategyFactory
```

## 执行步骤

### Step 1: 创建备份
```bash
cp strategies.py strategies.py.backup
```

### Step 2: 创建目录结构
```bash
mkdir -p base technical statistical_arbitrage factor
```

### Step 3: 创建基类文件

#### base/statistical_arbitrage_base.py
- 迁移 StatisticalArbitrageStrategy 类（完整实现）
- 包含所有方法和逻辑

#### base/factor_base.py
- 迁移 FactorStrategy 类（完整实现）
- 包含所有方法和逻辑

### Step 4: 创建技术分析策略文件

#### technical/bollinger_band.py
- 迁移 BollingerBandStrategy 类（完整实现）
- 包含所有方法：calculate_indicators, precompute_all_signals, generate_signals

#### technical/stochastic.py
- 迁移 StochasticStrategy 类（完整实现）

#### technical/cci.py
- 迁移 CCIStrategy 类（完整实现）

### Step 5: 创建统计套利策略文件

#### statistical_arbitrage/pairs_trading.py
- 迁移 PairsTradingStrategy 类（完整实现）

#### statistical_arbitrage/mean_reversion.py
- 迁移 MeanReversionStrategy 类（完整实现）

#### statistical_arbitrage/cointegration.py
- 迁移 CointegrationStrategy 类（完整实现）

### Step 6: 创建因子投资策略文件

#### factor/value_factor.py
- 迁移 ValueFactorStrategy 类（完整实现）

#### factor/momentum_factor.py
- 迁移 MomentumFactorStrategy 类（完整实现）

#### factor/low_volatility.py
- 迁移 LowVolatilityStrategy 类（完整实现）

#### factor/multi_factor.py
- 迁移 MultiFactorStrategy 类（完整实现）

### Step 7: 创建工厂文件

#### factory.py
- 迁移 AdvancedStrategyFactory 类（完整实现）

### Step 8: 创建 __init__.py 文件
每个子目录和主目录都需要 __init__.py 来统一导出。

### Step 9: 验证
- 语法检查
- 导入测试
- 功能测试

### Step 10: Git 提交
```bash
git add .
git commit -m "refactor: split strategies.py into modular files"
```

## 验证清单
- [ ] 备份文件已创建
- [ ] 所有策略类已迁移（完整实现）
- [ ] 所有方法都有完整实现（无 NotImplementedError）
- [ ] 所有导入正确
- [ ] 语法检查通过
- [ ] __init__.py 正确导出所有类
- [ ] 向后兼容性保持（from strategies import * 仍然有效）
- [ ] Git 提交完成

## 重要提醒
- **不要使用 NotImplementedError**
- **不要简化或删除任何逻辑**
- **完整迁移每个方法的所有代码**
- **保持原有功能不变**
- **确保所有导入语句正确**
