# Qlib模型优化功能单元测试

## 概述

本测试文件 (`test_qlib_optimizations.py`) 针对LGBM模型优化方案中实现的所有功能进行了全面的单元测试。

## 测试覆盖范围

### 1. 标签计算逻辑测试 (`TestLabelCalculation`)

- ✅ `test_process_stock_data_with_prediction_horizon`: 测试使用prediction_horizon参数计算标签
- ✅ `test_process_stock_data_multiindex`: 测试MultiIndex数据的标签计算
- ✅ `test_label_calculation_different_horizons`: 测试不同prediction_horizon值的标签计算

**验证点：**
- 标签列正确创建
- 标签值使用 `(future_price - current_price) / current_price` 计算
- prediction_horizon参数正确应用
- 最后N行标签正确处理（NaN或0）

### 2. 缺失值处理测试 (`TestMissingValueHandling`)

- ✅ `test_handle_missing_values_price_data`: 测试价格数据的缺失值处理
- ✅ `test_handle_missing_values_indicators`: 测试技术指标的缺失值处理
- ✅ `test_handle_missing_values_high_missing_rate`: 测试高缺失率的技术指标处理

**验证点：**
- 价格数据使用前向填充+后向填充
- 技术指标根据缺失率智能选择填充策略
- 高缺失率（>50%）使用中位数填充
- 低缺失率使用前向填充

### 3. 特征标准化测试 (`TestRobustFeatureScaler`)

- ✅ `test_robust_scaler_fit_transform`: 测试fit_transform方法
- ✅ `test_robust_scaler_transform`: 测试transform方法
- ✅ `test_robust_scaler_without_sklearn`: 测试sklearn不可用时的降级处理

**验证点：**
- 使用RobustScaler进行特征标准化
- 标签列不被标准化
- 时间序列安全的标准化（避免未来信息泄漏）
- sklearn不可用时优雅降级

### 4. 异常值处理测试 (`TestOutlierHandler`)

- ✅ `test_outlier_handler_winsorize`: 测试Winsorize方法
- ✅ `test_outlier_handler_clip`: 测试Clip方法
- ✅ `test_outlier_handler_no_label_column`: 测试没有标签列时的行为
- ✅ `test_outlier_handler_extreme_returns`: 测试极端收益率（除权除息）的处理

**验证点：**
- Winsorize方法正确截断到分位数（1%和99%）
- Clip方法使用Z-score检测异常值
- 极端收益率（>50%）被正确处理
- 没有标签列时返回原始数据

### 5. 损失函数优化测试 (`TestLossFunctionOptimization`)

- ✅ `test_lightgbm_adapter_huber_loss`: 测试LightGBM适配器使用Huber损失
- ✅ `test_lightgbm_adapter_default_huber_delta`: 测试默认huber_delta值
- ✅ `test_enhanced_provider_huber_loss`: 测试EnhancedQlibDataProvider使用Huber损失

**验证点：**
- 损失函数从MSE改为Huber
- huber_delta参数正确设置
- 默认huber_delta为0.1

### 6. 集成测试 (`TestIntegration`)

- ✅ `test_prepare_training_datasets_integration`: 测试完整的数据准备流程

**验证点：**
- 所有优化功能协同工作
- 数据准备流程不报错
- 训练集和验证集正确创建

## 运行测试

### 运行所有测试

```bash
cd backend
pytest tests/test_qlib_optimizations.py -v
```

### 运行特定测试类

```bash
# 只运行标签计算测试
pytest tests/test_qlib_optimizations.py::TestLabelCalculation -v

# 只运行缺失值处理测试
pytest tests/test_qlib_optimizations.py::TestMissingValueHandling -v

# 只运行特征标准化测试
pytest tests/test_qlib_optimizations.py::TestRobustFeatureScaler -v

# 只运行异常值处理测试
pytest tests/test_qlib_optimizations.py::TestOutlierHandler -v

# 只运行损失函数测试
pytest tests/test_qlib_optimizations.py::TestLossFunctionOptimization -v
```

### 运行特定测试方法

```bash
# 运行单个测试方法
pytest tests/test_qlib_optimizations.py::TestLabelCalculation::test_process_stock_data_with_prediction_horizon -v
```

### 生成测试覆盖率报告

```bash
pytest tests/test_qlib_optimizations.py --cov=app.services.qlib --cov-report=html
```

## 测试依赖

测试需要以下Python包：

- pytest
- pytest-asyncio (用于异步测试)
- numpy
- pandas
- scikit-learn (用于RobustScaler)

安装依赖：

```bash
pip install pytest pytest-asyncio numpy pandas scikit-learn
```

## 注意事项

1. **Qlib依赖**: 某些集成测试可能需要Qlib库可用。如果Qlib不可用，相关测试会被跳过。

2. **数据生成**: 测试使用随机数据生成，每次运行结果可能略有不同，但测试逻辑应该保持一致。

3. **Mock使用**: 某些测试可能需要mock外部依赖（如Qlib初始化），以确保测试的独立性和速度。

4. **异步测试**: 使用 `@pytest.mark.asyncio` 装饰器标记异步测试方法。

## 测试结果示例

```
tests/test_qlib_optimizations.py::TestLabelCalculation::test_process_stock_data_with_prediction_horizon PASSED
tests/test_qlib_optimizations.py::TestLabelCalculation::test_process_stock_data_multiindex PASSED
tests/test_qlib_optimizations.py::TestLabelCalculation::test_label_calculation_different_horizons PASSED
tests/test_qlib_optimizations.py::TestMissingValueHandling::test_handle_missing_values_price_data PASSED
tests/test_qlib_optimizations.py::TestMissingValueHandling::test_handle_missing_values_indicators PASSED
tests/test_qlib_optimizations.py::TestMissingValueHandling::test_handle_missing_values_high_missing_rate PASSED
tests/test_qlib_optimizations.py::TestRobustFeatureScaler::test_robust_scaler_fit_transform PASSED
tests/test_qlib_optimizations.py::TestRobustFeatureScaler::test_robust_scaler_transform PASSED
tests/test_qlib_optimizations.py::TestRobustFeatureScaler::test_robust_scaler_without_sklearn PASSED
tests/test_qlib_optimizations.py::TestOutlierHandler::test_outlier_handler_winsorize PASSED
tests/test_qlib_optimizations.py::TestOutlierHandler::test_outlier_handler_clip PASSED
tests/test_qlib_optimizations.py::TestOutlierHandler::test_outlier_handler_no_label_column PASSED
tests/test_qlib_optimizations.py::TestOutlierHandler::test_outlier_handler_extreme_returns PASSED
tests/test_qlib_optimizations.py::TestLossFunctionOptimization::test_lightgbm_adapter_huber_loss PASSED
tests/test_qlib_optimizations.py::TestLossFunctionOptimization::test_lightgbm_adapter_default_huber_delta PASSED
tests/test_qlib_optimizations.py::TestLossFunctionOptimization::test_enhanced_provider_huber_loss PASSED
tests/test_qlib_optimizations.py::TestIntegration::test_prepare_training_datasets_integration PASSED

======================== 17 passed in X.XXs ========================
```

## 持续集成

建议在CI/CD流程中包含这些测试：

```yaml
# .github/workflows/test.yml 示例
- name: Run Qlib Optimizations Tests
  run: |
    cd backend
    pytest tests/test_qlib_optimizations.py -v --cov=app.services.qlib
```

## 维护

- 当添加新的优化功能时，请添加相应的测试用例
- 保持测试的独立性和可重复性
- 定期更新测试以反映代码变更
