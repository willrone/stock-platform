# 预测引擎模块

该模块包含所有与股票预测、风险评估和技术分析相关的服务，提供完整的预测解决方案。

## 主要组件

### 预测引擎
- **PredictionEngine**: 主预测引擎，协调各种预测模型
- **ModelLoader**: 模型加载器，负责加载训练好的模型
- **PredictionConfig**: 预测配置
- **PredictionOutput**: 预测输出格式

### 预测回退机制
- **PredictionFallbackEngine**: 预测回退引擎，处理预测失败情况
- **RetryManager**: 重试管理器
- **PredictionErrorHandler**: 预测错误处理器
- **FallbackStrategy**: 回退策略枚举

### 风险评估
- **RiskAssessmentService**: 风险评估服务主类
- **ConfidenceIntervalCalculator**: 置信区间计算器
- **RiskMetricsCalculator**: 风险指标计算器
- **ScenarioAnalysis**: 情景分析

### 特征提取
- **FeatureExtractor**: 特征提取器主类
- **TechnicalIndicators**: 技术指标计算器
- **StatisticalFeatures**: 统计特征计算器
- **FeatureCache**: 特征缓存管理器

### 技术指标
- **TechnicalIndicatorCalculator**: 技术指标计算器
- **TechnicalIndicatorResult**: 技术指标结果
- **BatchIndicatorRequest**: 批量指标计算请求

## 使用示例

```python
# 导入预测服务
from app.services.prediction import PredictionEngine, FeatureExtractor

# 创建预测引擎
prediction_engine = PredictionEngine()

# 配置预测参数
config = PredictionConfig(
    model_id="model_v1.0",
    stock_code="000001.SZ",
    prediction_days=5
)

# 执行预测
result = await prediction_engine.predict(config)

# 特征提取
feature_extractor = FeatureExtractor()
features = await feature_extractor.extract_features("000001.SZ", start_date, end_date)
```

## 支持的预测类型

- **价格预测**: 股票价格趋势预测
- **波动率预测**: 价格波动率预测
- **方向预测**: 涨跌方向预测
- **风险评估**: VaR、CVaR 等风险指标

## 技术指标支持

- **趋势指标**: MA、EMA、MACD、ADX
- **动量指标**: RSI、STOCH、Williams %R
- **波动率指标**: Bollinger Bands、ATR
- **成交量指标**: OBV、VWAP

## 风险评估功能

- **置信区间**: 预测结果的置信区间计算
- **风险指标**: VaR、CVaR、最大回撤
- **情景分析**: 多种市场情景下的表现分析
- **压力测试**: 极端市场条件下的风险评估

## 配置

预测模块支持以下配置：

- 模型选择策略
- 特征工程参数
- 风险评估参数
- 回退策略配置

## 依赖关系

该模块依赖于：
- 模型模块（预测模型）
- 数据模块（历史数据）
- 基础设施模块（缓存）

## 注意事项

1. 预测结果仅供参考，不构成投资建议
2. 建议使用多个模型进行集成预测
3. 风险评估应结合市场环境进行调整
4. 特征缓存可以显著提升预测性能