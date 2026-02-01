# LightGBM模型特征选择功能实现总结

## 概述

实现了LightGBM模型（以及其他Qlib模型）的特征选择功能，允许用户在创建模型时选择使用哪些特征维度进行训练。

## 实现的功能

### 1. 配置类修改

**文件**: `backend/app/services/qlib/unified_qlib_training_engine.py`

- 在`QlibTrainingConfig`类中添加了`selected_features`参数
- 类型：`Optional[List[str]]`
- 默认值：`None`（表示使用所有可用特征）
- 当提供特征列表时，只使用用户指定的特征进行训练

### 2. 数据准备逻辑修改

**文件**: `backend/app/services/qlib/unified_qlib_training_engine.py`

- 修改了`_prepare_training_datasets`方法，添加了`config`参数
- 在特征选择时，如果配置中指定了`selected_features`，则只使用选定的特征
- 添加了特征验证逻辑：
  - 检查用户指定的特征是否在数据中存在
  - 如果指定的特征都不存在，回退到使用所有可用特征
  - 记录警告信息，提示哪些特征不存在

### 3. API接口修改

**文件**: `backend/app/api/v1/schemas.py`

- 在`ModelTrainingRequest`类中添加了`selected_features`字段
- 类型：`Optional[List[str]]`
- 描述：选择的特征列表，如果为空则使用所有可用特征

**文件**: `backend/app/api/v1/models.py`

- 修改了`train_model_task`函数，添加了`selected_features`参数
- 在创建`QlibTrainingConfig`时传递`selected_features`
- 在超参数调优时也传递`selected_features`，确保调优过程使用相同的特征集
- 修改了`create_training_task`路由，从请求中提取`selected_features`并传递给训练任务

### 4. 新增API接口

**文件**: `backend/app/api/v1/models.py`

- 新增`GET /api/v1/models/available-features`接口
- 功能：获取可用于模型训练的特征列表
- 支持参数：
  - `stock_code`（可选）：股票代码
  - `start_date`（可选）：开始日期
  - `end_date`（可选）：结束日期
- 返回内容：
  - 如果提供了股票代码和日期范围，返回基于实际数据的可用特征
  - 否则返回理论上的所有可能特征
  - 包含特征分类（基础特征、技术指标特征、基本面特征、Alpha因子特征）

## 可用特征类别

### 基础价格特征
- `open`, `high`, `low`, `close`, `volume`

### 技术指标特征
- 移动平均线：`ma_5`, `ma_10`, `ma_20`, `ma_60`, `sma`, `ema`, `wma`
- 动量指标：`rsi`, `stoch`, `williams_r`, `cci`, `momentum`, `roc`
- 趋势指标：`macd`, `macd_signal`, `macd_histogram`, `bb_upper`, `bb_middle`, `bb_lower`, `bb_width`, `bb_position`, `sar`, `adx`
- 成交量指标：`vwap`, `obv`, `ad_line`, `volume_rsi`
- 波动率指标：`atr`, `volatility`, `historical_volatility`
- 复合指标：`kdj_k`, `kdj_d`, `kdj_j`

### 基本面特征
- `price_change`, `price_change_5d`, `price_change_20d`
- `volume_change`, `volume_ma_ratio`
- `volatility_5d`, `volatility_20d`
- `price_position`

### Alpha因子特征
- Alpha158因子：`alpha_001` 到 `alpha_158`（如果启用Alpha因子）

## 使用示例

### 1. 获取可用特征列表

```bash
# 获取理论特征列表
GET /api/v1/models/available-features

# 获取基于实际数据的特征列表
GET /api/v1/models/available-features?stock_code=000001.SZ&start_date=2024-01-01&end_date=2024-12-31
```

### 2. 创建模型时选择特征

```json
{
  "model_name": "my_model",
  "model_type": "lightgbm",
  "stock_codes": ["000001.SZ"],
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "hyperparameters": {
    "learning_rate": 0.1,
    "num_iterations": 100
  },
  "selected_features": [
    "close",
    "volume",
    "ma_5",
    "ma_20",
    "rsi",
    "macd",
    "bb_upper",
    "bb_lower"
  ]
}
```

### 3. 使用所有特征（默认行为）

```json
{
  "model_name": "my_model",
  "model_type": "lightgbm",
  "stock_codes": ["000001.SZ"],
  "start_date": "2024-01-01",
  "end_date": "2024-12-31",
  "hyperparameters": {
    "learning_rate": 0.1,
    "num_iterations": 100
  }
  // 不提供selected_features或设置为null，将使用所有可用特征
}
```

## 注意事项

1. **特征验证**：如果用户指定的特征在数据中不存在，系统会：
   - 记录警告日志
   - 忽略不存在的特征
   - 如果所有指定特征都不存在，回退到使用所有可用特征

2. **预测时的特征对齐**：预测时会自动从模型中获取训练时使用的特征名称，确保预测数据使用相同的特征集。

3. **超参数调优**：超参数调优过程会使用相同的特征集，确保调优结果的一致性。

4. **特征数量限制**：建议选择的特征数量不要过多，以免影响训练速度和模型性能。

## 前端实现

### 1. DataService扩展

**文件**: `frontend/src/services/dataService.ts`

- 添加了`getAvailableFeatures`方法，用于获取可用特征列表
- 支持传入股票代码和日期范围，获取基于实际数据的特征列表
- 返回特征分类信息（基础特征、技术指标、基本面特征、Alpha因子）

### 2. 模型创建表单扩展

**文件**: `frontend/src/app/models/page.tsx`

- 在`formData`中添加了`selected_features`字段
- 添加了特征选择UI组件：
  - 复选框选项：使用所有特征（默认）或自定义选择
  - 按类别显示特征（基础价格、技术指标、基本面、Alpha因子）
  - 支持多选特征
  - 显示已选择特征数量
  - 提供清空选择功能
- 添加了特征列表加载功能：
  - 当选择股票和日期后，自动加载基于实际数据的特征列表
  - 提供手动刷新特征列表按钮
- 在表单提交时，根据用户选择传递`selected_features`参数

### 3. 用户体验优化

- 默认使用所有特征，适合大多数用户
- 高级用户可以选择特定特征进行训练
- 特征按类别分组显示，便于查找和选择
- 实时显示已选择特征数量
- 支持基于实际数据的特征列表，确保选择的特征在数据中存在

## 技术细节

- 特征选择在数据准备阶段进行，在`_prepare_training_datasets`方法中实现
- 特征过滤逻辑确保只使用数据中实际存在的特征
- 所有修改都向后兼容，不提供`selected_features`时行为与之前一致
- 前端自动根据选择的股票和日期加载对应的特征列表

