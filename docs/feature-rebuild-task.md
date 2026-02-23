# 任务重建功能

## 功能说明

在任务详情页面添加了"重建任务"按钮，允许用户基于现有任务的配置快速创建新任务。

## 使用方法

1. 打开任何已完成或失败的任务详情页
2. 点击页面顶部的"重建任务"按钮（Copy 图标）
3. 系统会自动跳转到任务创建页面，并预填充以下配置：
   - 任务名称（原名称 + "(重建)"后缀）
   - 任务类型（回测/预测）
   - 股票代码列表
   - 所有策略参数和配置

4. 根据需要修改配置参数
5. 点击"创建任务"提交

## 实现细节

### 前端修改

#### 1. 任务详情页 (`frontend/src/app/tasks/[id]/page.tsx`)

**新增功能：**
- 导入 `Copy` 图标
- 添加 `handleRebuild` 函数：
  - 提取当前任务的所有配置参数
  - 构建 URL 查询参数
  - 跳转到创建页面

**按钮位置：**
- 位于页面顶部操作栏
- 在"刷新"按钮之后，"重新运行"按钮之前
- 所有任务状态下都可用

#### 2. 任务创建页 (`frontend/src/app/tasks/create/page.tsx`)

**新增功能：**
- 导入 `useSearchParams` hook
- 添加 URL 参数解析逻辑：
  - 检测 `rebuild=true` 参数
  - 读取并应用所有配置参数
  - 支持回测和预测两种任务类型
  - 支持单一策略和组合策略

**支持的参数：**

**通用参数：**
- `rebuild`: 标识是否为重建模式
- `task_type`: 任务类型（backtest/prediction）
- `task_name`: 任务名称
- `stock_codes`: 股票代码列表（逗号分隔）

**回测任务参数：**
- `strategy_name`: 策略名称
- `start_date`: 开始日期
- `end_date`: 结束日期
- `initial_cash`: 初始资金
- `commission_rate`: 手续费率
- `slippage_rate`: 滑点率
- `enable_performance_profiling`: 是否启用性能分析
- `strategy_config`: 策略配置（JSON 字符串）

**预测任务参数：**
- `model_id`: 模型 ID
- `horizon`: 预测周期
- `confidence_level`: 置信度
- `risk_assessment`: 是否启用风险评估

## 技术要点

### URL 参数传递

使用 `URLSearchParams` 构建查询字符串：
```typescript
const params = new URLSearchParams();
params.set('rebuild', 'true');
params.set('task_type', currentTask.task_type);
// ... 更多参数
router.push(`/tasks/create?${params.toString()}`);
```

### 策略配置序列化

复杂的策略配置通过 JSON 序列化传递：
```typescript
if (bc.strategy_config) {
  params.set('strategy_config', JSON.stringify(bc.strategy_config));
}
```

### 组合策略支持

自动识别组合策略并正确设置：
```typescript
if (strategyName === 'portfolio' && config.strategies) {
  setPortfolioConfig({
    strategies: config.strategies || [],
    integration_method: config.integration_method || 'weighted_voting',
  });
}
```

## 测试场景

### 场景 1：重建回测任务（单一策略）
1. 打开一个使用 RSI 策略的回测任务
2. 点击"重建任务"
3. 验证所有参数已预填充
4. 修改股票数量或日期范围
5. 创建新任务

### 场景 2：重建回测任务（组合策略）
1. 打开一个使用组合策略的回测任务
2. 点击"重建任务"
3. 验证所有子策略和权重已预填充
4. 调整子策略权重
5. 创建新任务

### 场景 3：重建预测任务
1. 打开一个预测任务
2. 点击"重建任务"
3. 验证模型、股票、预测周期等参数已预填充
4. 修改置信度或风险评估设置
5. 创建新任务

## 注意事项

1. **任务名称自动添加后缀**：重建的任务会自动在原名称后添加 "(重建)" 后缀，避免名称冲突
2. **所有状��都支持**：无论任务是运行中、已完成还是失败，都可以重建
3. **配置完整性**：会尽可能保留原任务的所有配置，包括性能分析开关等细节
4. **灵活修改**：跳转到创建页面后，用户可以自由修改任何参数

## 未来改进

1. **配置模板保存**：允许将常用配置保存为模板
2. **批量重建**：支持选择多个任务批量重建
3. **参数对比**：在重建前显示与原任务的参数差异
4. **快��调整**：提供常用参数的快速调整选项（如日期范围、股票数量）

## 相关文件

- `frontend/src/app/tasks/[id]/page.tsx` - 任务详情页
- `frontend/src/app/tasks/create/page.tsx` - 任务创建页
- `frontend/src/services/taskService.ts` - 任务服务（未修改）

## 更新日期

2026-02-07
