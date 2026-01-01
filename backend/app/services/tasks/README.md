# 任务管理模块

该模块包含所有与异步任务调度、执行和通知相关的服务，提供完整的任务管理解决方案。

## 主要组件

### 任务管理
- **TaskManager**: 任务管理器，负责任务的创建、更新、查询和删除
- **TaskCreateRequest**: 任务创建请求
- **TaskUpdateRequest**: 任务更新请求
- **TaskQuery**: 任务查询条件
- **TaskSummary**: 任务摘要信息

### 任务队列
- **TaskQueueManager**: 任务队列管理器，统一管理多个调度器
- **TaskScheduler**: 任务调度器，负责任务的调度和分发
- **TaskExecutor**: 任务执行器，执行具体的任务
- **TaskPriority**: 任务优先级枚举
- **QueuedTask**: 队列中的任务
- **TaskExecutionContext**: 任务执行上下文

### 任务执行引擎
- **TaskExecutionEngine**: 任务执行引擎，统一管理所有类型的任务执行器
- **PredictionTaskExecutor**: 预测任务执行器
- **BacktestTaskExecutor**: 回测任务执行器
- **TrainingTaskExecutor**: 模型训练任务执行器
- **ProgressTracker**: 进度跟踪器
- **TaskProgress**: 任务进度信息

### 任务通知
- **TaskNotificationService**: 任务通知服务
- **TaskStatusNotification**: 任务状态通知
- **TaskProgressNotification**: 任务进度通知

## 使用示例

```python
# 导入任务服务
from app.services.tasks import TaskManager, TaskQueueManager, TaskExecutionEngine

# 创建任务管理器
task_manager = TaskManager()

# 创建任务
request = TaskCreateRequest(
    name="股票预测任务",
    task_type="prediction",
    parameters={"stock_code": "000001.SZ", "days": 5}
)
task = await task_manager.create_task(request)

# 提交任务到队列
queue_manager = TaskQueueManager()
await queue_manager.submit_task(task.id, priority=TaskPriority.HIGH)

# 执行任务
execution_engine = TaskExecutionEngine()
result = await execution_engine.execute_task(task.id)
```

## 支持的任务类型

### 预测任务
- 股票价格预测
- 波动率预测
- 趋势预测
- 批量预测

### 回测任务
- 策略回测
- 参数优化
- 组合回测
- 风险分析

### 训练任务
- 模型训练
- 超参数调优
- 集成学习
- 在线学习

### 数据任务
- 数据同步
- 数据清洗
- 特征工程
- 数据验证

## 任务优先级

任务支持三个优先级：

- **HIGH (1)**: 高优先级，紧急任务
- **MEDIUM (2)**: 中等优先级，常规任务
- **LOW (3)**: 低优先级，后台任务

## 任务状态

任务具有以下状态：

- **PENDING**: 等待执行
- **RUNNING**: 正在执行
- **COMPLETED**: 执行完成
- **FAILED**: 执行失败
- **CANCELLED**: 已取消

## 进度跟踪

任务执行过程中支持实时进度跟踪：

```python
# 获取任务进度
progress = await progress_tracker.get_progress(task_id)
print(f"进度: {progress.percentage:.1%}")
print(f"状态: {progress.status}")
print(f"消息: {progress.message}")
```

## 通知机制

任务管理模块支持多种通知方式：

- **WebSocket 通知**: 实时推送任务状态变化
- **回调通知**: 任务完成后调用指定的回调函数
- **邮件通知**: 重要任务的邮件提醒

## 配置选项

任务模块支持以下配置：

- **队列配置**: 队列大小、工作线程数
- **执行配置**: 超时时间、重试次数
- **通知配置**: 通知方式、通知频率
- **监控配置**: 性能监控、错误追踪

## 依赖关系

该模块依赖于：
- 模型模块（训练任务）
- 预测模块（预测任务）
- 回测模块（回测任务）
- 基础设施模块（缓存、监控、WebSocket）

## 注意事项

1. 长时间运行的任务建议设置合理的超时时间
2. 高优先级任务会优先执行
3. 任务失败时会自动重试（可配置）
4. 建议监控任务队列的长度和执行效率
5. 大批量任务建议分批提交以避免系统过载