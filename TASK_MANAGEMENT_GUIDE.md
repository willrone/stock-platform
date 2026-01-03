# 任务管理指南

## 问题描述

回测任务可能会卡在运行状态（通常是90%进度），无法正常完成、停止或删除。这通常发生在：

1. 回测执行完成后，在保存结果时出现异常
2. 任务进程意外中断，但数据库状态未更新
3. 网络或数据库连接问题导致状态更新失败

## 解决方案

### 1. 自动解决方案

系统现在包含了自动任务清理服务，会定期检查和清理卡住的任务：

- **检查间隔**: 每30分钟
- **任务超时**: 60分钟
- **自动处理**: 将卡住的任务标记为已取消

### 2. 手动解决方案

#### 使用管理工具

我们提供了 `manage_tasks.py` 工具来管理任务：

```bash
# 查看所有任务
python3 manage_tasks.py list

# 查看卡住的任务（默认30分钟超时）
python3 manage_tasks.py stuck

# 查看卡住的任务（自定义超时时间）
python3 manage_tasks.py stuck --timeout 60

# 停止指定任务
python3 manage_tasks.py stop --task-id <task_id>

# 删除指定任务
python3 manage_tasks.py delete --task-id <task_id>

# 强制完成任务（设置为已取消）
python3 manage_tasks.py force --task-id <task_id>

# 强制完成任务（设置为失败）
python3 manage_tasks.py force --task-id <task_id> --status failed

# 自动清理所有卡住的任务
python3 manage_tasks.py cleanup --auto-fix

# 查看任务统计
python3 manage_tasks.py stats
```

#### 使用API接口

系统提供了以下API接口来管理任务：

```bash
# 获取卡住的任务
curl "http://localhost:8000/api/v1/tasks/monitor/stuck?timeout_minutes=30"

# 清理卡住的任务
curl -X POST "http://localhost:8000/api/v1/tasks/monitor/cleanup?timeout_minutes=30&auto_fix=true"

# 强制完成指定任务
curl -X POST "http://localhost:8000/api/v1/tasks/monitor/force-complete/{task_id}?status=cancelled"

# 获取任务统计
curl "http://localhost:8000/api/v1/tasks/monitor/statistics"
```

#### 使用修复脚本

如果需要一次性修复所有问题，可以使用 `fix_stuck_tasks.py`：

```bash
python3 fix_stuck_tasks.py
```

这个脚本会：
1. 检查后端服务状态
2. 找到所有卡住的任务
3. 尝试通过API停止任务
4. 如果API失败，直接在数据库中更新状态
5. 删除已处理的任务

## 预防措施

### 1. 系统级预防

- **任务超时机制**: 任务执行超过设定时间会自动取消
- **定期清理服务**: 自动检查和处理卡住的任务
- **改进的错误处理**: 更好的异常捕获和状态更新
- **进度跟踪优化**: 更精确的进度更新逻辑

### 2. 监控和告警

- **任务监控**: 实时监控任务状态和进度
- **统计信息**: 提供任务执行统计和趋势分析
- **日志记录**: 详细的任务执行日志

### 3. 最佳实践

1. **定期检查**: 定期使用管理工具检查任务状态
2. **及时处理**: 发现卡住任务及时处理，避免积累
3. **监控日志**: 关注后端日志中的错误信息
4. **资源管理**: 确保系统有足够的资源执行任务

## 故障排除

### 常见问题

1. **任务卡在90%**: 通常是保存结果时出现问题
   - 解决方案: 使用 `force` 命令强制完成任务

2. **无法删除任务**: 只能删除已完成、失败或已取消的任务
   - 解决方案: 先停止或强制完成任务，再删除

3. **后端服务无响应**: 管理工具无法连接到后端
   - 解决方案: 检查后端服务状态，重启服务

### 紧急处理

如果系统出现严重问题，可以直接操作数据库：

```python
import sqlite3
from datetime import datetime

# 连接数据库
conn = sqlite3.connect('backend/data/app.db')
cursor = conn.cursor()

# 将所有运行中的任务标记为已取消
cursor.execute('''
    UPDATE tasks 
    SET status = 'cancelled', 
        completed_at = ?, 
        progress = 100.0 
    WHERE status = 'running'
''', (datetime.now().isoformat(),))

conn.commit()
conn.close()
```

## 系统改进

### 已实现的改进

1. **任务监控服务** (`TaskMonitor`): 检测和处理卡住的任务
2. **任务清理服务** (`TaskCleanupService`): 定期自动清理
3. **管理工具** (`manage_tasks.py`): 命令行管理界面
4. **API接口**: RESTful API管理接口
5. **改进的错误处理**: 更好的异常捕获和恢复

### 未来改进计划

1. **任务队列优化**: 更好的任务调度和执行
2. **实时监控界面**: Web界面实时监控任务状态
3. **告警系统**: 任务异常时自动告警
4. **性能优化**: 提高任务执行效率和稳定性

## 联系支持

如果遇到无法解决的问题，请：

1. 收集相关日志信息
2. 记录问题复现步骤
3. 提供任务配置信息
4. 联系技术支持团队