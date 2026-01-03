# 回测数据库扩展功能说明

## 概述

本扩展为回测结果可视化系统提供了完整的数据库支持，包括详细的回测数据存储、图表数据缓存和过期机制。

## 新增数据库表

### 1. backtest_detailed_results
存储回测的扩展风险指标和分析数据
- 扩展风险指标：索提诺比率、卡玛比率、VaR等
- 回撤详细分析数据
- 月度收益分析数据
- 持仓分析数据
- 基准对比数据
- 滚动指标数据

### 2. backtest_chart_cache
图表数据缓存表，提高图表加载性能
- 支持多种图表类型缓存
- 自动过期机制
- 数据哈希验证

### 3. portfolio_snapshots
组合快照表，存储组合历史数据用于绘制收益曲线
- 按日期存储组合价值
- 包含现金、持仓数量、收益率、回撤等信息
- 支持持仓详情JSON存储

### 4. trade_records
交易记录表，存储详细的交易记录
- 完整的交易信息：股票代码、动作、数量、价格等
- 盈亏计算和持仓天数
- 技术指标快照

### 5. backtest_benchmarks
基准数据表，存储基准指数数据用于对比
- 基准历史数据
- 对比指标：相关系数、贝塔、阿尔法等

## 核心服务

### 1. BacktestDetailedRepository
数据仓库类，提供所有新表的CRUD操作
- 批量数据插入
- 复杂查询和统计
- 数据清理功能

### 2. ChartCacheService
图表缓存服务
- 智能缓存管理
- 过期时间控制
- 数据哈希验证
- 支持的图表类型：
  - equity_curve (收益曲线)
  - drawdown_curve (回撤曲线)
  - monthly_heatmap (月度收益热力图)
  - trade_distribution (交易分布图)
  - position_weights (持仓权重图)
  - risk_metrics (风险指标图)
  - rolling_metrics (滚动指标图)
  - benchmark_comparison (基准对比图)

### 3. CacheCleanupService
缓存清理服务
- 定时清理过期缓存
- 清理旧的回测数据
- 可配置的清理策略

## 数据库迁移

### 执行迁移
```bash
cd backend
source venv/bin/activate
python manage_backtest_db.py migrate
```

### 验证迁移
```bash
python manage_backtest_db.py verify
```

### 回滚迁移（谨慎使用）
```bash
python manage_backtest_db.py rollback
```

### 查看统计信息
```bash
python manage_backtest_db.py stats
```

### 清理缓存和旧数据
```bash
python manage_backtest_db.py cleanup
```

## API端点

新增的API端点位于 `/api/v1/backtest-detailed/`：

### 获取详细结果
```
GET /api/v1/backtest-detailed/{task_id}/detailed-result
```

### 获取组合快照
```
GET /api/v1/backtest-detailed/{task_id}/portfolio-snapshots
参数：start_date, end_date, limit
```

### 获取交易记录
```
GET /api/v1/backtest-detailed/{task_id}/trade-records
参数：stock_code, action, start_date, end_date, offset, limit, order_by, order_desc
```

### 获取交易统计
```
GET /api/v1/backtest-detailed/{task_id}/trade-statistics
```

### 获取基准数据
```
GET /api/v1/backtest-detailed/{task_id}/benchmark-data
参数：benchmark_symbol
```

### 缓存管理
```
POST /api/v1/backtest-detailed/{task_id}/cache-chart
GET /api/v1/backtest-detailed/{task_id}/cached-chart/{chart_type}
DELETE /api/v1/backtest-detailed/{task_id}/cache
GET /api/v1/backtest-detailed/cache/statistics
DELETE /api/v1/backtest-detailed/cache/cleanup
```

### 数据管理
```
DELETE /api/v1/backtest-detailed/{task_id}/data
```

## 使用示例

### 1. 创建详细回测数据
```python
from app.repositories.backtest_detailed_repository import BacktestDetailedRepository

async def create_detailed_data(session, task_id, backtest_id):
    repository = BacktestDetailedRepository(session)
    
    # 扩展风险指标
    extended_metrics = {
        'sortino_ratio': 1.5,
        'calmar_ratio': 0.8,
        'max_drawdown_duration': 15,
        'var_95': -0.02,
        'downside_deviation': 0.12
    }
    
    # 分析数据
    analysis_data = {
        'drawdown_analysis': {...},
        'monthly_returns': [...],
        'position_analysis': [...]
    }
    
    # 创建详细结果
    await repository.create_detailed_result(
        task_id, backtest_id, extended_metrics, analysis_data
    )
    
    # 批量创建组合快照
    snapshots_data = [...]
    await repository.batch_create_portfolio_snapshots(
        task_id, backtest_id, snapshots_data
    )
    
    # 批量创建交易记录
    trades_data = [...]
    await repository.batch_create_trade_records(
        task_id, backtest_id, trades_data
    )
```

### 2. 使用图表缓存
```python
from app.services.backtest.chart_cache_service import chart_cache_service

# 缓存图表数据
chart_data = {"series": [...], "options": {...}}
await chart_cache_service.cache_chart_data(
    task_id="task_001",
    chart_type="equity_curve",
    chart_data=chart_data,
    expiry_hours=24
)

# 获取缓存数据
cached_data = await chart_cache_service.get_cached_chart_data(
    task_id="task_001",
    chart_type="equity_curve"
)
```

### 3. 清理服务
```python
from app.services.backtest.cache_cleanup_service import cache_cleanup_service

# 手动清理
cleanup_results = await cache_cleanup_service.manual_cleanup(
    cleanup_expired_cache=True,
    cleanup_old_data=True,
    custom_retention_days=30
)

# 获取统计信息
stats = await cache_cleanup_service.get_cleanup_statistics()
```

## 测试

运行测试脚本验证功能：
```bash
cd backend
source venv/bin/activate
python test_backtest_db_extension.py
```

## 配置

### 缓存配置
- 默认缓存过期时间：24小时
- 支持的图表类型：8种
- 自动清理间隔：6小时

### 数据保留策略
- 默认数据保留期：30天
- 可通过清理服务配置调整

## 注意事项

1. **数据一致性**：所有数据操作都在事务中进行，确保数据一致性
2. **性能优化**：使用批量插入和索引优化查询性能
3. **缓存策略**：图表缓存使用哈希验证，避免数据不一致
4. **清理机制**：定期清理过期数据，避免数据库膨胀
5. **错误处理**：完善的错误处理和日志记录

## 扩展性

该扩展设计具有良好的扩展性：
- 可以轻松添加新的图表类型
- 支持自定义分析数据格式
- 可以扩展更多的风险指标
- 支持多种基准对比

## 维护

定期执行以下维护操作：
1. 检查缓存统计信息
2. 清理过期数据
3. 监控数据库大小
4. 备份重要数据

通过这些功能，回测结果可视化系统将具备完整的数据支持能力，为前端提供丰富的数据接口和高性能的缓存机制。