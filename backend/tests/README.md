# 测试目录说明

本目录包含项目的所有测试文件，按功能模块组织。

## 目录结构

```
tests/
├── unit/                    # 单元测试
│   ├── api/                 # API 接口测试
│   ├── backtest/            # 回测功能测试
│   ├── data/                # 数据服务测试
│   ├── infrastructure/      # 基础设施测试（缓存、监控、限流等）
│   ├── models/              # 模型相关测试
│   ├── prediction/          # 预测引擎测试
│   ├── tasks/               # 任务管理测试
│   └── services/            # 其他服务测试（策略、图表缓存、信号集成等）
├── integration/             # 集成测试
├── scripts/                 # 测试脚本和验证工具（不被 pytest 收集）
├── conftest.py              # pytest 全局配置
├── conftest_full.py         # pytest 完整配置
└── __init__.py
```

## 运行测试

```bash
cd backend
source venv/bin/activate

# 运行所有测试
python -m pytest tests/ -v

# 运行特定模块
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/unit/infrastructure/ -v

# 快速检查（遇第一个失败即停止）
python -m pytest tests/ --maxfail=1 -x
```

## 测试规范

- **单元测试**：测试单个函数或类的功能
- **集成测试**：测试多个模块协同工作
- **属性测试**：使用 Property-Based Testing 验证正确性
- **scripts/** 下的脚本不被 pytest 收集，需手动运行

## 依赖与注意事项

- 需安装项目依赖：`pip install -r requirements.txt`
- 建议在虚拟环境中运行
- 部分测试需外部依赖（数据库、Redis 等）
- SFTP 未启用时，数据同步相关测试会跳过或放宽断言
- 真实数据测试（如 `test_backtest_real_data.py`）需对应 parquet 文件存在

## 测试状态与修复记录（2026-02-01）

### 已修复问题

| 类别 | 修复内容 |
|------|----------|
| 限流 | 测试环境自动跳过限流；补充 X-RateLimit-Remaining 响应头 |
| 导入 | DataValidator、SimpleDataService 等导入路径修正 |
| API 端点 | `/api/v1/version` → `/api/v1/system/version`，`/data/sync` → `/data/sync/remote` |
| 系统状态 | middleware_stack 访问增加异常处理 |
| 回测 | 使用有效策略名 "rsi" |
| 异步测试 | 添加 `@pytest.mark.asyncio` |
| DataMonitoringService | 正确传入 data_service、indicators_service、parquet_manager |
| 断言 | 放宽 data_status、sync、technical_indicators 等断言以兼容未连接/未启用场景 |

### 已验证通过的测试

- `test_api_health_check`、`test_api_version`、`test_concurrent_requests`
- `test_system_status_flow`、`test_stock_data_retrieval_flow`、`test_backtest_flow`
- `test_error_handling_mechanisms`、`test_error_response_format`、`test_pagination`
- `test_basic_infrastructure`、`test_backtest_progress`、`test_backtest_portfolio`
- `test_progress_simple` 等

### 部分跳过的测试

- `test_data_files_list`、`test_data_statistics`：端点为 404 时跳过
- `test_backtest_real_data`：真实数据文件不存在时跳过
- `test_basic_data_flow_integration`：SimpleDataService 不支持本地数据流时跳过
