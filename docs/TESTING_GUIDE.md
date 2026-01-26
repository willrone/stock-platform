# 测试指南

本文档描述了项目的测试策略、测试类型和最佳实践。

## 📋 目录

- [测试策略](#测试策略)
- [测试类型](#测试类型)
- [测试工具](#测试工具)
- [编写测试](#编写测试)
- [运行测试](#运行测试)
- [测试覆盖率](#测试覆盖率)

## 🎯 测试策略

### 测试金字塔

```
        /\
       /E2E\        ← 少量端到端测试
      /------\
     /Integration\  ← 适量集成测试
    /------------\
   /   Unit Tests  \ ← 大量单元测试
  /----------------\
```

### 测试原则
1. **快速**: 单元测试应该快速执行
2. **独立**: 测试之间不应相互依赖
3. **可重复**: 测试结果应该一致
4. **自验证**: 测试应该自动验证结果
5. **及时**: 测试应该及时编写

## 🧪 测试类型

### 1. 单元测试（Unit Tests）

测试单个函数或类的行为。

**后端示例**:
```python
import pytest
from app.services.data_service import DataService

def test_fetch_stock_data_success():
    """测试成功获取股票数据"""
    service = DataService()
    data = service.fetch_stock_data("000001", "2024-01-01", "2024-01-31")
    
    assert data is not None
    assert len(data) > 0
    assert "close" in data.columns
```

**前端示例**:
```typescript
import { render, screen } from '@testing-library/react';
import { PositionAnalysis } from './PositionAnalysis';

describe('PositionAnalysis', () => {
  it('应该渲染持仓分析表格', () => {
    const mockData = [
      { stock_code: '000001', total_return: 0.15, win_rate: 0.6 }
    ];
    
    render(<PositionAnalysis positionAnalysis={mockData} stockCodes={[]} />);
    
    expect(screen.getByText('000001')).toBeInTheDocument();
  });
});
```

### 2. 集成测试（Integration Tests）

测试多个组件之间的交互。

**后端示例**:
```python
import pytest
from fastapi.testclient import TestClient
from app.main import create_application

@pytest.fixture
def client():
    app = create_application()
    return TestClient(app)

def test_create_backtest_task(client):
    """测试创建回测任务"""
    response = client.post(
        "/api/v1/backtest/tasks",
        json={
            "strategy": "momentum",
            "start_date": "2024-01-01",
            "end_date": "2024-01-31"
        }
    )
    
    assert response.status_code == 200
    assert "task_id" in response.json()["data"]
```

### 3. 属性测试（Property-Based Tests）

使用Hypothesis生成测试用例，验证代码属性。

```python
from hypothesis import given, strategies as st
from app.services.technical_indicators import calculate_rsi

@given(
    prices=st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=14, max_size=100)
)
def test_rsi_properties(prices):
    """测试RSI指标的基本属性"""
    rsi = calculate_rsi(prices)
    
    # 属性1: RSI值在0-100之间
    assert 0 <= rsi <= 100
    
    # 属性2: 如果价格持续上涨，RSI应该较高
    if all(prices[i] < prices[i+1] for i in range(len(prices)-1)):
        assert rsi > 50
```

### 4. 端到端测试（E2E Tests）

测试完整的用户工作流程。

```python
def test_complete_prediction_workflow(client):
    """测试完整的预测工作流程"""
    # 1. 创建预测任务
    task_response = client.post("/api/v1/prediction/tasks", json={...})
    task_id = task_response.json()["data"]["task_id"]
    
    # 2. 等待任务完成
    while True:
        status_response = client.get(f"/api/v1/tasks/{task_id}")
        status = status_response.json()["data"]["status"]
        if status == "completed":
            break
        time.sleep(1)
    
    # 3. 获取预测结果
    result_response = client.get(f"/api/v1/prediction/tasks/{task_id}/results")
    assert result_response.status_code == 200
    assert "predictions" in result_response.json()["data"]
```

## 🛠️ 测试工具

### 后端工具

| 工具 | 用途 | 安装 |
|------|------|------|
| pytest | 测试框架 | `pip install pytest` |
| pytest-cov | 覆盖率报告 | `pip install pytest-cov` |
| pytest-asyncio | 异步测试 | `pip install pytest-asyncio` |
| pytest-mock | Mock对象 | `pip install pytest-mock` |
| hypothesis | 属性测试 | `pip install hypothesis` |

### 前端工具

| 工具 | 用途 | 安装 |
|------|------|------|
| Jest | 测试框架 | `npm install --save-dev jest` |
| Testing Library | React组件测试 | `npm install --save-dev @testing-library/react` |
| fast-check | 属性测试 | `npm install --save-dev fast-check` |

## ✍️ 编写测试

### 测试命名

使用描述性的测试名称，说明测试的内容：

```python
# ✅ 好的命名
def test_calculate_returns_with_positive_prices():
    """测试使用正价格计算收益率"""
    pass

def test_fetch_data_handles_network_error():
    """测试获取数据时处理网络错误"""
    pass

# ❌ 避免的命名
def test_function1():
    pass

def test_data():
    pass
```

### 测试结构（AAA模式）

使用 Arrange-Act-Assert 模式：

```python
def test_process_stock_data():
    # Arrange: 准备测试数据
    raw_data = [100.0, 105.0, 110.0]
    expected_result = {"avg": 105.0, "max": 110.0}
    
    # Act: 执行被测试的操作
    result = process_stock_data(raw_data)
    
    # Assert: 验证结果
    assert result["avg"] == expected_result["avg"]
    assert result["max"] == expected_result["max"]
```

### 使用Fixtures

共享测试数据和设置：

```python
@pytest.fixture
def sample_stock_data():
    """提供示例股票数据"""
    return {
        "code": "000001",
        "prices": [100.0, 105.0, 110.0],
        "dates": ["2024-01-01", "2024-01-02", "2024-01-03"]
    }

def test_analyze_stock(sample_stock_data):
    """使用fixture的测试"""
    result = analyze_stock(sample_stock_data)
    assert result is not None
```

### Mock外部依赖

```python
from unittest.mock import Mock, patch

@patch('app.services.data_service.requests.get')
def test_fetch_remote_data(mock_get):
    """测试获取远程数据"""
    # 模拟API响应
    mock_get.return_value.json.return_value = {"data": "test"}
    mock_get.return_value.status_code = 200
    
    service = DataService()
    result = service.fetch_remote_data("url")
    
    assert result == {"data": "test"}
    mock_get.assert_called_once_with("url")
```

## 🚀 运行测试

### 运行所有测试

```bash
# 后端
cd backend
pytest tests/

# 前端
cd frontend
npm test
```

### 运行特定测试

```bash
# 运行特定文件
pytest tests/test_data_service.py

# 运行特定测试函数
pytest tests/test_data_service.py::test_fetch_data

# 运行标记的测试
pytest -m "not slow"
```

### 生成覆盖率报告

```bash
# 后端
pytest tests/ --cov=app --cov-report=html

# 前端
npm run test:coverage
```

### 使用脚本

```bash
# 运行所有测试
./scripts/run-tests.sh

# 生成质量报告
./scripts/generate-reports.sh
```

## 📊 测试覆盖率

### 覆盖率目标

- **后端**: ≥ 80%
- **前端**: ≥ 70%
- **关键模块**: ≥ 90%

### 查看覆盖率

```bash
# 后端HTML报告
open backend/htmlcov/index.html

# 前端HTML报告
open frontend/coverage/index.html
```

### 覆盖率类型

1. **行覆盖率**: 执行的代码行数
2. **分支覆盖率**: 执行的分支数
3. **函数覆盖率**: 调用的函数数

### 提高覆盖率

1. 识别未覆盖的代码
2. 编写测试用例
3. 移除死代码
4. 使用覆盖率报告指导测试

## 🎯 测试最佳实践

### DO ✅

- 测试边界情况（空值、极值、null）
- 测试错误处理
- 使用描述性的测试名称
- 保持测试独立
- 使用fixtures共享数据
- Mock外部依赖

### DON'T ❌

- 不要测试实现细节
- 不要编写脆弱的测试
- 不要忽略失败的测试
- 不要编写过长的测试
- 不要依赖测试执行顺序

## 📚 参考资源

- [pytest文档](https://docs.pytest.org/)
- [Jest文档](https://jestjs.io/docs/getting-started)
- [Testing Library文档](https://testing-library.com/)
- [Hypothesis文档](https://hypothesis.readthedocs.io/)

---

**最后更新**: 2026-01-26
