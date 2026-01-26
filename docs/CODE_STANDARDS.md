# 代码规范指南

本文档定义了项目的代码编写规范和最佳实践。

## 📋 目录

- [通用规范](#通用规范)
- [Python代码规范](#python代码规范)
- [TypeScript/React代码规范](#typescriptreact代码规范)
- [提交规范](#提交规范)
- [代码审查规范](#代码审查规范)

## 🔧 通用规范

### 文件命名
- **Python**: 使用小写字母和下划线，如 `data_service.py`
- **TypeScript/React**: 使用PascalCase，如 `PositionAnalysis.tsx`
- **配置文件**: 使用小写和连字符，如 `code-quality.yml`

### 编码
- 所有文件使用 **UTF-8** 编码
- 使用 **LF** 作为行结束符（Unix风格）

### 注释
- 使用清晰、简洁的注释
- 解释"为什么"而不是"是什么"
- 保持注释与代码同步

## 🐍 Python代码规范

### 代码风格
遵循 [PEP 8](https://pep8.org/) 规范，使用以下工具自动检查：

- **Black**: 代码格式化（行长度88字符）
- **isort**: 导入排序
- **Flake8**: 代码风格检查

### 类型注解
- 所有公共函数必须有类型注解
- 使用 `typing` 模块的类型提示
- 复杂类型使用 `TypeAlias`

```python
from typing import List, Dict, Optional

def process_data(
    items: List[str],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, int]:
    """处理数据并返回结果"""
    pass
```

### 文档字符串
使用Google风格的文档字符串：

```python
def calculate_returns(
    prices: List[float],
    start_date: str,
    end_date: str
) -> float:
    """
    计算指定时间段的收益率。
    
    Args:
        prices: 价格列表
        start_date: 开始日期（YYYY-MM-DD格式）
        end_date: 结束日期（YYYY-MM-DD格式）
    
    Returns:
        收益率（小数形式，如0.15表示15%）
    
    Raises:
        ValueError: 当日期格式不正确时
    """
    pass
```

### 导入顺序
1. 标准库
2. 第三方库
3. 本地应用/库

使用 `isort` 自动排序。

### 错误处理
- 使用具体的异常类型
- 提供有意义的错误消息
- 记录错误日志

```python
from loguru import logger

try:
    result = process_data(data)
except ValueError as e:
    logger.error(f"数据验证失败: {e}")
    raise
except Exception as e:
    logger.exception("处理数据时发生未知错误")
    raise RuntimeError(f"数据处理失败: {e}") from e
```

### 测试
- 测试函数名以 `test_` 开头
- 使用描述性的测试名称
- 每个测试只验证一个行为
- 使用 fixtures 共享测试数据

```python
import pytest

def test_calculate_returns_with_valid_data():
    """测试使用有效数据计算收益率"""
    prices = [100.0, 105.0, 110.0]
    result = calculate_returns(prices, "2024-01-01", "2024-01-03")
    assert result == 0.10
```

## ⚛️ TypeScript/React代码规范

### 代码风格
- 使用 **ESLint** 和 **Prettier** 自动格式化
- 遵循 Next.js 和 React 最佳实践

### 类型定义
- 所有组件props必须有类型定义
- 使用 `interface` 定义对象类型
- 避免使用 `any`，使用 `unknown` 或具体类型

```typescript
interface PositionAnalysisProps {
  positionAnalysis: PositionData[];
  stockCodes: string[];
  taskId?: string;
}

export function PositionAnalysis({ 
  positionAnalysis, 
  stockCodes, 
  taskId 
}: PositionAnalysisProps) {
  // ...
}
```

### 组件规范
- 使用函数组件和Hooks
- 组件名使用PascalCase
- 文件名与组件名保持一致

```typescript
// ✅ 好的做法
export function StockChart({ data }: StockChartProps) {
  const [selectedDate, setSelectedDate] = useState<string>();
  
  return (
    <div>
      {/* ... */}
    </div>
  );
}

// ❌ 避免
export const stockChart = ({ data }) => {
  // ...
};
```

### Hooks规范
- 自定义Hook以 `use` 开头
- 在组件顶层调用Hooks
- 使用依赖数组避免不必要的重渲染

```typescript
function useStockData(stockCode: string) {
  const [data, setData] = useState<StockData | null>(null);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetchStockData(stockCode)
      .then(setData)
      .finally(() => setLoading(false));
  }, [stockCode]);
  
  return { data, loading };
}
```

### 状态管理
- 优先使用本地状态（useState）
- 共享状态使用Context或Zustand
- 避免过度使用全局状态

### 性能优化
- 使用 `React.memo` 优化组件重渲染
- 使用 `useMemo` 和 `useCallback` 优化计算
- 懒加载大型组件

```typescript
const ExpensiveComponent = React.memo(({ data }: Props) => {
  const processedData = useMemo(
    () => processLargeDataset(data),
    [data]
  );
  
  return <div>{/* ... */}</div>;
});
```

### 错误处理
- 使用错误边界捕获组件错误
- 提供用户友好的错误消息
- 记录错误到监控系统

```typescript
class ErrorBoundary extends React.Component {
  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('组件错误:', error, errorInfo);
    // 发送到错误监控服务
  }
  
  render() {
    if (this.state.hasError) {
      return <ErrorFallback />;
    }
    return this.props.children;
  }
}
```

## 📝 提交规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

### 提交格式
```
<type>(<scope>): <subject>

<body>

<footer>
```

### 类型（type）
- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 重构
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建/工具/依赖更新
- `ci`: CI/CD配置

### 示例
```
feat(backtest): 添加多策略回测支持

- 实现策略组合功能
- 添加策略权重配置
- 更新回测结果展示

Closes #123
```

## 🔍 代码审查规范

### 审查检查清单

#### 功能
- [ ] 代码实现了需求
- [ ] 边界情况已处理
- [ ] 错误处理完善

#### 代码质量
- [ ] 代码符合项目规范
- [ ] 没有代码重复
- [ ] 函数/类职责单一
- [ ] 命名清晰有意义

#### 测试
- [ ] 有适当的测试覆盖
- [ ] 测试用例清晰
- [ ] 所有测试通过

#### 文档
- [ ] 公共API有文档
- [ ] 复杂逻辑有注释
- [ ] README已更新（如需要）

#### 性能
- [ ] 没有明显的性能问题
- [ ] 数据库查询已优化
- [ ] 前端组件已优化

#### 安全
- [ ] 没有安全漏洞
- [ ] 敏感信息已保护
- [ ] 输入已验证

## 🛠️ 工具使用

### 自动格式化
```bash
# 后端
black app/
isort app/

# 前端
npm run format
```

### 代码检查
```bash
# 后端
flake8 app/
mypy app/

# 前端
npm run lint
npm run type-check
```

### 运行测试
```bash
# 后端
pytest tests/

# 前端
npm test
```

## 📚 参考资源

- [PEP 8 - Python代码风格指南](https://pep8.org/)
- [TypeScript官方文档](https://www.typescriptlang.org/docs/)
- [React最佳实践](https://react.dev/learn)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Google Python风格指南](https://google.github.io/styleguide/pyguide.html)

---

**最后更新**: 2026-01-26
