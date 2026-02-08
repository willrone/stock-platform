# Willrone 代码开发规范

**版本**: 1.0.0  
**生效日期**: 2026-02-08  
**适用范围**: 所有 Willrone 项目代码（后端 Python + 前端 TypeScript/React）

---

## 📋 目录

1. [通用规范](#通用规范)
2. [Python 后端规范](#python-后端规范)
3. [TypeScript/React 前端规范](#typescriptreact-前端规���)
4. [Git 提交规范](#git-提交规范)
5. [代码审查规范](#代码审查规范)
6. [测试规范](#测试规范)

---

## 通用规范

### 1.1 文件和目录结构

#### 强制规则
- ✅ **单一职责原则**: 每个文件/模块只负责一个功能领域
- ✅ **代码长度限制**:
  - 函数: ≤50 行（警戒线 100 行，禁止 >200 行）
  - 类: ≤300 行（警戒线 500 行，禁止 >800 行）
  - 文件: ≤500 行（警戒线 800 行，禁止 >1000 行）
- ✅ **模块化拆分**: 超过警戒线必须拆分为多个模块

#### 目录命名规范
```
backend/
├── app/
│   ├── api/v1/          # API 路由（按功能模块分组）
│   ├── services/        # 业务逻辑（按领域分组）
│   ├── models/          # 数据模型
│   ├── core/            # 核心工具和配置
│   └── tests/           # 测试文件

frontend/
├── src/
│   ├── app/             # Next.js 页面路由
│   ├── components/      # React 组件（按功能分组）
│   ├── hooks/           # 自定义 Hooks
│   ├── utils/           # 工具函数
│   └── types/           # TypeScript 类型定义
```

### 1.2 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 文件名（Python） | snake_case | `backtest_executor.py` |
| 文件名（TypeScript） | PascalCase（组件）/ camelCase（工具） | `TaskDetail.tsx`, `formatDate.ts` |
| 类名 | PascalCase | `BacktestExecutor`, `TaskManager` |
| 函数名 | snake_case（Python）/ camelCase（TS） | `execute_backtest()`, `executeBacktest()` |
| 变量名 | snake_case（Python）/ camelCase（TS） | `task_id`, `taskId` |
| 常量 | UPPER_SNAKE_CASE | `MAX_RETRY_COUNT`, `API_BASE_URL` |
| 私有方法 | 前缀 `_` | `_calculate_metrics()` |

### 1.3 注释规范

#### 强制要求
- ✅ **所有公共 API 必须有文档字符串**
- ✅ **复杂逻辑必须有行内注释**
- ✅ **TODO/FIXME 必须包含日期和负责人**

#### Python 文档字符串
```python
def execute_backtest(
    strategy_name: str,
    stock_codes: List[str],
    start_date: datetime,
    end_date: datetime,
) -> BacktestResult:
    """
    执行回测任务
    
    Args:
        strategy_name: 策略名称（如 'RSI', 'MA'）
        stock_codes: 股票代码列表
        start_date: 回测开始日期
        end_date: 回测结束日期
    
    Returns:
        BacktestResult: 回测结果对象，包含收益率、夏普比率等指标
    
    Raises:
        ValueError: 当日期范围无效时
        TaskError: 当回测执行失败时
    
    Example:
        >>> result = execute_backtest('RSI', ['000001.SZ'], date(2023,1,1), date(2024,1,1))
        >>> print(result.total_return)
        0.1523
    """
    pass
```

#### TypeScript JSDoc
```typescript
/**
 * 执行回测任务
 * 
 * @param strategyName - 策略名称（如 'RSI', 'MA'）
 * @param stockCodes - 股票代码列表
 * @param startDate - 回测开始日期
 * @param endDate - 回测结束日期
 * @returns 回测结果对象
 * @throws {Error} 当日期范围无效时
 * 
 * @example
 * ```ts
 * const result = await executeBacktest('RSI', ['000001.SZ'], new Date('2023-01-01'), new Date('2024-01-01'));
 * console.log(result.totalReturn);
 * ```
 */
async function executeBacktest(
  strategyName: string,
  stockCodes: string[],
  startDate: Date,
  endDate: Date
): Promise<BacktestResult> {
  // 实现
}
```

---

## 复杂度规范

### 2.1 圈复杂度 (Cyclomatic Complexity)

#### 定义
圈复杂度衡量代码的控制流复杂度，即代码中独立路径的数量。

#### 强制规则
- ✅ **函数圈复杂度 ≤10**（警戒线 15，禁止 >20）
- ✅ **类圈复杂度 ≤50**（警戒线 80，禁止 >100）

#### 计算方法
```
圈复杂度 = 判断节点数 + 1

判断节点包括：
- if/elif/else
- for/while 循环
- and/or 逻辑运算符
- try/except
- 三元运算符
- match/case (Python 3.10+)
```

#### 示例

```python
# ❌ 坏的示例：圈复杂度 = 8（过高）
def calculate_discount(user_type: str, amount: float, is_vip: bool, 
                       has_coupon: bool, order_count: int) -> float:
    discount = 0.0
    
    if user_type == "new":  # +1
        discount = 0.1
    elif user_type == "regular":  # +1
        discount = 0.05
    elif user_type == "premium":  # +1
        discount = 0.15
    
    if is_vip:  # +1
        discount += 0.05
    
    if has_coupon:  # +1
        discount += 0.1
    
    if order_count > 10:  # +1
        discount += 0.05
    elif order_count > 5:  # +1
        discount += 0.02
    
    return amount * (1 - discount)

# ✅ 好的示例：拆分为多个函数，降低复杂度
def get_base_discount(user_type: str) -> float:
    """获取基础折扣"""
    discount_map = {
        "new": 0.1,
        "regular": 0.05,
        "premium": 0.15,
    }
    return discount_map.get(user_type, 0.0)

def get_loyalty_discount(is_vip: bool, order_count: int) -> float:
    """获取忠诚度折扣"""
    discount = 0.05 if is_vip else 0.0
    
    if order_count > 10:
        discount += 0.05
    elif order_count > 5:
        discount += 0.02
    
    return discount

def calculate_discount(user_type: str, amount: float, is_vip: bool,
                       has_coupon: bool, order_count: int) -> float:
    """计算最终折扣（圈复杂度 = 2）"""
    discount = get_base_discount(user_type)
    discount += get_loyalty_discount(is_vip, order_count)
    
    if has_coupon:  # +1
        discount += 0.1
    
    return amount * (1 - discount)
```

#### 降低圈复杂度的方法
1. **提取方法**: 将复杂逻辑拆分为多个小函数
2. **使用字典/映射**: 替代多个 if-elif
3. **策略模式**: 使用多态替代条件判断
4. **提前返回**: 减少嵌套层级
5. **使用卫语句**: 先处理异常情况

#### 检测工具
- Python: `radon` - `pip install radon`
  ```bash
  radon cc backend/app/services/ -a -nb
  ```
- TypeScript: `eslint-plugin-complexity`

---

### 2.2 时间复杂度 (Time Complexity)

#### 定义
算法执行时间随输入规模增长的趋势。

#### 强制规则
- ✅ **核心算法必须标注时间复杂度**
- ✅ **避免不必要的 O(n²) 及以上复杂度**
- ✅ **大数据集操作优先使用 O(n) 或 O(n log n)**

#### 常见复杂度等级

| 复杂度 | 名称 | 示例 | 性能 |
|--------|------|------|------|
| O(1) | 常数 | 数组索引、哈希表查找 | ⭐⭐⭐⭐⭐ |
| O(log n) | 对数 | 二分查找 | ⭐⭐⭐⭐ |
| O(n) | 线性 | 遍历数组 | ⭐⭐⭐ |
| O(n log n) | 线性对数 | 快速排序、归并排序 | ⭐⭐ |
| O(n²) | 平方 | 双层循环 | ⭐ |
| O(2ⁿ) | 指数 | 递归斐波那契 | ❌ |

#### 示例

```python
# ❌ 坏的示例：O(n²) - 双层循环
def find_duplicates(data: List[str]) -> List[str]:
    """查找重复项（时间复杂度 O(n²)）"""
    duplicates = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] == data[j] and data[i] not in duplicates:
                duplicates.append(data[i])
    return duplicates

# ✅ ��的示例：O(n) - 使用哈希表
def find_duplicates(data: List[str]) -> List[str]:
    """
    查找重复项
    
    时间复杂度: O(n)
    空间复杂度: O(n)
    """
    seen = set()
    duplicates = set()
    
    for item in data:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    
    return list(duplicates)

# ✅ 好的示例：O(n) - 使用 pandas 向量化操作
def calculate_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """
    计算移动平均
    
    时间复杂度: O(n)
    空间复杂度: O(n)
    """
    return prices.rolling(window=window).mean()

# ❌ 坏的示例：O(n²) - 逐行计算
def calculate_moving_average_slow(prices: List[float], window: int) -> List[float]:
    """时间复杂度 O(n²)"""
    result = []
    for i in range(len(prices)):
        window_data = prices[max(0, i-window+1):i+1]
        result.append(sum(window_data) / len(window_data))
    return result
```

#### 优化策略
1. **使用合适的数据结构**:
   - 查找频繁 → 使用 dict/set (O(1))
   - 有序数据 → 使用二分查找 (O(log n))
   - 大数据集 → 使用 numpy/pandas 向量化

2. **避免重复计算**:
   - 使用缓存 (`@lru_cache`)
   - 使用动态规划

3. **批量操作**:
   - 数据库批量插入/更新
   - 向量化计算代替循环

---

### 2.3 空间复杂度 (Space Complexity)

#### 定义
算法执行过程中所需的额外内存空间随输入规模增长的趋势。

#### 强制规则
- ✅ **核心算法必须标注空间复杂度**
- ✅ **大数据集处理优先使用生成器/迭代器**
- ✅ **避免不必要的数据复制**

#### 示例

```python
# ❌ 坏的示例：O(n) 空间 - 创建新列表
def process_large_file(file_path: str) -> List[dict]:
    """
    处理大文件
    
    时间复杂度: O(n)
    空间复杂度: O(n) - 将所有数据加载到内存
    """
    with open(file_path) as f:
        data = [json.loads(line) for line in f]  # 全部加载到内存
    
    return [process_item(item) for item in data]

# ✅ 好的示例：O(1) 空间 - 使用生成器
def process_large_file(file_path: str) -> Iterator[dict]:
    """
    处理大文件
    
    时间复杂度: O(n)
    空间复杂度: O(1) - 逐行处理，不占用额外内存
    """
    with open(file_path) as f:
        for line in f:
            item = json.loads(line)
            yield process_item(item)

# 使用示例
for result in process_large_file("large_data.jsonl"):
    save_to_db(result)

# ✅ 好的示例：避免数据复制
def filter_stock_data(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """
    过滤股票数据
    
    时间复杂度: O(n)
    空间复杂度: O(1) - 使用视图，不复制数据
    """
    # 使用 query 返回视图，不复制数据
    return df.query(condition, inplace=False)

# ❌ 坏的示例：不必要的数据复制
def filter_stock_data_slow(df: pd.DataFrame, min_price: float) -> pd.DataFrame:
    """空间复杂度 O(n) - 创建多个副本"""
    df_copy = df.copy()  # 第一次复制
    filtered = df_copy[df_copy['price'] > min_price]  # 第二次复制
    return filtered.copy()  # 第三次复制
```

#### 优化策略
1. **使用生成器**:
   ```python
   # ✅ 好的
   def read_large_file(path: str) -> Iterator[str]:
       with open(path) as f:
           for line in f:
               yield line.strip()
   
   # ❌ 坏的
   def read_large_file(path: str) -> List[str]:
       with open(path) as f:
           return [line.strip() for line in f]
   ```

2. **就地修改**:
   ```python
   # ✅ 好的：就地排序
   data.sort()  # O(1) 空间
   
   # ❌ 坏的：创建新列表
   sorted_data = sorted(data)  # O(n) 空间
   ```

3. **使用视图而非副本**:
   ```python
   # ✅ 好的：使用切片视图
   subset = df.iloc[100:200]  # 视图，O(1) 空间
   
   # ❌ 坏的：复制数据
   subset = df.iloc[100:200].copy()  # 副本，O(n) 空间
   ```

4. **分块处理**:
   ```python
   # ✅ 好的：分块读取大文件
   def process_large_csv(file_path: str, chunk_size: int = 10000):
       """
       分块处理大型 CSV
       
       时间复杂度: O(n)
       空间复杂度: O(chunk_size) - 固定内存占用
       """
       for chunk in pd.read_csv(file_path, chunksize=chunk_size):
           process_chunk(chunk)
   ```

---

### 2.4 复杂度标注规范

#### 强制规则
- ✅ **所有核心算法函数必须在文档字符串中标注复杂度**
- ✅ **复杂度分析必须包含最坏情况**
- ✅ **如果时间和空间复杂度有权衡，必须说明**

#### 标注模板

```python
def algorithm_name(data: List[int]) -> List[int]:
    """
    算法功能描述
    
    Args:
        data: 输入数据描述
    
    Returns:
        返回值描述
    
    Complexity:
        Time: O(n log n) - 快速排序的平均情况
        Space: O(log n) - 递归调用栈深度
        
        最坏情况:
        Time: O(n²) - 当数据已排序时
        Space: O(n) - 递归调用栈最大深度
    
    Note:
        使用快速排序而不是归并排序，因为：
        1. 平均性能更好
        2. 空间复杂度更低（O(log n) vs O(n)）
        3. 缓存友好
    """
    pass
```

#### TypeScript 标注

```typescript
/**
 * 算法功能描述
 * 
 * @param data - 输入数据描述
 * @returns 返回值描述
 * 
 * @complexity
 * Time: O(n log n) - 快速排序的平均情况
 * Space: O(log n) - 递归调用栈深度
 * 
 * @remarks
 * 最坏情况下时间复杂度为 O(n²)
 */
function algorithmName(data: number[]): number[] {
  // 实现
}
```

---

## Python 后端规范

### 2.1 代码风格

#### 强制规则
- ✅ **遵循 PEP 8 规范**
- ✅ **使用 Black 格式化代码**（行宽 100）
- ✅ **使用 isort 排序导入**
- ✅ **使用 mypy 进行类型检查**

#### 导入顺序
```python
# 1. 标准库
import os
from datetime import datetime
from typing import List, Dict, Optional

# 2. 第三方库
import pandas as pd
import numpy as np
from fastapi import APIRouter

# 3. 本地模块
from app.core.config import settings
from app.models.task import Task
from app.services.backtest import BacktestExecutor
```

### 2.2 类型注解

#### 强制规则
- ✅ **所有函数参数和返回值必须有类型注解**
- ✅ **复杂类型使用 TypedDict 或 Pydantic 模型**

```python
from typing import List, Dict, Optional, TypedDict
from pydantic import BaseModel

# ✅ 好的示例
class BacktestConfig(BaseModel):
    strategy_name: str
    stock_codes: List[str]
    start_date: datetime
    end_date: datetime
    initial_capital: float = 1000000.0

def execute_backtest(config: BacktestConfig) -> Dict[str, float]:
    """执行回测"""
    pass

# ❌ 坏的示例
def execute_backtest(config):  # 缺少类型注解
    pass
```

### 2.3 错误处理

#### 强制规则
- ✅ **使用自定义异常类**
- ✅ **记录详细的错误日志**
- ✅ **不要捕获通用 Exception（除非重新抛出）**

```python
from app.core.error_handler import TaskError
from loguru import logger

# ✅ 好的示例
def load_stock_data(stock_code: str) -> pd.DataFrame:
    """加载股票数据"""
    try:
        data = pd.read_parquet(f"data/{stock_code}.parquet")
        if data.empty:
            raise TaskError(f"股票 {stock_code} 数据为空")
        return data
    except FileNotFoundError:
        logger.error(f"股票数据文件不存在: {stock_code}")
        raise TaskError(f"找不到股票 {stock_code} 的数据文件")
    except Exception as e:
        logger.exception(f"加载股票数据失败: {stock_code}")
        raise TaskError(f"加载数据失败: {str(e)}") from e

# ❌ 坏的示例
def load_stock_data(stock_code):
    try:
        data = pd.read_parquet(f"data/{stock_code}.parquet")
        return data
    except:  # 捕获所有异常且不记录
        return None
```

### 2.4 数据库操作

#### 强制规则
- ✅ **使用 SQLAlchemy ORM**
- ✅ **使用异步会话**
- ✅ **使用上下文管理器管理事务**

```python
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.task import Task

# ✅ 好的示例
async def create_task(db: AsyncSession, task_data: dict) -> Task:
    """创建任务"""
    task = Task(**task_data)
    db.add(task)
    await db.commit()
    await db.refresh(task)
    return task

async def get_task(db: AsyncSession, task_id: str) -> Optional[Task]:
    """获取任务"""
    result = await db.execute(
        select(Task).where(Task.task_id == task_id)
    )
    return result.scalar_one_or_none()
```

### 2.5 性能优化

#### 强制规则
- ✅ **避免 N+1 查询问题**
- ✅ **使用批量操作代替循环**
- ✅ **大数据集使用生成器**
- ✅ **CPU 密集型任务使用多进程**

```python
# ✅ 好的示例：批量操作
def calculate_indicators_batch(data: pd.DataFrame) -> pd.DataFrame:
    """批量计算技术指标"""
    data['ma_5'] = data['close'].rolling(5).mean()
    data['ma_10'] = data['close'].rolling(10).mean()
    data['rsi'] = calculate_rsi(data['close'], 14)
    return data

# ❌ 坏的示例：逐行操作
def calculate_indicators_loop(data: pd.DataFrame) -> pd.DataFrame:
    for i in range(len(data)):
        data.loc[i, 'ma_5'] = data['close'].iloc[max(0, i-4):i+1].mean()
    return data
```

---

## TypeScript/React 前端规范

### 3.1 代码风格

#### 强制规则
- ✅ **使用 ESLint + Prettier 格式化**
- ✅ **使用 TypeScript 严格模式**
- ✅ **组件使用函数式组件 + Hooks**

#### tsconfig.json 配置
```json
{
  "compilerOptions": {
    "strict": true,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true
  }
}
```

### 3.2 组件规范

#### 强制规则
- ✅ **组件文件名使用 PascalCase**
- ✅ **一个文件只导出一个主组件**
- ✅ **Props 必须定义接口**
- ✅ **使用自定义 Hooks 提取业务逻辑**

```typescript
// ✅ 好的示例
interface TaskDetailProps {
  taskId: string;
  onDelete?: (taskId: string) => void;
}

export default function TaskDetail({ taskId, onDelete }: TaskDetailProps) {
  const { task, loading, error } = useTaskDetail(taskId);
  const { handleDelete } = useTaskActions(taskId, onDelete);
  
  if (loading) return <LoadingSpinner />;
  if (error) return <ErrorMessage error={error} />;
  if (!task) return <NotFound />;
  
  return (
    <div className="task-detail">
      <TaskHeader task={task} onDelete={handleDelete} />
      <TaskContent task={task} />
    </div>
  );
}

// ❌ 坏的示例
export default function TaskDetail(props: any) {  // any 类型
  const [task, setTask] = useState();  // 缺少类型
  
  useEffect(() => {
    // 业务逻辑直接写在组件里
    fetch(`/api/tasks/${props.taskId}`)
      .then(res => res.json())
      .then(data => setTask(data));
  }, []);
  
  return <div>{task?.name}</div>;
}
```

### 3.3 Hooks 规范

#### 强制规则
- ✅ **自定义 Hook 必须以 `use` 开头**
- ✅ **提取可复用的业务逻辑到 Hooks**
- ✅ **使用 useMemo/useCallback 优化性能**

```typescript
// ✅ 好的示例
export function useTaskDetail(taskId: string) {
  const [task, setTask] = useState<Task | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  
  useEffect(() => {
    let cancelled = false;
    
    async function fetchTask() {
      try {
        setLoading(true);
        const response = await fetch(`/api/v1/tasks/${taskId}`);
        if (!response.ok) throw new Error('Failed to fetch task');
        const data = await response.json();
        if (!cancelled) {
          setTask(data);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err as Error);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }
    
    fetchTask();
    return () => { cancelled = true; };
  }, [taskId]);
  
  return { task, loading, error };
}
```

### 3.4 状态管理

#### 强制规则
- ✅ **优先使用 React Context + Hooks**
- ✅ **复杂状态使用 useReducer**
- ✅ **避免 prop drilling（超过 3 层使用 Context）**

```typescript
// ✅ 好的示例：使用 Context
interface TaskContextValue {
  tasks: Task[];
  loading: boolean;
  createTask: (data: CreateTaskData) => Promise<void>;
  deleteTask: (taskId: string) => Promise<void>;
}

const TaskContext = createContext<TaskContextValue | undefined>(undefined);

export function TaskProvider({ children }: { children: React.ReactNode }) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [loading, setLoading] = useState(false);
  
  const createTask = useCallback(async (data: CreateTaskData) => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/tasks', {
        method: 'POST',
        body: JSON.stringify(data),
      });
      const newTask = await response.json();
      setTasks(prev => [...prev, newTask]);
    } finally {
      setLoading(false);
    }
  }, []);
  
  const value = useMemo(
    () => ({ tasks, loading, createTask, deleteTask }),
    [tasks, loading, createTask, deleteTask]
  );
  
  return <TaskContext.Provider value={value}>{children}</TaskContext.Provider>;
}

export function useTaskContext() {
  const context = useContext(TaskContext);
  if (!context) {
    throw new Error('useTaskContext must be used within TaskProvider');
  }
  return context;
}
```

---

## Git 提交规范

### 4.1 提交消息格式

#### 强制规则
- ✅ **使用 Conventional Commits 规范**
- ✅ **提交消息必须包含类型和描述**
- ✅ **破坏性变更必须标注 `BREAKING CHANGE`**

#### 提交类型
```
feat:     新功能
fix:      Bug 修复
refactor: 代码重构（不改变功能）
perf:     性能优化
style:    代码格式调整（不影响逻辑）
docs:     文档更新
test:     测试相关
chore:    构建/工具链相关
```

#### 示例
```bash
# ✅ 好的示例
git commit -m "feat: 添加 RSI 策略回测功能"
git commit -m "fix: 修复回测结果计算错误"
git commit -m "refactor: 拆分 backtest_executor.py 为多个模块"
git commit -m "perf: 优化数据加载性能，提升 57%"
git commit -m "docs: 更新 API 文档"

# 多行提交消息
git commit -m "feat: 添加任务重建功能

- 支持从已完成任务复制配置
- 自动填充表单字段
- 添加 URL 参数支持

Closes #123"

# ❌ 坏的示例
git commit -m "update"
git commit -m "fix bug"
git commit -m "修改代码"
```

### 4.2 分支管理

#### 强制规则
- ✅ **主分支**: `main`（受保护，只能通过 PR 合并）
- ✅ **功能分支**: `feature/<功能名>`
- ✅ **修复分支**: `fix/<问题描述>`
- ✅ **重构分支**: `refactor/<模块名>`

```bash
# ✅ 好的示例
git checkout -b feature/task-rebuild
git checkout -b fix/backtest-calculation-error
git checkout -b refactor/backtest-executor

# ❌ 坏的示例
git checkout -b dev
git checkout -b test
git checkout -b temp
```

### 4.3 代码合并

#### 强制规则
- ✅ **合并前必须通过所有测试**
- ✅ **合并前必须解决所有冲突**
- ✅ **使用 `--no-ff` 保留分支历史**

```bash
# ✅ 好的示例
git checkout main
git merge feature/task-rebuild --no-ff -m "feat: 合并任务重建功能"

# 或使用 rebase 保持线性历史
git checkout feature/task-rebuild
git rebase main
git checkout main
git merge feature/task-rebuild --ff-only
```

---

## 代码审查规范

### 5.1 审查清单

#### 功能性
- [ ] 代码实现了需求的所有功能
- [ ] 边界条件和异常情况都有处理
- [ ] 没有明显的逻辑错误

#### 可读性
- [ ] 变量和函数命名清晰易懂
- [ ] 复杂逻辑有注释说明
- [ ] 代码结构清晰，易于理解

#### 可维护性
- [ ] 遵循单一职责原则
- [ ] 没有重复代码
- [ ] 函数/类长度符合规范

#### 性能
- [ ] 没有明显的性能问题
- [ ] 数据库查询已优化
- [ ] 大数据集使用了合适的数据结构

#### 安全性
- [ ] 输入验证完整
- [ ] 没有 SQL 注入风险
- [ ] 敏感信息没有硬编码

#### 测试
- [ ] 关键功能有单元测试
- [ ] 测试覆盖率 ≥80%
- [ ] 所有测试通过

---

## 测试规范

### 6.1 测试覆盖率

#### 强制规则
- ✅ **核心业务逻辑测试覆盖率 ≥80%**
- ✅ **API 端点必须有集成测试**
- ✅ **关键组件必须有单元测试**

### 6.2 Python 测试

```python
import pytest
from app.services.backtest import BacktestExecutor

class TestBacktestExecutor:
    """回测执行器测试"""
    
    @pytest.fixture
    def executor(self):
        """创建测试用的执行器实例"""
        return BacktestExecutor()
    
    def test_execute_backtest_success(self, executor):
        """测试回测执行成功"""
        config = {
            'strategy_name': 'RSI',
            'stock_codes': ['000001.SZ'],
            'start_date': '2023-01-01',
            'end_date': '2024-01-01',
        }
        result = executor.execute(config)
        
        assert result is not None
        assert result['total_return'] > 0
        assert result['sharpe_ratio'] > 0
    
    def test_execute_backtest_invalid_date(self, executor):
        """测试无效日期范围"""
        config = {
            'strategy_name': 'RSI',
            'stock_codes': ['000001.SZ'],
            'start_date': '2024-01-01',
            'end_date': '2023-01-01',  # 结束日期早于开始日期
        }
        
        with pytest.raises(ValueError, match="日期范围无效"):
            executor.execute(config)
```

### 6.3 TypeScript 测试

```typescript
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import TaskDetail from './TaskDetail';

describe('TaskDetail', () => {
  it('should render task details', async () => {
    const mockTask = {
      task_id: '123',
      task_name: 'Test Task',
      status: 'completed',
    };
    
    render(<TaskDetail taskId="123" />);
    
    await waitFor(() => {
      expect(screen.getByText('Test Task')).toBeInTheDocument();
      expect(screen.getByText('completed')).toBeInTheDocument();
    });
  });
  
  it('should handle delete action', async () => {
    const onDelete = jest.fn();
    render(<TaskDetail taskId="123" onDelete={onDelete} />);
    
    const deleteButton = screen.getByRole('button', { name: /delete/i });
    await userEvent.click(deleteButton);
    
    expect(onDelete).toHaveBeenCalledWith('123');
  });
});
```

---

## 附录：工具配置

### A.1 Python 工具链

#### pyproject.toml
```toml
[tool.black]
line-length = 100
target-version = ['py312']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.12"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

### A.2 前端工具链

#### .eslintrc.json
```json
{
  "extends": [
    "next/core-web-vitals",
    "plugin:@typescript-eslint/recommended"
  ],
  "rules": {
    "@typescript-eslint/no-explicit-any": "error",
    "@typescript-eslint/no-unused-vars": "error",
    "react-hooks/rules-of-hooks": "error",
    "react-hooks/exhaustive-deps": "warn"
  }
}
```

#### .prettierrc
```json
{
  "semi": true,
  "trailingComma": "es5",
  "singleQuote": true,
  "printWidth": 100,
  "tabWidth": 2
}
```

---

## 📝 规范更新记录

| 版本 | 日期 | 变更内容 |
|------|------|---------|
| 1.0.0 | 2026-02-08 | 初始版本，基于重构经验制定 |

---

**本规范为强制执行规范，所有代码提交前必须通过规范检查。**
