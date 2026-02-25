# 编码规范文档 (CODING_STANDARDS)

> 版本：1.0.0 | 最后更新：2026-02-25  
> 适用项目：stock-platform（Python FastAPI + Next.js TypeScript）  
> **规范是强制要求，不是建议。违反规范的代码不得合并。**

---

## 目录

1. [Python 后端规范](#一python-后端规范)
2. [TypeScript/React 前端规范](#二typescriptreact-前端规范)
3. [通用规范](#三通用规范)
4. [工具配置](#四工具配置)
5. [代码审查清单](#五代码审查清单)

---

## 一、Python 后端规范

### 1. 代码长度限制（硬性规则）

| 单元 | 限制 | 说明 |
|------|------|------|
| 函数/方法 | ≤ 50 行 | 不含空行和注释 |
| 类 | ≤ 300 行 | 含所有方法 |
| 文件/模块 | ≤ 500 行 | 超出必须拆分 |
| 行长度 | ≤ 88 字符 | 与 black 配置一致 |

**超出限制时的处理方式：**
- 函数过长 → 提取子函数（Extract Function）
- 类过长 → 拆分为多个类或使用组合模式
- 文件过长 → 按职责拆分模块

### 2. 复杂度限制

| 指标 | 限制 | 工具 |
|------|------|------|
| 函数圈复杂度 | ≤ 10 | flake8-complexity |
| 类复杂度 | ≤ 50 | radon |
| 嵌套深度 | ≤ 4 层 | 人工审查 |
| 函数参数数量 | ≤ 5 个 | flake8 |

**参数超过 5 个时，使用 dataclass / Pydantic model 封装：**

```python
# ❌ 错误：参数过多
def calculate_signal(
    symbol: str,
    start_date: str,
    end_date: str,
    window_size: int,
    threshold: float,
    use_volume: bool,
) -> float:
    ...

# ✅ 正确：用 Pydantic model 封装
from pydantic import BaseModel

class SignalConfig(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    window_size: int
    threshold: float
    use_volume: bool

def calculate_signal(config: SignalConfig) -> float:
    ...
```

**减少嵌套深度 — 使用 Guard Clause（早返回）：**

```python
# ❌ 错误：深层嵌套
def process_stock_data(data: list[dict]) -> list[dict]:
    result = []
    if data:
        for item in data:
            if item.get("close"):
                if item["close"] > 0:
                    if item.get("volume"):
                        result.append(item)
    return result

# ✅ 正确：Guard Clause 早返回
def process_stock_data(stock_records: list[dict]) -> list[dict]:
    if not stock_records:
        return []

    valid_records = []
    for record in stock_records:
        if not record.get("close"):
            continue
        if record["close"] <= 0:
            continue
        if not record.get("volume"):
            continue
        valid_records.append(record)

    return valid_records
```

### 3. 命名规范（PEP 8 + Clean Code）

#### 3.1 基本规则

| 类型 | 风格 | 示例 |
|------|------|------|
| 变量 | snake_case，名词 | `closing_price`, `trade_volume` |
| 函数/方法 | snake_case，动词开头 | `get_stock_price()`, `calculate_rsi()` |
| 类 | CamelCase，名词 | `StockPredictor`, `BacktestEngine` |
| 常量 | UPPER_SNAKE_CASE | `MAX_RETRY_COUNT`, `DEFAULT_WINDOW_SIZE` |
| 私有成员 | `_` 前缀 | `_cache`, `_validate_input()` |
| 布尔变量 | `is_/has_/can_/should_` 前缀 | `is_valid`, `has_data`, `should_retry` |

#### 3.2 禁止使用的命名

```python
# ❌ 禁止：无意义名称
x, y, z
tmp, temp
data, info, obj
result, res, ret
val, value
i, j, k（循环变量除外）

# ✅ 正确：描述性名称
closing_price
temporary_buffer
stock_daily_records
calculation_result
rsi_value
row_index  # 循环变量也要有意义
```

#### 3.3 允许的领域缩写

以下是股票/金融领域通用缩写，允许使用：

```python
# ✅ 允许的领域缩写
rsi    # Relative Strength Index
macd   # Moving Average Convergence Divergence
atr    # Average True Range
ema    # Exponential Moving Average
sma    # Simple Moving Average
boll   # Bollinger Bands
kdj    # KDJ Indicator
vwap   # Volume Weighted Average Price
pe     # Price-to-Earnings Ratio
pb     # Price-to-Book Ratio
```

#### 3.4 函数命名动词前缀规范

```python
# 查询/获取数据
get_stock_price()
fetch_historical_data()
load_model_weights()
read_config()

# 计算/处理
calculate_rsi()
compute_returns()
process_raw_data()
transform_features()

# 验证
validate_date_range()
check_data_integrity()
verify_model_output()

# 创建/构建
create_backtest_task()
build_feature_matrix()
generate_signals()

# 保存/更新
save_model()
update_task_status()
store_prediction_result()
```

### 4. 类型注解（强制）

**所有公共函数的参数和返回值必须有类型注解。**

```python
# ❌ 错误：缺少类型注解
def calculate_moving_average(prices, window):
    return sum(prices[-window:]) / window

# ✅ 正确：完整类型注解
from typing import Optional

def calculate_moving_average(
    prices: list[float],
    window: int,
) -> float:
    return sum(prices[-window:]) / window
```

#### 4.1 类型注解规则

```python
# ✅ Python 3.9+ 使用内置集合类型（不用 typing 模块）
prices: list[float]
stock_map: dict[str, float]
coordinates: tuple[int, int]
unique_symbols: set[str]

# ✅ 使用 Optional 而非 Union[X, None]
from typing import Optional
closing_price: Optional[float] = None  # ✅
closing_price: float | None = None     # ✅ Python 3.10+

# ❌ 避免
from typing import List, Dict, Tuple  # 旧式写法

# ✅ 复杂类型用 TypeAlias
from typing import TypeAlias
PriceHistory: TypeAlias = list[tuple[str, float]]
StockDataFrame: TypeAlias = dict[str, list[float]]

# ✅ 回调函数类型
from collections.abc import Callable
SignalHandler: TypeAlias = Callable[[str, float], bool]
```

#### 4.2 Pydantic Model 类型注解

```python
from pydantic import BaseModel, Field
from datetime import date
from typing import Optional

class StockPrediction(BaseModel):
    symbol: str = Field(..., description="股票代码", min_length=6, max_length=6)
    prediction_date: date = Field(..., description="预测日期")
    predicted_return: float = Field(..., description="预测收益率")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="置信度")
    model_version: Optional[str] = Field(None, description="模型版本")
```

### 5. 文档字符串（Docstring）

**所有公共函数、类、模块必须有 docstring，使用 Google 风格。**

#### 5.1 函数 Docstring

```python
def calculate_rsi(
    prices: list[float],
    period: int = 14,
) -> float:
    """计算相对强弱指数（RSI）。

    使用 Wilder 平滑方法计算 RSI，适用于日线数据。
    RSI 值范围为 0-100，通常 >70 为超买，<30 为超卖。

    Args:
        prices: 收盘价列表，按时间升序排列，至少需要 period+1 个数据点。
        period: 计算周期，默认 14 天。必须为正整数。

    Returns:
        RSI 值，范围 [0, 100]。数据不足时返回 50.0（中性值）。

    Raises:
        ValueError: 当 period <= 0 时抛出。
        TypeError: 当 prices 包含非数值类型时抛出。

    Example:
        >>> prices = [10.0, 11.0, 10.5, 12.0, 11.5, 13.0]
        >>> rsi = calculate_rsi(prices, period=5)
        >>> 0 <= rsi <= 100
        True
    """
    if period <= 0:
        raise ValueError(f"period 必须为正整数，当前值：{period}")

    if len(prices) < period + 1:
        return 50.0

    # ... 实现
```

#### 5.2 类 Docstring

```python
class BacktestEngine:
    """回测引擎，用于评估交易策略的历史表现。

    支持单股票和多股票组合回测，提供详细的绩效指标分析。
    回测基于日线数据，不考虑盘中价格变动。

    Attributes:
        initial_capital: 初始资金（元）。
        commission_rate: 手续费率，默认 0.0003（万三）。
        slippage_rate: 滑点率，默认 0.001。

    Example:
        >>> engine = BacktestEngine(initial_capital=1_000_000)
        >>> result = engine.run(strategy, start_date="2023-01-01", end_date="2023-12-31")
        >>> print(result.sharpe_ratio)
    """

    def __init__(
        self,
        initial_capital: float,
        commission_rate: float = 0.0003,
        slippage_rate: float = 0.001,
    ) -> None:
        ...
```

#### 5.3 模块 Docstring

```python
"""股票技术指标计算模块。

提供常用技术指标的计算函数，包括趋势类、震荡类和成交量类指标。
所有函数均支持 numpy 数组和 Python list 输入。

Typical usage example:
    from indicators import calculate_rsi, calculate_macd

    rsi_value = calculate_rsi(closing_prices, period=14)
    macd_line, signal_line, histogram = calculate_macd(closing_prices)
"""
