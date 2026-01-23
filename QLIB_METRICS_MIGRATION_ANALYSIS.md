---
name: 数据层qlib预计算优化
overview: 采用混合预计算模式（Qlib计算基础指标+Alpha158因子，Pandas计算技术指标），全部离线预计算后存储为Qlib格式，回测时直接读取，避免重复计算，大幅提升回测速度（预计50-200%）。前端数据管理页面提供一键触发按钮。
todos:
  - id: backend-offline-precompute-service
    content: 在后端 services 中设计并实现离线预计算服务（混合模式：Qlib计算基础指标+Alpha158因子，Pandas计算技术指标+基本面特征），全部预计算后存储为Qlib格式（含任务封装）
    status: pending
  - id: backend-precompute-api
    content: 在 api/v1 中新增触发预计算的 REST 接口并接入任务系统
    status: pending
  - id: frontend-data-page-button
    content: 在前端数据管理页面新增“离线生成 Qlib 指标/因子”按钮（无需参数配置，一键触发）
    status: pending
  - id: frontend-progress-display
    content: 复用任务/进度组件，在前端展示离线预计算任务进度和结果
    status: pending
  - id: backtest-training-integration
    content: 调整回测与训练流程，优先从预计算结果直接读取所有指标（不再现场计算），大幅提升回测速度（预计50-200%），保留向后兼容的 fallback 模式
    status: pending
  - id: indicator-registry
    content: 实现指标注册机制，支持动态注册新指标
    status: pending
  - id: version-management
    content: 实现指标版本管理和增量更新机制
    status: pending
  - id: qlib-alpha158-integration
    content: 优化Alpha158因子计算，直接使用Qlib内置Alpha158 handler获得158个标准因子（替代当前32个因子的简化实现）
    status: pending
  - id: data-format-conversion
    content: 实现Parquet到Qlib格式的转换工具（MultiIndex构建、列名映射、数据类型优化）
    status: pending
  - id: qlib-storage-structure
    content: 设计并实现Qlib数据存储结构（目录组织、文件格式、索引机制）
    status: pending
  - id: incremental-update
    content: 实现增量更新机制（检测数据变化、增量计算、数据合并）
    status: pending
  - id: data-validation
    content: 实现预计算数据验证机制（完整性检查、质量检查、一致性检查）
    status: pending
  - id: error-handling
    content: 实现错误处理和容错机制（单股票失败处理、任务中断重试、数据损坏恢复）
    status: pending
  - id: performance-optimization
    content: 实现性能优化（分批处理策略、并行计算、内存管理）
    status: pending
  - id: monitoring-logging
    content: 实现监控和日志（预计算进度监控、数据质量监控、性能监控）
    status: pending
  - id: testing-validation
    content: 实现测试和验证（数据准确性验证、性能对比测试、兼容性测试）
    status: pending
  - id: deployment-ops
    content: 实现部署和运维（任务调度、数据备份、数据清理）
    status: pending
isProject: false
---

# 数据层 Qlib 预计算与前端触发方案

## 目标

- **后端数据层**: 采用**混合预计算模式**（Qlib + Pandas），提前离线计算好所有指标和因子：
  - **Qlib计算**：基础统计指标（Mean, Std, Corr等）和Alpha158因子（158个标准因子），发挥批量计算性能优势
  - **Pandas计算**：技术指标（RSI, MACD, 布林带等）和基本面特征，保持灵活性和成熟度
  - 全部写入 Qlib 兼容的数据目录（如 `QLIB_DATA_PATH`），供回测/训练直接读取
  - **核心优势**：回测时不再重复计算，直接读取预计算结果，**回测速度大幅提升（预计50-200%）**
- **前端数据管理页面**: 增加一个“离线生成 Qlib 指标/因子”的按钮，调用后端任务接口，触发后台离线作业，并展示执行进度/结果。

## 后端改造方案

- **1. 新增离线预计算服务**
- 位置建议: `backend/app/services/data/` 或 `backend/app/services/qlib/` 新增模块，例如：
- `offline_factor_precompute.py`（名称示例）
- 职责：
- **默认参数**（无需用户选择）：
- 股票范围：**全市场所有股票**（从 Parquet 目录或数据服务获取所有可用股票列表）
- 日期范围：**所有可用时间**（从数据源自动检测最早和最晚日期）
- 指标/因子类型：**所有需要用到的指标和因子**（包括基础技术指标 MA/RSI/MACD 等，以及 Alpha158 因子）
- 调用现有数据服务：
- 复用 `SimpleDataService`、`StockDataLoader`、`parquet_manager` 等，从 Parquet 或远程数据源批量拉取所有股票的 OHLCV 数据。
- 计算特征（混合模式：Qlib + Pandas）：
  - **基础统计指标（使用Qlib计算）**：
    - 使用Qlib表达式引擎：Mean, Std, Max, Min, Quantile, Corr等
    - 价格变化率、移动平均、相关性等
    - **优势**：多股票批量计算性能提升50-100%
  - **Alpha158因子（使用Qlib内置Alpha158）**：
    - 代码中已经导入了 `from qlib.contrib.data.handler import Alpha158` 和 `from qlib.contrib.data.loader import Alpha158DL`
    - 直接使用 `Alpha158` handler 或通过 `Alpha158DL.get_feature_config()` 获取因子表达式，然后使用Qlib的表达式引擎计算，获得完整的158个标准因子
    - 当前代码在 `Alpha158Calculator` 中已经获取了158个因子的配置（`self.alpha_fields` 和 `self.alpha_names`），但实际计算时使用的是简化版本（32个因子）
    - **优势**：因子计算性能提升100-200%，获得完整的158个标准因子
  - **技术指标（使用Pandas计算）**：
    - 复用 `TechnicalIndicatorCalculator` 计算所有技术指标（MA/RSI/MACD/布林带/KDJ/ATR/VWAP/OBV等）
    - **原因**：Qlib不直接支持这些技术指标，需要用表达式实现，工作量大且可能性能不如pandas
    - **优势**：代码成熟、灵活、单股票场景性能好
  - **基本面特征（使用Pandas计算）**：
    - 复用 `FeatureCalculator` 计算基本面特征
- 将结果转换为 Qlib 标准格式（MultiIndex: (instrument, date)）并写入 `settings.QLIB_DATA_PATH` 下的 qlib 数据目录（bin 格式或 Qlib 可直接读取的结构），与 `EnhancedQlibDataProvider` / `UnifiedQlibTrainingEngine` 保持兼容。

- **2. 使用任务系统异步执行**
- 利用现有任务模块 `backend/app/services/tasks/`：
- 在 `task_manager.py` / `task_execution_engine.py` 挂一个新任务类型，如 `QLIB_OFFLINE_PRECOMPUTE`。
- 任务 payload 无需参数（或仅包含任务名称等元信息），因为所有参数都是默认的（全市场、全时间、所有指标）。
- 在任务执行过程中：
- 自动扫描数据源获取所有股票列表和日期范围。
- 分批处理股票和日期，避免一次性占满内存。
- 定期更新任务进度（已处理股票数/总股票数、已处理日期段/总日期段），写入任务表，并通过现有 WebSocket/监控通道推送进度（可参考回测任务进度实现）。

- **3. 新增 API 接口**
- 位置: `backend/app/api/v1/data.py` 或新增 `backend/app/api/v1/qlib.py` 下的接口：
- `POST /api/v1/data/qlib/precompute`：创建离线预计算任务，返回 `task_id`。
- **无需请求参数**（或仅包含可选的 `task_name` 用于标识任务）
- 自动使用默认配置：全市场所有股票、所有可用时间、所有指标和因子
- （可选）`GET /api/v1/data/qlib/precompute/status/{task_id}`：查询进度（也可以复用通用任务查询接口）。

## 前端改造方案

- **1. 确认数据管理页面位置**
- 主要页面: `frontend/src/app/data/page.tsx`。
- 相关组件: `frontend/src/components/data/EnhancedDataFileTable.tsx`（如果已经用于展示数据文件）。

- **2. 新增服务封装**
- 在 `frontend/src/services/dataService.ts` 中新增：
- `triggerQlibPrecompute()`：`POST /api/v1/data/qlib/precompute`（无需参数），返回 `task_id`。
- （可选）复用 `taskService` 来轮询/订阅该任务进度，而不是单独写轮询逻辑。

- **3. 页面增加按钮和进度展示**
- 在数据管理页增加一个操作区域：
- 按钮文案示例：**“离线生成 Qlib 指标/因子”**。
- **无需弹窗选择参数**，直接点击按钮即可触发：
- 点击后直接调用 `dataService.triggerQlibPrecompute()`，拿到 `task_id`。
- 通过现有任务状态组件/`useTaskStore` 显示进度（可以参考回测任务进度的 UI 实现）。
- 显示提示信息：**"正在为全市场所有股票计算所有指标和因子，请耐心等待..."**
- 完成后给出“预计算完成，可用于训练/回测”的提示。

## 当前使用的指标清单

### 1. 技术指标（TechnicalIndicatorCalculator）

**移动平均线类**：

- MA5, MA10, MA20, MA60（简单移动平均）
- SMA（简单移动平均，默认20日）
- EMA（指数移动平均，默认20日）
- WMA（加权移动平均，默认20日）

**动量指标类**：

- RSI（相对强弱指数，默认14日）
- STOCH（随机指标，K和D值）
- WILLIAMS_R（威廉指标，默认14日）
- CCI（商品通道指数，默认20日）
- MOMENTUM（动量指标）
- ROC（变化率）

**趋势指标类**：

- MACD（MACD线、信号线、柱状图）
- BOLLINGER（布林带：上轨、中轨、下轨）
- SAR（抛物线SAR）
- ADX（平均趋向指数）
- ICHIMOKU（一目均衡图）

**成交量指标类**：

- VWAP（成交量加权平均价格）
- OBV（能量潮）
- AD_LINE（累积/派发线）
- VOLUME_RSI（成交量相对强弱指数）

**波动率指标类**：

- ATR（平均真实波幅，默认14日）
- VOLATILITY（波动率）
- HISTORICAL_VOLATILITY（历史波动率）

**复合指标类**：

- KDJ（K值、D值、J值）

### 2. Alpha因子（Alpha158Calculator）

**Alpha158因子（可直接使用Qlib内置实现）**：

**当前状态**：

- 代码中已经导入了Qlib内置Alpha158：`from qlib.contrib.data.handler import Alpha158` 和 `from qlib.contrib.data.loader import Alpha158DL`
- `Alpha158Calculator` 已经通过 `Alpha158DL.get_feature_config()` 获取了158个标准因子的配置（`self.alpha_fields` 和 `self.alpha_names`）
- **但实际计算时使用的是简化版本**（`_calculate_factors_for_stock`），只计算了32个因子

**优化方案**：

- **直接复用Qlib内置Alpha158**：使用 `Alpha158` handler 或通过Qlib表达式引擎计算158个标准因子
- 标准Alpha158因子集包含158个因子：
  - K线特征（KMID, KLEN, KUP, KLOW等）
  - 价格特征（OPEN, HIGH, LOW, VWAP等）
  - 滚动操作因子（ROC, MA, STD, BETA, RSQR, RESI, MAX, MIN, QTLU, QTLD, RANK, RSV等）
  - 成交量因子（VMA, VSTD, WVMA, VSUMP, VSUMN, VSUMD等）
  - 相关性因子（CORR, CORD等）
  - 统计因子（CNTP, CNTN, CNTD, SUMP, SUMN, SUMD等）

**实现方式**（在预计算服务中）：

```python
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL

# 方式1：使用Alpha158 handler（推荐，直接获得158个因子）
handler = Alpha158(
    instruments=stock_codes,
    start_time=start_date,
    end_time=end_date,
    fit_start_time=start_date,
    fit_end_time=end_date,
)
alpha_factors = handler.fetch()  # 直接获得158个标准因子

# 方式2：使用Alpha158DL获取因子配置，然后用Qlib表达式引擎计算
fields, names = Alpha158DL.get_feature_config(config)
# 使用Qlib的表达式引擎计算这些因子表达式
```

### 3. 基本面特征（FeatureCalculator）

**价格特征**：

- price_change（价格变化率）
- price_change_5d（5日价格变化率）
- price_change_20d（20日价格变化率）
- price_position（价格位置：当前价格在20日区间的位置）

**成交量特征**：

- volume_change（成交量变化率）
- volume_ma_ratio（成交量与20日均量比率）

**波动率特征**：

- volatility_5d（5日波动率）
- volatility_20d（20日波动率）

**动量特征**：

- momentum_5d（5日价格动量）
- momentum_20d（20日价格动量）
- volume_momentum_5d（5日成交量动量）
- volume_momentum_20d（20日成交量动量）

**价格范围特征**：

- price_range（价格范围：最高-最低）
- price_range_ratio（价格范围比率）

**价格比率特征**：

- open_close_ratio（开盘收盘比率）
- high_close_ratio（最高收盘比率）
- low_close_ratio（最低收盘比率）

### 4. 回测策略中使用的指标

**移动平均策略**：

- sma_short（短期移动平均）
- sma_long（长期移动平均）
- ma_diff（移动平均差值）

**RSI策略**：

- RSI（相对强弱指数）
- 趋势对齐指标（结合移动平均）

**MACD策略**：

- MACD线、信号线、柱状图

**布林带策略**：

- 上轨、中轨、下轨

**多因子策略**：

- 各种因子组合

## 一次性迁移方案

### 迁移策略：**全部迁移到预计算模式**

**可行性分析**：

- ✅ **技术可行**：所有指标都可以通过现有计算器实现，可以统一预计算
- ✅ **架构可行**：Qlib数据格式支持MultiIndex，可以存储所有指标
- ✅ **性能可行**：预计算一次，多次使用，显著提升回测和训练性能

**迁移步骤**：

1. **预计算服务实现**：

   - 创建 `OfflineFactorPrecomputeService`，统一调用所有指标计算器
   - 计算顺序：

     1. 基础OHLCV数据（从Parquet加载）
     2. **基础统计指标（使用Qlib计算，性能优势明显）**：

        - 使用Qlib表达式引擎计算：Mean, Std, Max, Min, Quantile, Corr等
        - 价格变化率：`Ref($close, n) / $close - 1`
        - 移动平均：`Mean($close, n)`
        - 相关性：`Corr($close, $volume, n)`
        - **优势**：多股票批量计算性能提升50-100%

     1. **Alpha158因子（使用Qlib内置Alpha158，获得158个标准因子）**：

        - 代码中已导入：`from qlib.contrib.data.handler import Alpha158` 和 `from qlib.contrib.data.loader import Alpha158DL`
        - 使用 `Alpha158` handler 直接计算158个标准因子，或通过 `Alpha158DL.get_feature_config()` 获取因子表达式后用Qlib表达式引擎计算
        - 不再使用当前简化版的32个因子实现（`_calculate_factors_for_stock`）
        - **优势**：因子计算性能提升100-200%

     1. **技术指标（使用Pandas计算，灵活且成熟）**：

        - 通过 `TechnicalIndicatorCalculator` 计算：RSI, MACD, 布林带, KDJ, ATR, VWAP, OBV等
        - **原因**：Qlib不直接支持这些技术指标，需要用表达式实现，工作量大且可能性能不如pandas
        - **优势**：代码成熟、灵活、单股票场景性能好

     1. **基本面特征（使用Pandas计算）**：

        - 通过 `FeatureCalculator` 计算：价格变化率、成交量变化率、波动率、价格位置等

   - **合并所有指标到一个DataFrame，转换为Qlib格式存储**
   - **关键优势**：全部预计算后，回测时直接读取，无需重复计算，回测速度大幅提升（预计50-200%）

2. **数据存储格式**：

   - 使用Qlib标准格式：MultiIndex DataFrame `(stock_code, date)`
   - 列名规范：
     - 基础统计指标（Qlib计算）：`MEAN_5`, `STD_20`, `CORR_10` 等
     - Alpha因子（Qlib计算）：`alpha_001` 到 `alpha_158`（标准Alpha158因子）
     - 技术指标（Pandas计算）：`MA5`, `RSI14`, `MACD`, `BOLLINGER_UPPER`, `KDJ_K` 等
     - 基本面特征（Pandas计算）：`price_change`, `volatility_5d`, `volume_ma_ratio` 等
   - 存储到 `QLIB_DATA_PATH` 目录
   - **数据完整性**：确保所有指标都预计算完成，回测时可以直接使用

3. **回测/训练改造**（全部预计算，直接读取）：

   - **回测服务**（`backtest_executor.py`）：
     - 修改数据加载逻辑：**优先从Qlib预计算目录读取所有指标**（包括基础指标、Alpha158因子、技术指标、基本面特征）
     - 策略中的 `calculate_indicators` 方法改为从预计算数据中提取所需指标，**不再现场计算**
     - 如果预计算数据不存在或缺少某个指标，fallback到pandas现场计算（向后兼容）
     - **性能提升**：
       - ✅ 回测时不再需要计算指标，直接读取，**回测速度大幅提升（预计50-200%）**
       - ✅ 避免重复计算，节省CPU资源
       - ✅ 支持更大规模的回测（更多股票、更长时间范围）
   - **训练服务**（`UnifiedQlibTrainingEngine`）：
     - `EnhancedQlibDataProvider.prepare_qlib_dataset` 优先使用预计算数据
     - 如果预计算数据存在，直接加载；否则现场计算
     - **性能提升**：训练时不再需要计算指标，直接加载，训练速度大幅提升

4. **迁移检查清单**：

   - [ ] 所有技术指标都已预计算
   - [ ] 所有Alpha因子都已预计算
   - [ ] 所有基本面特征都已预计算
   - [ ] 回测服务已改造为读取预计算数据
   - [ ] 训练服务已改造为读取预计算数据
   - [ ] 保留fallback机制确保向后兼容
   - [ ] 测试验证：回测和训练都能正常工作

## 指标扩展机制

### 设计原则

1. **可扩展性**：新增指标不需要修改核心代码，只需注册新指标
2. **向后兼容**：新指标不影响已有功能
3. **版本管理**：指标版本化，支持增量更新

### 实现方案

#### 1. 指标注册机制

创建 `backend/app/services/data/indicator_registry.py`：

```python
class IndicatorRegistry:
    """指标注册表"""
    
    # 技术指标注册
    TECHNICAL_INDICATORS = {
        'MA5': {
            'calculator': TechnicalIndicatorCalculator,
            'method': 'calculate_moving_average',
            'params': {'period': 5},
            'category': 'trend'
        },
        # ... 其他指标
    }
    
    # Alpha因子注册
    ALPHA_FACTORS = {
        'RESI5': {
            'calculator': Alpha158Calculator,
            'method': '_calculate_resi',
            'params': {'period': 5},
            'category': 'return'
        },
        # ... 其他因子
    }
    
    # 基本面特征注册
    FUNDAMENTAL_FEATURES = {
        'price_change': {
            'calculator': FeatureCalculator,
            'method': 'calculate_price_change',
            'params': {},
            'category': 'price'
        },
        # ... 其他特征
    }
    
    @classmethod
    def register_indicator(cls, name: str, config: dict, category: str = 'technical'):
        """注册新指标"""
        if category == 'technical':
            cls.TECHNICAL_INDICATORS[name] = config
        elif category == 'alpha':
            cls.ALPHA_FACTORS[name] = config
        elif category == 'fundamental':
            cls.FUNDAMENTAL_FEATURES[name] = config
    
    @classmethod
    def get_all_indicators(cls) -> Dict[str, dict]:
        """获取所有已注册的指标"""
        return {
            **cls.TECHNICAL_INDICATORS,
            **cls.ALPHA_FACTORS,
            **cls.FUNDAMENTAL_FEATURES
        }
```

#### 2. 预计算服务支持扩展

在 `OfflineFactorPrecomputeService` 中：

```python
class OfflineFactorPrecomputeService:
    def __init__(self):
        self.indicator_registry = IndicatorRegistry()
        self.calculators = {
            'technical': TechnicalIndicatorCalculator(),
            'alpha': Alpha158Calculator(),
            'fundamental': FeatureCalculator()
        }
    
    async def precompute_all_indicators(self, stock_codes: List[str], date_range: Tuple):
        """预计算所有已注册的指标"""
        all_indicators = self.indicator_registry.get_all_indicators()
        
        for indicator_name, config in all_indicators.items():
            calculator = self.calculators[config['category']]
            method = getattr(calculator, config['method'])
            result = await method(**config['params'])
            # 存储结果
```

#### 3. 扩展新指标的步骤

**步骤1：实现指标计算逻辑**

- 在对应的计算器中添加计算方法（如 `TechnicalIndicatorCalculator.calculate_new_indicator`）

**步骤2：注册新指标**

```python
# 在 indicator_registry.py 或配置文件中
IndicatorRegistry.register_indicator(
    name='NEW_INDICATOR',
    config={
        'calculator': TechnicalIndicatorCalculator,
        'method': 'calculate_new_indicator',
        'params': {'period': 10},
        'category': 'technical'
    },
    category='technical'
)
```

**步骤3：重新运行预计算**

- 在前端点击"离线生成 Qlib 指标/因子"按钮
- 预计算服务会自动检测新注册的指标并计算
- 新指标会被添加到Qlib数据目录

**步骤4：使用新指标**

- 回测策略：在 `calculate_indicators` 中从预计算数据读取新指标
- 模型训练：新指标会自动包含在特征集中

#### 4. 版本管理

**指标版本化**：

- 在Qlib数据目录中维护 `indicator_versions.json`：
```json
{
    "version": "1.0.0",
    "indicators": {
        "MA5": {"version": "1.0.0", "added_date": "2026-01-01"},
        "NEW_INDICATOR": {"version": "1.1.0", "added_date": "2026-01-20"}
    }
}
```


**增量更新**：

- 检测新指标：对比注册表和版本文件
- 只计算新增的指标，避免重复计算已有指标
- 合并新旧指标数据

#### 5. 扩展示例

**示例：添加新的技术指标 "TRIX"（三重指数平滑移动平均）**

1. **实现计算逻辑**：
```python
# 在 TechnicalIndicatorCalculator 中添加
def calculate_trix(self, data: List[StockData], period: int = 14) -> List[Optional[float]]:
    # TRIX计算逻辑
    ...
```

2. **注册指标**：
```python
IndicatorRegistry.register_indicator(
    name='TRIX',
    config={
        'calculator': TechnicalIndicatorCalculator,
        'method': 'calculate_trix',
        'params': {'period': 14},
        'category': 'technical'
    }
)
```

3. **重新预计算**：

- 前端触发预计算任务
- 系统自动检测到新指标并计算

4. **使用新指标**：

- 回测策略中可以直接使用 `indicators['TRIX']`
- 模型训练中自动包含TRIX特征

## 与现有回测/训练的衔接

- **回测**
- 当前回测主要在 `backend/app/services/backtest/` 中用 pandas 现场计算一些指标。
- **优化方案**（全部预计算，直接读取）：
  - 在 `backtest_executor.py` 的数据加载阶段，**优先从 Qlib 预计算目录读取所有指标**（包括基础指标、Alpha158因子、技术指标、基本面特征）
  - 策略中的 `calculate_indicators` 方法改为从预计算数据中提取所需指标，**不再现场计算**
  - 如果预计算数据不存在或缺少某个指标，fallback 到 pandas 现场计算（保证兼容性）
  - **性能提升**：
    - ✅ 回测时不再需要计算指标，直接读取，**回测速度大幅提升（预计提升50-200%）**
    - ✅ 避免重复计算，节省CPU资源
    - ✅ 支持更大规模的回测（更多股票、更长时间范围）

- **模型训练 / 预测**
- 现有 `EnhancedQlibDataProvider` / `UnifiedQlibTrainingEngine` 已经是 Qlib 数据消费方：
- **迁移方案**：
- `EnhancedQlibDataProvider.prepare_qlib_dataset` 优先从预计算目录加载数据
- 如果预计算数据完整，直接使用；否则补充计算缺失的指标
- **性能提升**：训练时不再需要计算指标，直接加载，大幅提升训练速度

## 数据格式转换细节

### Parquet → Qlib格式转换

**转换步骤**：

1. **读取Parquet数据**：

   - 使用 `StockDataLoader` 从 `data/parquet/stock_data/{safe_code}.parquet` 加载
   - 数据格式：单股票DataFrame，索引为日期，列名为 `open, high, low, close, volume`

2. **构建MultiIndex**：
   ```python
   # 单股票数据转换为MultiIndex格式
   df['stock_code'] = stock_code
   df = df.set_index(['stock_code', df.index])  # MultiIndex: (stock_code, date)
   ```

3. **列名标准化**：
   ```python
   column_mapping = {
       'open': '$open',
       'high': '$high',
       'low': '$low',
       'close': '$close',
       'volume': '$volume'
   }
   df = df.rename(columns=column_mapping)
   ```

4. **数据类型优化**：

   - 价格列：float32（节省内存）
   - 成交量列：int64
   - 日期索引：datetime64[ns]

5. **合并多股票数据**：
   ```python
   # 合并所有股票的MultiIndex DataFrame
   all_data = pd.concat([df1, df2, ...], axis=0)
   all_data = all_data.sort_index()  # 按(stock_code, date)排序
   ```


### Qlib数据存储结构

**存储位置**：`settings.QLIB_DATA_PATH`（默认：`../data/qlib_data`）

**存储格式**：

- 使用Qlib的bin格式或Parquet格式（与Qlib兼容）
- 目录结构建议：
  ```
  qlib_data/
  ├── instruments/          # 股票列表
  ├── calendars/           # 交易日历
  └── features/            # 特征数据（按日期或按股票组织）
      ├── day/            # 日线数据
      │   ├── {stock_code}.bin  # 或按日期组织
      └── 1min/            # 分钟数据（如果有）
  ```


**数据组织方式**：

- 方案1：按股票组织（推荐，便于增量更新）
  - 每个股票一个文件：`{stock_code}.parquet` 或 `{stock_code}.bin`
  - 包含该股票的所有日期和所有指标
- 方案2：按日期组织
  - 每个日期一个文件：`{date}.parquet`
  - 包含该日期所有股票的数据

## 增量更新机制

### 检测数据变化

**触发条件**：

1. **新数据同步**：当SFTP同步服务或数据服务更新了Parquet数据时
2. **定时检查**：定期检查Parquet文件的修改时间
3. **手动触发**：前端点击"更新预计算数据"按钮

**检测方法**：

```python
# 1. 检查Parquet文件修改时间
parquet_mtime = get_file_mtime(parquet_file)
qlib_mtime = get_file_mtime(qlib_file)

# 2. 检查数据日期范围
parquet_date_range = get_date_range(parquet_file)
qlib_date_range = get_date_range(qlib_file)

# 3. 如果Parquet更新或日期范围扩展，触发增量更新
if parquet_mtime > qlib_mtime or parquet_date_range > qlib_date_range:
    trigger_incremental_update()
```

### 增量更新策略

**更新范围**：

- **新股票**：检测到新的股票代码，计算该股票的所有指标
- **新日期**：检测到新的日期数据，只计算新增日期的指标（需要历史数据依赖的指标可能需要重新计算）
- **数据修正**：检测到数据被修改，重新计算受影响的时间段

**更新流程**：

1. 检测变化：对比Parquet和Qlib数据的时间戳、日期范围
2. 确定更新范围：哪些股票、哪些日期需要更新
3. 增量计算：只计算变化的部分
4. 合并数据：将新计算结果合并到现有Qlib数据中
5. 验证数据：确保数据完整性和一致性

## 数据完整性和验证

### 预计算数据验证

**验证内容**：

1. **数据完整性检查**：

   - 检查所有股票是否都有数据
   - 检查所有日期是否都有数据
   - 检查所有指标是否都已计算

2. **数据质量检查**：

   - 使用 `DataValidator` 验证数据质量
   - 检查缺失值、异常值、数据逻辑
   - 验证指标计算的合理性（如RSI在0-100之间）

3. **数据一致性检查**：

   - 对比预计算数据与原始Parquet数据的一致性
   - 验证指标计算结果的正确性（抽样对比）

**验证机制**：

```python
class PrecomputeValidator:
    def validate_precomputed_data(self, qlib_data: pd.DataFrame, 
                                  parquet_data: Dict[str, pd.DataFrame]) -> ValidationResult:
        """验证预计算数据的完整性和正确性"""
        # 1. 检查数据完整性
        missing_stocks = self._check_missing_stocks(qlib_data, parquet_data)
        missing_dates = self._check_missing_dates(qlib_data, parquet_data)
        missing_indicators = self._check_missing_indicators(qlib_data)
        
        # 2. 抽样验证指标计算正确性
        sample_validation = self._sample_validate_indicators(qlib_data, parquet_data)
        
        # 3. 数据质量检查
        quality_issues = self._check_data_quality(qlib_data)
        
        return ValidationResult(...)
```

### 数据损坏恢复

**恢复策略**：

1. **自动恢复**：

   - 检测到数据损坏时，自动重新计算该部分数据
   - 从Parquet原始数据重新生成Qlib数据

2. **备份机制**：

   - 预计算完成后，创建数据备份
   - 定期备份Qlib数据目录

3. **回退机制**：

   - 如果预计算数据不可用，回退到pandas现场计算
   - 确保系统始终可用

## 错误处理和容错

### 预计算错误处理

**错误类型和处理**：

1. **单股票计算失败**：

   - 记录失败股票，继续处理其他股票
   - 任务完成后报告失败股票列表
   - 支持重试失败的股票

2. **部分指标计算失败**：

   - 记录失败的指标，继续计算其他指标
   - 在结果中标记缺失的指标
   - 回测时fallback到pandas计算缺失指标

3. **内存不足**：

   - 分批处理，减少批次大小
   - 及时释放内存，使用流式处理
   - 记录内存使用情况

4. **磁盘空间不足**：

   - 检查磁盘空间，提前告警
   - 清理旧数据或临时文件
   - 任务失败时清理部分写入的数据

### 任务中断和重试

**中断处理**：

- 支持任务中断（Ctrl+C或API取消）
- 保存中间状态，支持断点续传
- 记录已完成的股票和日期，避免重复计算

**重试机制**：

- 支持任务重试（从上次中断点继续）
- 支持部分重试（只重试失败的股票）
- 重试次数限制（避免无限重试）

## 性能优化细节

### 分批处理策略

**股票分批**：

- 每批处理股票数：50-100只（根据内存情况调整）
- 并行处理：使用 `ThreadPoolExecutor` 或 `ProcessPoolExecutor`
- 最大并发数：`min(CPU核心数, 8)`（避免过多线程开销）

**日期分批**：

- 按月份或季度分批处理
- 避免一次性加载所有历史数据

**内存管理**：

- 及时释放中间数据
- 使用流式处理，避免大内存峰值
- 监控内存使用，超过阈值时强制GC

### 并行计算策略

**Qlib计算**：

- 多股票批量计算：使用Qlib的并行优化
- 因子计算：使用 `ProcessPoolExecutor`（绕过GIL限制）

**Pandas计算**：

- 技术指标：使用 `ThreadPoolExecutor`（I/O密集型）
- 单股票内并行：对于多指标计算，可以使用向量化操作

## 数据一致性保证

### 版本管理

**数据版本**：

- 在Qlib数据目录维护 `data_version.json`：
```json
{
    "version": "1.0.0",
    "precompute_date": "2026-01-23",
    "parquet_version": "2026-01-23",
    "indicators": {
        "technical": ["MA5", "RSI14", ...],
        "alpha": ["alpha_001", ..., "alpha_158"],
        "fundamental": ["price_change", ...]
    },
    "stock_count": 5000,
    "date_range": {
        "start": "2020-01-01",
        "end": "2026-01-23"
    }
}
```


**一致性检查**：

- 预计算时记录Parquet数据版本
- 回测时检查Qlib数据版本是否与Parquet一致
- 如果不一致，触发增量更新或重新预计算

### 数据校验机制

**校验点**：

1. **预计算完成后**：验证数据完整性
2. **回测加载时**：快速检查数据是否存在
3. **定期校验**：后台任务定期校验数据一致性

## 回测改造详细方案

### 数据加载器改造

**文件**：`backend/app/services/backtest/execution/data_loader.py`

**改造点**：

```python
class DataLoader:
    def load_stock_data_with_indicators(self, stock_code: str, start_date: datetime, 
                                       end_date: datetime) -> pd.DataFrame:
        """加载股票数据，优先从预计算结果读取指标"""
        # 1. 尝试从Qlib预计算目录加载
        qlib_data = self._load_from_precomputed(stock_code, start_date, end_date)
        if qlib_data is not None and self._validate_precomputed_data(qlib_data):
            logger.info(f"从预计算结果加载: {stock_code}")
            return qlib_data
        
        # 2. Fallback：从Parquet加载并现场计算
        logger.info(f"预计算结果不可用，从Parquet加载并计算: {stock_code}")
        return self._load_from_parquet_and_calculate(stock_code, start_date, end_date)
    
    def _load_from_precomputed(self, stock_code: str, start_date: datetime, 
                               end_date: datetime) -> Optional[pd.DataFrame]:
        """从Qlib预计算目录加载数据"""
        # 加载预计算的Qlib格式数据
        # 转换为回测需要的格式
        ...
    
    def load_multiple_stocks_with_indicators(self, stock_codes: List[str], 
                                            start_date: datetime, end_date: datetime,
                                            parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """加载多只股票数据，优先从预计算结果读取"""
        # 并行加载预计算数据，性能优势明显
        ...
```

**数据格式适配**：

- 预计算数据是MultiIndex格式 `(stock_code, date)`
- 回测需要的是单股票DataFrame，索引为日期
- 需要从MultiIndex中提取单股票数据并转换格式
- 列名映射：`$close` → `close`（回测策略期望的格式）

### 策略指标提取改造

**文件**：`backend/app/services/backtest/strategies/technical/basic_strategies.py`

**改造点**：

```python
class RSIStrategy(BaseStrategy):
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """从预计算数据中提取指标，不再现场计算"""
        # 如果数据已经包含预计算的指标，直接提取
        if 'RSI14' in data.columns:
            return {
                'rsi': data['RSI14'],
                'price': data['$close'] if '$close' in data.columns else data['close']
            }
        
        # Fallback：现场计算（向后兼容）
        return self._calculate_indicators_fallback(data)
```

## 监控和日志

### 预计算进度监控

**监控指标**：

- 已处理股票数 / 总股票数
- 已处理日期数 / 总日期数
- 已计算指标数 / 总指标数
- 预计剩余时间
- 内存使用情况
- CPU使用情况

**日志记录**：

- 每个股票的预计算结果（成功/失败）
- 每个指标的计算时间
- 数据质量检查结果
- 错误和警告信息

### 数据质量监控

**监控内容**：

- 预计算数据完整性（缺失股票、缺失日期、缺失指标）
- 数据质量评分
- 数据一致性检查结果
- 预计算数据与原始数据的一致性

## 测试和验证

### 数据准确性验证

**验证方法**：

1. **抽样对比**：随机抽取部分股票和日期，对比预计算结果与pandas现场计算结果
2. **全量对比**：小规模数据全量对比验证
3. **回测结果对比**：使用预计算数据和现场计算数据分别回测，对比结果

### 性能对比测试

**测试场景**：

- 单股票回测：对比预计算 vs 现场计算的速度
- 多股票回测：对比预计算 vs 现场计算的速度
- 长时间范围回测：对比预计算 vs 现场计算的速度

## 部署和运维

### 预计算任务调度

**调度策略**：

- **初次预计算**：手动触发（前端按钮）
- **增量更新**：
  - 方案1：数据同步后自动触发增量更新
  - 方案2：定时任务（如每天凌晨）检查并更新
  - 方案3：手动触发（前端按钮）

### 数据备份和清理

**备份策略**：

- 预计算完成后自动备份
- 定期备份（如每周）
- 备份到独立目录或远程存储

**清理策略**：

- 保留最近N个版本的预计算数据
- 清理过期的预计算数据
- 清理临时文件和缓存

### 数据加载性能优化

**索引机制**：

- 为Qlib数据建立索引文件（如 `index.json`），记录每个股票的数据位置和日期范围
- 快速定位和加载所需数据，避免全量扫描
- 支持按股票代码、日期范围快速查询

**缓存机制**：

- 内存缓存：最近使用的预计算数据缓存在内存中（LRU策略）
- 磁盘缓存：使用Parquet格式存储，支持快速读取
- 缓存失效：数据更新时自动失效缓存

**数据压缩**：

- 使用Parquet的压缩功能（如snappy、gzip）
- 平衡压缩率和读取速度
- 评估：snappy压缩速度快，gzip压缩率高

### 数据同步触发机制

**自动触发**：

- 监听Parquet文件变化（使用文件系统监控或定时检查）
- 数据同步服务完成后，自动触发增量预计算
- 配置选项：是否启用自动触发（默认关闭，避免频繁触发）

**手动触发**：

- 前端提供"更新预计算数据"按钮
- 支持选择更新范围（全部/指定股票/指定日期）
- 显示更新进度和结果

### 多进程/多线程配置

**Qlib计算**：

- 使用 `ProcessPoolExecutor`（绕过GIL限制）
- 最大进程数：`min(CPU核心数, 8)`
- 每个进程处理一批股票（50-100只）

**Pandas计算**：

- 使用 `ThreadPoolExecutor`（I/O密集型）
- 最大线程数：`min(CPU核心数 * 2, 16)`
- 每个线程处理一只股票

**内存控制**：

- 监控内存使用，超过阈值（如80%）时减少并发数
- 分批处理，及时释放内存
- 使用流式处理，避免大内存峰值

## 风险与注意事项

- 初次运行离线预计算在全市场+长时间范围下可能耗时较长，需要：
  - 任务可中断/重试机制。
  - 分批写入避免大内存峰值。
  - 进度监控和日志记录。
- 需要明确 Qlib 数据目录与原始 Parquet 目录的对应关系，避免路径错配。
- 确保 Qlib 版本和数据格式与当前 `EnhancedQlibDataProvider` 的实现一致，避免格式不兼容。
- **数据一致性风险**：
  - Parquet数据更新后，Qlib预计算数据可能过期
  - 需要建立数据版本管理和一致性检查机制
- **存储空间风险**：
  - 预计算数据可能占用大量存储空间（全市场+全时间+所有指标）
  - 需要评估存储需求，必要时实施数据压缩或清理策略
- **性能风险**：
  - 预计算数据加载可能比现场计算慢（如果数据量大且I/O慢）
  - 需要优化数据加载和索引机制

## 实施检查清单

### 技术实现检查

- [ ] **数据格式转换**：
  - [ ] Parquet → Qlib格式转换工具实现
  - [ ] MultiIndex构建逻辑
  - [ ] 列名映射（open → $open等）
  - [ ] 数据类型优化

- [ ] **预计算服务**：
  - [ ] Qlib基础指标计算（Mean, Std, Corr等）
  - [ ] Qlib Alpha158因子计算（158个标准因子）
  - [ ] Pandas技术指标计算（RSI, MACD等）
  - [ ] Pandas基本面特征计算
  - [ ] 数据合并和格式转换
  - [ ] 数据存储到Qlib目录

- [ ] **任务系统集成**：
  - [ ] 任务类型定义（QLIB_OFFLINE_PRECOMPUTE）
  - [ ] 任务执行逻辑
  - [ ] 进度更新机制
  - [ ] WebSocket进度推送
  - [ ] 任务中断和重试

- [ ] **API接口**：
  - [ ] POST /api/v1/data/qlib/precompute
  - [ ] 任务查询接口（或复用通用接口）
  - [ ] 错误处理

### 数据管理检查

- [ ] **增量更新**：
  - [ ] 数据变化检测机制
  - [ ] 增量计算逻辑
  - [ ] 数据合并策略

- [ ] **数据验证**：
  - [ ] 完整性检查
  - [ ] 质量检查
  - [ ] 一致性检查

- [ ] **错误处理**：
  - [ ] 单股票失败处理
  - [ ] 部分指标失败处理
  - [ ] 任务中断和重试
  - [ ] 数据损坏恢复

### 回测/训练改造检查

- [ ] **数据加载器改造**：
  - [ ] 优先从预计算结果读取
  - [ ] 数据格式适配（MultiIndex → 单股票DataFrame）
  - [ ] Fallback机制（pandas现场计算）
  - [ ] 多股票并行加载优化

- [ ] **策略改造**：
  - [ ] 从预计算数据提取指标
  - [ ] 列名映射（$close → close）
  - [ ] Fallback机制（现场计算）

- [ ] **训练服务改造**：
  - [ ] EnhancedQlibDataProvider优先使用预计算数据
  - [ ] 数据格式兼容性

### 前端实现检查

- [ ] **服务封装**：
  - [ ] triggerQlibPrecompute()方法
  - [ ] 任务进度查询

- [ ] **UI组件**：
  - [ ] 预计算触发按钮
  - [ ] 进度显示组件
  - [ ] 状态展示（未开始/进行中/已完成/失败）
  - [ ] 统计信息展示
  - [ ] 结果详情展示

### 性能优化检查

- [ ] **分批处理**：
  - [ ] 股票分批策略
  - [ ] 日期分批策略
  - [ ] 内存管理

- [ ] **并行计算**：
  - [ ] Qlib多进程配置
  - [ ] Pandas多线程配置
  - [ ] 并发数控制

- [ ] **数据加载优化**：
  - [ ] 索引机制
  - [ ] 缓存机制
  - [ ] 数据压缩

### 监控和测试检查

- [ ] **监控**：
  - [ ] 预计算进度监控
  - [ ] 数据质量监控
  - [ ] 性能监控

- [ ] **测试**：
  - [ ] 数据准确性验证
  - [ ] 性能对比测试
  - [ ] 兼容性测试
  - [ ] 错误处理测试

### 部署和运维检查

- [ ] **任务调度**：
  - [ ] 初次预计算触发
  - [ ] 增量更新触发机制
  - [ ] 定时任务配置

- [ ] **数据管理**：
  - [ ] 数据备份策略
  - [ ] 数据清理策略
  - [ ] 存储空间管理

## 关键注意事项总结

1. **数据格式一致性**：确保预计算数据格式与回测/训练期望的格式一致
2. **向后兼容性**：保留fallback机制，确保预计算数据不可用时系统仍可用
3. **性能平衡**：预计算数据加载速度 vs 现场计算速度，需要实际测试验证
4. **存储空间**：评估全市场+全时间+所有指标的存储需求
5. **数据同步**：建立Parquet数据更新与Qlib预计算数据同步机制
6. **错误恢复**：完善的错误处理和恢复机制，确保系统稳定性
7. **测试验证**：充分测试，确保预计算结果与现场计算结果一致