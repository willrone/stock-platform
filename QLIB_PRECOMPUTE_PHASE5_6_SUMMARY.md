# Qlib预计算优化方案 - 阶段五、六实施总结

## 阶段五：Alpha158升级 ✅

### 已完成工作

1. **升级Alpha158Calculator**
   - 导入Qlib内置的`Alpha158Handler`
   - 重写`calculate_alpha_factors()`方法，支持两种计算方式：
     - **方法1**：使用`Alpha158Handler.fetch()`直接获取158个标准因子（推荐）
     - **方法2**：使用表达式引擎计算（基于Alpha158DL.get_feature_config()获取的因子表达式）
   - 添加`_calculate_using_alpha158_handler()`方法
   - 添加`_calculate_using_expression_engine()`方法
   - 添加`_calculate_alpha_factors_from_expressions()`方法（pandas实现）

2. **因子完整性验证**
   - 支持158个标准Alpha158因子
   - 自动检测因子数量，确保完整性
   - 如果handler不可用，自动fallback到表达式引擎

### 技术实现

- **优先使用Alpha158Handler**：直接调用`handler.fetch()`获取158个标准因子
- **Fallback机制**：如果handler不可用，使用表达式引擎计算
- **兼容性**：保留原有的简化版本作为最后fallback

---

## 阶段六：增量更新与优化 ✅

### 已完成工作

1. **增量更新机制** (`incremental_updater.py`)
   - **数据变化检测**：
     - 检测Parquet文件修改时间
     - 检测日期范围变化
     - 识别新股票、更新股票、无变化股票
   - **增量计算**：
     - 只计算需要更新的股票
     - 合并新旧数据（避免重复计算）
   - **智能更新**：
     - 自动检测需要更新的股票列表
     - 支持强制全量更新

2. **指标注册机制** (`indicator_registry.py`)
   - **动态注册**：
     - 支持动态注册新指标
     - 指标分类管理（技术指标、Alpha因子、基本面特征、基础指标）
     - 指标版本管理
     - 指标启用/禁用
   - **默认注册**：
     - 自动注册所有技术指标（MA5, MA10, MA20, MA60, RSI14, MACD, BOLLINGER等）
     - 自动注册158个Alpha158因子
     - 自动注册基本面特征
   - **扩展性**：
     - 新增指标只需注册，无需修改核心代码
     - 支持指标版本管理

3. **数据版本管理** (`version_manager.py`)
   - **版本信息**：
     - 记录预计算日期
     - 记录Parquet数据版本
     - 记录股票数量、日期范围
     - 记录所有指标列表
   - **版本一致性检查**：
     - 检查股票数量一致性
     - 检查日期范围一致性
     - 检查指标完整性
   - **版本对比**：
     - 与Parquet数据版本对比
     - 自动检测是否需要更新

4. **性能优化**
   - **分批处理**：默认50只股票/批，可配置
   - **并行计算**：多进程/多线程并行处理
   - **内存管理**：
     - 及时释放中间数据
     - 分批写入避免大内存峰值
   - **增量更新**：只计算变化的部分，大幅减少计算时间

5. **集成到预计算服务**
   - 预计算服务支持增量更新模式
   - 自动检测需要更新的股票
   - 自动合并增量数据
   - 自动更新版本信息

---

## 核心功能总结

### 1. Alpha158升级
- ✅ 支持158个标准Alpha158因子
- ✅ 优先使用Qlib内置Alpha158Handler
- ✅ 自动fallback机制

### 2. 增量更新
- ✅ 自动检测数据变化
- ✅ 只计算需要更新的股票
- ✅ 智能合并新旧数据

### 3. 指标注册
- ✅ 动态注册新指标
- ✅ 指标版本管理
- ✅ 指标启用/禁用

### 4. 版本管理
- ✅ 数据版本记录
- ✅ 版本一致性检查
- ✅ 自动版本对比

### 5. 性能优化
- ✅ 分批处理
- ✅ 并行计算
- ✅ 内存管理
- ✅ 增量更新

---

## 使用方式

### 增量更新

预计算服务默认启用增量更新模式：

```python
# 自动检测并只更新变化的股票
result = await precompute_service.precompute_all_stocks(
    incremental=True,  # 默认True
    force_update=False  # 强制全量更新
)
```

### 注册新指标

```python
from app.services.data.indicator_registry import IndicatorRegistry, IndicatorCategory

# 注册新指标
IndicatorRegistry.register_indicator(
    name='TRIX',
    category=IndicatorCategory.TECHNICAL,
    calculator_class='TechnicalIndicatorCalculator',
    calculator_method='calculate_trix',
    params={'period': 14},
    description='三重指数平滑移动平均',
    version='1.0.0'
)
```

### 检查版本一致性

```python
from app.services.data.version_manager import VersionManager

version_manager = VersionManager()
consistency = version_manager.check_version_consistency(
    expected_stocks=stock_codes,
    expected_date_range={'start': '2020-01-01', 'end': '2026-01-25'}
)
```

---

## 文件清单

### 新增文件
1. `backend/app/services/data/indicator_registry.py` - 指标注册机制
2. `backend/app/services/data/incremental_updater.py` - 增量更新机制
3. `backend/app/services/data/version_manager.py` - 数据版本管理

### 修改文件
1. `backend/app/services/qlib/enhanced_qlib_provider.py` - Alpha158Calculator升级
2. `backend/app/services/data/offline_factor_precompute.py` - 集成增量更新和版本管理

---

## 验证建议

1. **Alpha158因子验证**：
   - 运行预计算任务
   - 检查生成的因子数量是否为158个
   - 验证因子名称和格式

2. **增量更新验证**：
   - 运行一次全量预计算
   - 修改部分Parquet文件
   - 再次运行预计算（增量模式）
   - 验证只更新了变化的股票

3. **版本管理验证**：
   - 检查`data/qlib_data/data_version.json`文件
   - 验证版本信息是否正确
   - 测试版本一致性检查

4. **指标注册验证**：
   - 注册一个新指标
   - 运行预计算
   - 验证新指标是否被计算

---

## 完成状态

✅ **阶段五：Alpha158升级** - 已完成
✅ **阶段六：增量更新与优化** - 已完成

所有核心功能已实现，可以开始验证测试！
