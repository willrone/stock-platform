# Willrone 模型训练流程修复计划

**创建日期**: 2026-02-10
**状态**: 待执行
**优先级**: Critical → Major → Minor

---

## 一、背景

通过对 Willrone 项目 16 个模型训练相关文件的深度分析，发现 5 个 Critical、13 个 Major、12 个 Minor 级别问题。本文档制定分阶段修复计划。

## 二、修复阶段

### 阶段 1：Critical 阻塞性 Bug（必须立即修复）

#### C1+C2: training.py 中 self 引用错误 + engine.py 调用不匹配

**文件**:
- `backend/app/services/qlib/training/training.py`
- `backend/app/services/qlib/unified_qlib_training_engine.py`

**问题**: `_train_qlib_model` 和 `_simulate_training_with_early_stopping` 是模块级函数，但函数体内使用了 `self`（从类中提取后未清理）。`engine.py` 调用时也没有传入 self。

**修复方案**:
1. 移除 `training.py` 中所有 `self.` 引用
2. 将 `self._simulate_training_with_early_stopping(...)` 改为直接调用 `_simulate_training_with_early_stopping(...)`
3. 确保函数签名和调用方式一致

**验证**: 运行 `python -c "from app.services.qlib.training.training import _train_qlib_model"` 确认无语法错误

---

#### C3: engine.py 调用了未导入的函数名

**文件**: `backend/app/services/qlib/unified_qlib_training_engine.py`

**问题**: 导入的是 `_analyze_feature_correlations`（带下划线），但调用时用的是 `analyze_feature_correlations`（不带下划线）。`_extract_feature_importance` 同理。

**修复方案**:
1. 统一函数命名：在 `training.py` 中将函数名改为不带前缀下划线的公开函数
2. 或在 `engine.py` 中修改调用名称为带下划线的版本
3. 推荐方案：统一为不带下划线（这些是模块级公开函数）

**验证**: `python -c "from app.services.qlib.unified_qlib_training_engine import UnifiedQlibTrainingEngine"` 确认导入无错误

---

#### C4+C5: pickle/torch.load 安全漏洞

**文件**:
- `backend/app/services/qlib/training/model_io.py`
- `backend/app/services/models/model_storage.py`
- `backend/app/services/models/model_evaluation.py`
- `backend/app/services/models/advanced_training.py`

**问题**: `pickle.load()` 和 `torch.load()` 无安全校验，可导致 RCE。

**修复方案**:
1. 所有 `torch.load()` 添加 `weights_only=True` 参数
2. `pickle.load()` 前添加文件哈希校验，不匹配则拒绝加载（而非仅 warning）
3. 在 `model_storage.py` 的 `_calculate_file_hash` 中将 MD5 改为 SHA256

**验证**: grep 确认所有 `torch.load` 都有 `weights_only=True`，所有 `pickle.load` 前都有哈希校验

---

### 阶段 2：Major 级别问题（短期修复）

#### M3: training_progress.py 中 task_manager 未导入

**文件**: `backend/app/api/v1/training_progress.py`

**问题**: `task_manager` 导入被注释掉，所有 API 端点都会 NameError。

**修复方案**: 取消注释导入语句，或实现延迟导入避免循环依赖。

---

#### M5: QlibDataProvider 引用未定义属性

**文件**: `backend/app/services/models/model_training.py`

**问题**: `QlibDataProvider.__init__` 中没有定义 `self.feature_engineer` 和 `self.data_root`。

**修复方案**: 在 `__init__` 中初始化这些属性，或移除对它们的引用。

---

#### M2: 数据库会话线程安全

**文件**: `backend/app/api/v1/models/models_training.py`

**问题**: `SessionLocal()` 在主线程创建，在子线程中使用，不是线程安全的。

**修复方案**: 在 `train_model_task` 函数内部创建新的 session，而非从外部传入。

---

#### M4: ProbAttention 逻辑错误

**文件**: `backend/app/services/models/modern_models.py`

**问题**: Top-k 选择后用 `V[:, :, :top_k, :]` 取值，应该用 `torch.gather` 按 `top_indices` 取值。

**修复方案**: 使用 `torch.gather(V, 2, top_indices.unsqueeze(-1).expand(-1, -1, -1, V.size(-1)))` 替代。

---

#### M6: 模型缓存无大小限制

**文件**: `backend/app/services/models/model_storage.py`

**问题**: `self.model_cache` 无限增长，可能 OOM。

**修复方案**: 使用 `collections.OrderedDict` 实现 LRU 缓存，设置最大缓存数量（如 10 个模型）。

---

#### M10: training.py 大量代码重复

**文件**: `backend/app/services/qlib/training/training.py`

**问题**: 获取 `evals_result` 的代码块被复制粘贴了 3 次。

**修复方案**: 提取为独立函数 `_extract_training_history(model, config)`。

---

#### M11: 金融指标使用固定收益率

**文件**: `backend/app/services/models/model_training.py`

**问题**: `_calculate_metrics` 使用固定 ±1% 收益计算夏普比率，毫无参考价值。

**修复方案**: 使用实际价格数据计算收益率，参考 `model_evaluation.py` 中的 `FinancialMetricsCalculator`。

---

### 阶段 3：Minor 级别问题（可选改进）

| 编号 | 问题 | 文件 | 修复方案 |
|------|------|------|----------|
| m1 | PositionalEncoding 重复定义 3 次 | modern_models.py, model_training.py, custom_models.py | 提取到公共模块 |
| m2 | PatchTST 用 Python 循环创建 patches | modern_models.py | 改用 torch.unfold |
| m3 | TimesBlock 中 FFT_for_Period 未使用 | modern_models.py | 集成或标注为简化版 |
| m4 | Inception_Block_V1 未使用 | modern_models.py | 删除 |
| m5 | 裸 except | training_progress.py | 改为 except Exception |
| m7 | 深度学习训练未用 DataLoader | model_training.py | 改用 DataLoader |
| m8 | 验证集整体前向可能 OOM | model_training.py | 分批验证 |
| m9 | WebSocket 无认证 | training_progress.py | 添加 token 验证 |
| m10 | 硬编码文件路径 | model_deployment_service.py | 使用 settings 配置 |
| m11 | 状态机不一致 | model_lifecycle_manager.py | 统一状态转换 |

---

## 三、执行规范

1. **代码规范**: 严格遵守 `/Users/ronghui/Documents/GitHub/willrone/docs/CODING_STANDARDS.md`
   - 函数 ≤50 行，类 ≤300 行
   - 圈复杂度 ≤10
   - 函数参数 ≤3 个
   - 完整类型注解
   - 无魔法数字

2. **Git 规范**: 每个阶段一个 commit
   - 阶段 1: `fix(critical): 修复模型训练流程阻塞性bug`
   - 阶段 2: `fix(major): 修复模型训练流程主要问题`
   - 阶段 3: `refactor(minor): 模型训练代码质量改进`

3. **验证**: 每个修复后运行基础导入测试
   ```bash
   cd /Users/ronghui/Projects/willrone/backend
   python -c "
   from app.services.qlib.training.training import _train_qlib_model
   from app.services.qlib.unified_qlib_training_engine import UnifiedQlibTrainingEngine
   from app.api.v1.training_progress import router
   print('All imports OK')
   "
   ```

---

## 四、预期成果

- 修复 3 个阻塞性 bug，使训练流程可以正常运行
- 加固模型加载安全性，防止 RCE
- 修复线程安全和内存泄漏问题
- 消除代码重复，提升可维护性
