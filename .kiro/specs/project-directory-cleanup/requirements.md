# 需求文档：项目目录清理系统

## 介绍

本功能旨在系统地清理项目目录结构，解决当前代码仓库中存在的严重目录混乱问题。项目根目录和 backend/ 目录中散落着大量临时文件、重复文档和 macOS 元数据文件，违反了项目 README.md 中明确规定的目录规范。

清理系统将提供自动化工具和规范流程，确保：
- 所有文档按分类归档到 docs/ 目录
- 测试和工具脚本放置在 tests/ 目录
- macOS 元数据文件被正确忽略
- 重复和临时文件被识别并清理
- 建立持续维护机制防止未来混乱

## 术语表

- **Project_Root**: 项目根目录，包含 backend/, frontend/, docs/ 等顶层目录
- **Metadata_Files**: macOS 系统生成的 `._*` 格式元数据文件
- **Temporary_Documents**: 临时性的 Markdown 文档，如优化报告、进度跟踪、修复记录等
- **Duplicate_Files**: 文件名包含 " 2" 后缀的重复文件
- **Cleanup_Tool**: 自动化清理脚本，用于扫描、分类和移动文件
- **Archive_Directory**: docs/ 目录下的分类子目录，用于归档文档
- **Gitignore_File**: .gitignore 配置文件，定义 Git 应忽略的文件模式
- **Documentation_Index**: docs/README.md 文件，作为所有文档的索引和导航
- **Quality_Check**: 质量检查工具，验证目录结构符合项目规范

## 需求

### 需求 1：macOS 元数据文件处理

**用户故事：** 作为开发者，我希望 macOS 元数据文件不被提交到 Git 仓库，这样可以保持仓库整洁并避免跨平台协作问题。

#### 验收标准

1. THE Gitignore_File SHALL 包含 `._*` 模式以忽略所有 macOS 元数据文件
2. WHEN Cleanup_Tool 执行时，THE Cleanup_Tool SHALL 识别并列出所有现有的 Metadata_Files
3. WHEN 用户确认删除时，THE Cleanup_Tool SHALL 删除所有识别出的 Metadata_Files
4. THE Cleanup_Tool SHALL 生成删除报告，包含已删除文件的数量和路径列表
5. WHEN Git 状态检查时，THE Git SHALL 不显示任何 `._*` 文件为待提交状态

### 需求 2：临时文档归档

**用户故事：** 作为项目维护者，我希望所有临时文档被归档到正确的分类目录，这样可以方便查找历史记录并保持项目根目录整洁。

#### 验收标准

1. THE Cleanup_Tool SHALL 扫描 Project_Root 和 backend/ 目录中的所有 Markdown 文件
2. WHEN 发现临时文档时，THE Cleanup_Tool SHALL 根据文件名模式自动分类文档类型
3. THE Cleanup_Tool SHALL 支持以下文档分类：
   - PHASE*_*.md → docs/reports/optimization/
   - OPT*_*.md → docs/reports/optimization/
   - OPTIMIZATION_*.md → docs/reports/optimization/
   - SIGNAL_FIX*.md → docs/fixes/signals/
   - BACKTEST_*.md → docs/backtest/analysis/
   - *_PROGRESS*.md → docs/reports/progress/
   - *_REPORT*.md → docs/reports/general/
4. WHEN 目标目录不存在时，THE Cleanup_Tool SHALL 自动创建所需的子目录
5. THE Cleanup_Tool SHALL 移动文件到目标目录并保留原始文件名
6. THE Cleanup_Tool SHALL 生成归档报告，列出每个文件的源路径和目标路径

### 需求 3：重复文件处理

**用户故事：** 作为开发者，我希望识别并处理重复文件，这样可以避免维护多个版本并减少混淆。

#### 验收标准

1. THE Cleanup_Tool SHALL 识别所有文件名包含 " 2" 后缀的文件
2. WHEN 发现重复文件时，THE Cleanup_Tool SHALL 比较原始文件和重复文件的内容
3. IF 两个文件内容相同，THEN THE Cleanup_Tool SHALL 标记重复文件为可安全删除
4. IF 两个文件内容不同，THEN THE Cleanup_Tool SHALL 标记为需要人工审查
5. THE Cleanup_Tool SHALL 生成重复文件报告，包含文件对比结果和建议操作
6. WHEN 用户确认删除时，THE Cleanup_Tool SHALL 仅删除标记为可安全删除的文件

### 需求 4：临时脚本整理

**用户故事：** 作为开发者，我希望临时测试脚本被移动到 tests/ 目录，这样可以保持根目录整洁并方便测试管理。

#### 验收标准

1. THE Cleanup_Tool SHALL 扫描 Project_Root 和 backend/ 目录中的所有 Python 和 Shell 脚本
2. THE Cleanup_Tool SHALL 识别以下临时脚本模式：
   - test_*.py → tests/scripts/
   - debug_*.py → tests/scripts/
   - check_*.py → tests/scripts/
   - create_*.py → tests/scripts/
   - monitor_*.sh → tests/scripts/
   - fix_*.py → tests/scripts/
3. THE Cleanup_Tool SHALL 排除核心脚本文件（run.py, setup.py, manage.py 等）
4. WHEN 移动脚本时，THE Cleanup_Tool SHALL 保持文件的可执行权限
5. THE Cleanup_Tool SHALL 生成脚本整理报告，列出移动的脚本及其新位置

### 需求 5：临时数据文件清理

**用户故事：** 作为开发者，我希望临时数据文件被识别并清理，这样可以减少仓库大小并避免提交不必要的文件。

#### 验收标准

1. THE Cleanup_Tool SHALL 识别以下临时数据文件：
   - *.txt（基准测试结果、任务 ID 等）
   - *.json（测试结果、配置快照等）
   - profile_result.txt
   - bench_*.txt
   - *_result.txt
   - phase*_task_id.txt
2. THE Cleanup_Tool SHALL 排除重要配置文件（requirements.txt, package.json 等）
3. WHEN 发现临时数据文件时，THE Cleanup_Tool SHALL 标记为可删除并请求用户确认
4. THE Cleanup_Tool SHALL 生成临时文件清理报告，包含文件大小和总节省空间

### 需求 6：文档索引更新

**用户故事：** 作为项目维护者，我希望文档索引自动更新，这样可以方便团队成员查找归档的文档。

#### 验收标准

1. WHEN 文档被归档后，THE Cleanup_Tool SHALL 更新 Documentation_Index
2. THE Documentation_Index SHALL 按分类列出所有归档文档的链接
3. THE Documentation_Index SHALL 包含每个文档的简短描述（从文件名推断）
4. THE Documentation_Index SHALL 按时间倒序排列同类文档（最新的在前）
5. THE Documentation_Index SHALL 包含归档日期和原始位置信息

### 需求 7：清理预览和确认

**用户故事：** 作为开发者，我希望在执行清理前预览所有变更，这样可以避免误删重要文件。

#### 验收标准

1. THE Cleanup_Tool SHALL 提供 `--dry-run` 模式，仅显示计划的操作而不执行
2. WHEN 在 dry-run 模式下运行时，THE Cleanup_Tool SHALL 生成详细的变更预览报告
3. THE 变更预览报告 SHALL 包含以下信息：
   - 将被删除的文件列表
   - 将被移动的文件及其目标位置
   - 将被创建的新目录
   - 预计节省的磁盘空间
4. THE Cleanup_Tool SHALL 在交互模式下逐类别请求用户确认
5. THE Cleanup_Tool SHALL 支持 `--yes` 标志以跳过所有确认（用于自动化）

### 需求 8：清理报告生成

**用户故事：** 作为项目维护者，我希望获得详细的清理报告，这样可以审查清理结果并记录项目历史。

#### 验收标准

1. WHEN 清理完成后，THE Cleanup_Tool SHALL 生成综合清理报告
2. THE 清理报告 SHALL 包含以下部分：
   - 执行摘要（处理的文件总数、删除数、移动数）
   - 按类别分组的详细操作列表
   - 遇到的错误和警告
   - 需要人工审查的项目
   - 清理前后的目录统计对比
3. THE 清理报告 SHALL 保存为 Markdown 格式到 docs/reports/cleanup/
4. THE 清理报告 SHALL 包含执行时间戳和执行者信息
5. THE Cleanup_Tool SHALL 在控制台输出报告摘要

### 需求 9：Git 集成

**用户故事：** 作为开发者，我希望清理工具与 Git 集成，这样可以安全地跟踪文件移动并避免丢失历史记录。

#### 验收标准

1. THE Cleanup_Tool SHALL 使用 `git mv` 命令移动已跟踪的文件
2. THE Cleanup_Tool SHALL 使用 `git rm` 命令删除已跟踪的文件
3. WHEN 文件未被 Git 跟踪时，THE Cleanup_Tool SHALL 使用标准文件系统操作
4. THE Cleanup_Tool SHALL 在操作前检查 Git 工作目录状态
5. IF Git 工作目录有未提交的更改，THEN THE Cleanup_Tool SHALL 警告用户并建议先提交或暂存
6. THE Cleanup_Tool SHALL 提供 `--no-git` 标志以禁用 Git 集成

### 需求 10：持续维护机制

**用户故事：** 作为项目维护者，我希望建立持续维护机制，这样可以防止未来目录再次混乱。

#### 验收标准

1. THE Quality_Check SHALL 作为 Git pre-commit hook 运行
2. WHEN 提交包含违规文件时，THE Quality_Check SHALL 阻止提交并显示错误信息
3. THE Quality_Check SHALL 检测以下违规：
   - 根目录中的 .md 文件（除 README.md, CONTRIBUTING.md 等核心文档）
   - 根目录中的临时脚本（test_*.py, debug_*.py 等）
   - backend/ 目录中的 .md 文件
   - 未被 .gitignore 忽略的 `._*` 文件
4. THE Quality_Check SHALL 提供 `--fix` 选项自动修复简单违规
5. THE Quality_Check SHALL 生成违规报告并建议正确的文件位置

### 需求 11：配置和自定义

**用户故事：** 作为项目维护者，我希望清理工具可配置，这样可以适应项目特定的需求和规则。

#### 验收标准

1. THE Cleanup_Tool SHALL 支持配置文件（.cleanup.yaml）定义清理规则
2. THE 配置文件 SHALL 允许定义自定义文件模式和目标目录映射
3. THE 配置文件 SHALL 允许定义排除模式（不应被清理的文件）
4. THE 配置文件 SHALL 允许定义临时文件的保留期限（例如：30天）
5. THE Cleanup_Tool SHALL 在配置文件不存在时使用合理的默认规则
6. THE Cleanup_Tool SHALL 验证配置文件格式并报告配置错误

### 需求 12：安全和回滚

**用户故事：** 作为开发者，我希望清理操作可以安全回滚，这样可以在出错时恢复文件。

#### 验收标准

1. WHEN 执行清理前，THE Cleanup_Tool SHALL 创建备份清单文件
2. THE 备份清单 SHALL 记录所有将被修改的文件的原始位置和内容哈希
3. THE Cleanup_Tool SHALL 提供 `--backup` 选项创建实际文件备份到 .cleanup_backup/
4. THE Cleanup_Tool SHALL 提供 `rollback` 命令根据备份清单恢复文件
5. WHEN 回滚时，THE Cleanup_Tool SHALL 验证目标位置的文件未被修改
6. IF 回滚冲突发生，THEN THE Cleanup_Tool SHALL 报告冲突并请求用户决策

