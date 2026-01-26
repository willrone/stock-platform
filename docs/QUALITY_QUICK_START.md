# 质量加固快速开始指南

本指南帮助您快速开始使用项目的代码质量工具。

## 🚀 快速设置

### 1. 安装Pre-commit Hooks（推荐）

```bash
# 安装pre-commit
pip install pre-commit

# 安装hooks
pre-commit install

# 手动运行所有文件检查
pre-commit run --all-files
```

### 2. 安装后端开发依赖

```bash
cd backend
pip install -r requirements.txt
```

### 3. 安装前端开发依赖

```bash
cd frontend
npm install
```

## 📝 日常使用

### 代码提交前检查

Pre-commit hooks会自动运行，但您也可以手动运行：

```bash
# 运行所有检查
pre-commit run --all-files

# 运行特定hook
pre-commit run black --all-files
pre-commit run eslint --all-files
```

### 代码格式化

```bash
# 后端
cd backend
black app/
isort app/

# 前端
cd frontend
npm run format
```

### 代码检查

```bash
# 后端
cd backend
flake8 app/
mypy app/ --ignore-missing-imports

# 前端
cd frontend
npm run lint
npm run type-check
```

### 运行测试

```bash
# 使用脚本（推荐）
./scripts/run-tests.sh

# 或手动运行
# 后端
cd backend && pytest tests/

# 前端
cd frontend && npm test
```

### 生成质量报告

```bash
./scripts/generate-reports.sh
```

报告将生成在 `quality-reports/` 目录。

## 🔧 常用命令

### 后端

```bash
# 代码质量检查
./scripts/check-code-quality.sh

# 格式化代码
black app/
isort app/

# 运行测试
pytest tests/ -v

# 测试覆盖率
pytest tests/ --cov=app --cov-report=html

# 安全扫描
bandit -r app/
safety check
```

### 前端

```bash
# 代码质量检查
npm run quality:check

# 自动修复
npm run quality:fix

# 运行测试
npm test

# 测试覆盖率
npm run test:coverage

# 安全审计
npm audit
```

## 📊 查看报告

### 测试覆盖率

```bash
# 后端
open backend/htmlcov/index.html

# 前端
open frontend/coverage/index.html
```

### 安全扫描报告

```bash
# 后端Bandit报告
cat backend/bandit-report.json

# 前端npm审计
cd frontend && npm audit
```

## ⚙️ IDE配置

### VS Code

安装推荐扩展：
- Python: Python, Pylance
- TypeScript: ESLint, Prettier
- 通用: EditorConfig

### 设置

`.vscode/settings.json`:
```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true
  },
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "eslint.validate": ["javascript", "javascriptreact", "typescript", "typescriptreact"]
}
```

## 🐛 常见问题

### Pre-commit失败

如果pre-commit检查失败，工具通常会尝试自动修复。如果无法自动修复：

1. 查看错误信息
2. 手动运行对应的工具修复
3. 重新提交

### 测试失败

```bash
# 查看详细错误信息
pytest tests/ -v

# 运行特定测试
pytest tests/test_specific.py -v

# 使用pdb调试
pytest tests/ --pdb
```

### 类型检查错误

```bash
# 查看详细错误
mypy app/ --ignore-missing-imports

# 如果某些模块无法检查，可以添加类型忽略
# type: ignore
```

## 📚 更多信息

- [代码规范指南](./CODE_STANDARDS.md)
- [测试指南](./TESTING_GUIDE.md)
- [质量加固方案](./QUALITY_IMPROVEMENT_PLAN.md)

## 🆘 获取帮助

如果遇到问题：
1. 查看相关文档
2. 检查工具配置
3. 提交Issue

---

**最后更新**: 2026-01-26
