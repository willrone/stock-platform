# 贡献指南

感谢您对股票预测平台项目的贡献！

## 🚀 快速开始

### 1. 克隆和设置

```bash
git clone <your-fork>
cd stock-prediction-platform

# 设置提交模板
git config commit.template .gitmessage
```

### 2. 提交前检查

```bash
# 快速检查（推荐）
./scripts/quick-check.sh

# 完整检查（可选）
./scripts/pre-commit-check.sh
```

## 📝 提交规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<类型>(<范围>): <描述>

[可选的正文]

[可选的脚注]
```

### 类型说明

- `feat`: 新功能
- `fix`: 修复bug
- `docs`: 文档更新
- `style`: 代码格式
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建/工具

### 范围说明

- `backend`: 后端相关
- `frontend`: 前端相关
- `api`: API相关
- `ui`: 用户界面
- `db`: 数据库
- `test`: 测试
- `docs`: 文档

### 提交示例

```bash
git commit -m "feat(backend): 添加股票数据获取API"
git commit -m "fix(frontend): 修复任务列表分页问题"
git commit -m "docs: 更新README安装说明"
```

## 🔍 代码检查

### 后端 (Python)

```bash
cd backend

# 代码格式化
black app/
isort app/

# 类型检查
mypy app/

# 运行测试
python -m pytest
```

### 前端 (TypeScript)

```bash
cd frontend

# 类型检查
npm run type-check

# 代码格式化
npm run lint

# 运行测试
npm test
```

## 🚫 避免提交的文件

以下文件已在 `.gitignore` 中配置，请勿提交：

- 环境配置文件 (`.env`, `.env.local`)
- 依赖目录 (`node_modules/`, `venv/`)
- 构建产物 (`.next/`, `dist/`, `build/`)
- 缓存文件 (`__pycache__/`, `.cache/`)
- 数据库文件 (`*.db`, `*.sqlite`)
- 日志文件 (`*.log`)
- 敏感信息 (`*.key`, `*.pem`, `secrets.json`)

## 🔧 开发工作流

1. **创建分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **开发和测试**
   ```bash
   # 进行开发...
   ./scripts/quick-check.sh  # 检查代码
   ```

3. **提交更改**
   ```bash
   git add .
   git commit  # 使用模板格式
   ```

4. **推送和PR**
   ```bash
   git push origin feature/your-feature-name
   # 创建 Pull Request
   ```

## 📋 Pull Request 检查清单

- [ ] 代码遵循项目规范
- [ ] 添加了必要的测试
- [ ] 更新了相关文档
- [ ] 通过了所有检查
- [ ] 提交信息符合规范
- [ ] 没有提交敏感文件

## 🆘 常见问题

### Q: 如何设置开发环境？
A: 参考主 README.md 的"快速开始"部分

### Q: 提交时遇到类型错误怎么办？
A: 运行 `npm run type-check` 查看详细错误信息

### Q: 如何跳过某些检查？
A: 使用 `git commit --no-verify` (不推荐)

### Q: 大文件如何处理？
A: 考虑使用 Git LFS 或将其添加到 `.gitignore`

## 📞 获取帮助

- 查看 [Issues](../../issues) 了解已知问题
- 创建新 [Issue](../../issues/new) 报告问题
- 参考项目 [Wiki](../../wiki) 获取更多信息

感谢您的贡献！🎉