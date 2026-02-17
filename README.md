# 股票预测平台

基于AI的智能股票预测和任务管理系统，采用现代化的微服务架构，集成先进的机器学习模型和实时数据处理能力。

## 🚀 项目概述

本项目是一个完整的股票预测解决方案，包含：

- **后端服务**: 基于FastAPI的高性能API服务
- **前端应用**: 基于Next.js + React的现代化Web界面
- **机器学习**: 集成Qlib量化框架和多种深度学习模型
- **数据管理**: 支持本地和远程数据源的统一管理
- **实时通信**: WebSocket支持的任务状态实时更新

## 🏗️ 技术架构

### 后端技术栈
- **框架**: FastAPI + SQLAlchemy
- **数据库**: SQLite (开发) / PostgreSQL (生产)
- **机器学习**: Qlib + PyTorch + XGBoost
- **数据存储**: Parquet文件格式
- **异步处理**: Celery + Redis
- **API文档**: 自动生成的OpenAPI文档

### 前端技术栈
- **框架**: Next.js 14 (App Router)
- **UI库**: Ant Design Pro
- **状态管理**: Zustand
- **图表库**: ECharts + TradingView
- **实时通信**: Socket.IO Client
- **类型安全**: TypeScript

### 机器学习模型
- **传统模型**: XGBoost, LSTM
- **现代模型**: Transformer, TimesNet, PatchTST, Informer
- **集成学习**: 模型ensemble和在线学习
- **回测引擎**: Vectorbt集成

## 📁 项目结构

```
stock-prediction-platform/
├── backend/                    # Python后端服务
│   ├── app/                   # 应用核心代码
│   │   ├── api/              # API路由
│   │   ├── services/         # 业务逻辑服务
│   │   ├── models/           # 数据模型
│   │   ├── middleware/       # 中间件
│   │   └── core/            # 核心配置
│   ├── tests/               # 测试文件
│   ├── data/               # 数据存储目录
│   └── requirements.txt    # Python依赖
├── frontend/               # React前端应用
│   ├── src/               # 源代码
│   │   ├── app/          # Next.js页面
│   │   ├── components/   # React组件
│   │   ├── services/     # API服务
│   │   └── stores/       # 状态管理
│   ├── public/           # 静态资源
│   └── package.json      # Node.js依赖
├── data/                 # 共享数据目录
├── .kiro/               # 项目规范和任务管理
│   └── specs/           # 功能规范文档
└── README.md            # 项目说明文档
```

## 🚀 快速开始

### 环境要求

- **Python**: 3.9+
- **Node.js**: 18+ (可选，仅前端需要)
- **Git**: 最新版本
- **Docker**: 29+ (仅生产环境需要)

> 开发环境和生产环境使用不同端口，可以同时运行，互不干扰。

---

### 🛠️ 开发环境（start.sh）

适合日常开发调试，支持热重载。

```bash
# 一键启动（自动创建虚拟环境、安装依赖、启动前后端）
./start.sh

# 仅启动后端
./start.sh backend-only

# 停止服务
./scripts/stop-simple.sh
```

脚本会自动完成：虚拟环境创建、依赖安装（国内镜像源）、环境变量配置、前后端启动。

访问地址：

| 服务 | 地址 |
|------|------|
| 前端界面 | http://localhost:3000 |
| 后端 API | http://localhost:8000 |
| API 文档 | http://localhost:8000/api/v1/docs |

---

### 🐳 生产环境（Docker Compose）

适合部署运行，包含 Nginx 反代、Prometheus 监控、Grafana 仪表板。

```bash
# 启动全部容器（后台运行）
sudo docker compose up -d

# 查看容器状态
sudo docker ps

# 查看日志
sudo docker compose logs -f backend
sudo docker compose logs -f frontend

# 停止全部容器
sudo docker compose down

# 重新构建镜像（代码更新后）
sudo docker compose up -d --build
```

访问地址：

| 服务 | 地址 |
|------|------|
| Nginx 入口 | http://localhost:80 |
| 前端界面 | http://localhost:3001 |
| 后端 API | http://localhost:8001 |
| Grafana | http://localhost:3002 |
| Prometheus | http://localhost:9091 |

生产端口可通过 `.env` 文件自定义：

```env
PROD_BACKEND_PORT=8001
PROD_FRONTEND_PORT=3001
PROD_GRAFANA_PORT=3002
```

---

### 🔀 两套环境并行运行

开发和生产可以同时启动，端口完全隔离：

```bash
# 1. 启动开发环境 → :8000 / :3000
./start.sh

# 2. 启动生产环境 → :8001 / :3001 / :80
sudo docker compose up -d
```

---

### 手动启动（高级用户）

#### 后端

```bash
cd backend
python3 -m venv venv
source venv/bin/activate

# 安装依赖（二选一）
pip install -r requirements-minimal.txt  # 快速启动
pip install -r requirements.txt          # 完整功能

python run.py
```

#### 前端

```bash
cd frontend
npm install
npm run dev
```

## 🔧 开发指南

### 后端开发

```bash
# 运行测试
cd backend
python -m pytest

# 代码格式化
black app/
isort app/

# 类型检查
mypy app/
```

### 前端开发

```bash
# 运行测试
cd frontend
npm test

# 类型检查
npm run type-check

# 代码格式化
npm run lint

# 构建生产版本
npm run build
```

## 📊 功能特性

### ✅ 已实现功能

- [x] **项目基础架构**: 完整的前后端项目搭建
- [x] **数据服务层**: 支持本地和远程数据源
- [x] **技术指标计算**: MA、RSI、MACD、布林带等
- [x] **数据库存储**: SQLite数据库和Parquet文件管理
- [x] **任务管理**: 完整的任务生命周期管理
- [x] **机器学习**: 多种模型训练和评估
- [x] **预测引擎**: 多时间维度预测和风险评估
- [x] **API网关**: 完整的RESTful API和文档
- [x] **前端架构**: React应用和状态管理

### 🚧 开发中功能

- [ ] **任务管理界面**: 任务创建和监控界面
- [ ] **数据可视化**: 图表和技术指标展示
- [ ] **回测分析**: 策略回测和性能分析
- [ ] **系统监控**: 实时状态监控和告警

### 🔮 计划功能

- [ ] **用户认证**: 多用户支持和权限管理
- [ ] **模型市场**: 预训练模型分享和下载
- [ ] **实时交易**: 模拟交易和实盘对接
- [ ] **移动端**: React Native移动应用

## 🧪 测试

项目包含完整的测试套件：

- **单元测试**: 核心业务逻辑测试
- **属性测试**: 基于Property-Based Testing的正确性验证
- **集成测试**: API端到端测试
- **前端测试**: React组件和服务测试

```bash
# 运行所有测试
cd backend && python -m pytest
cd frontend && npm test
```

## 📈 性能优化

- **后端**: 异步处理、数据库连接池、缓存策略
- **前端**: 代码分割、懒加载、状态优化
- **数据**: Parquet格式、增量更新、并行计算
- **模型**: GPU加速、模型量化、批处理推理

## 🔒 安全考虑

- **API安全**: 请求限流、输入验证、错误处理
- **数据安全**: 敏感信息加密、访问控制
- **前端安全**: XSS防护、CSRF保护
- **部署安全**: HTTPS、环境变量管理

## 📝 开发规范

- **代码风格**: 遵循PEP8 (Python) 和ESLint (JavaScript)
- **提交规范**: 使用Conventional Commits格式
- **分支策略**: Git Flow工作流
- **文档要求**: 代码注释和API文档

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- **项目维护者**: [Your Name]
- **邮箱**: [your.email@example.com]
- **项目链接**: [https://github.com/yourusername/stock-prediction-platform]

## 🙏 致谢

感谢以下开源项目的支持：

- [Qlib](https://github.com/microsoft/qlib) - 量化投资平台
- [FastAPI](https://fastapi.tiangolo.com/) - 现代Python Web框架
- [Next.js](https://nextjs.org/) - React生产框架
- [Ant Design](https://ant.design/) - 企业级UI设计语言

---

**注意**: 本项目仅用于学习和研究目的，不构成投资建议。投资有风险，入市需谨慎。
## 📂 项目目录规范

**为了保持项目整洁，请严格遵守以下目录存放规则：**

### 1. 文档归档 (`docs/`)
- 所有项目相关的 `.md` 文档必须归档至 `docs/` 下的对应分类子目录中。
- `docs/README.md` 是所有文档的索引。
- 分类包括：`guides/` (指南), `mobile/` (移动端优化), `backtest/` (回测), `qlib/` (模型算法), `mlops/` (运维管理), `quality/` (质量规范), `reports/` (阶段总结), `fixes/` (修复记录)。

### 2. 测试与工具脚本 (`tests/`)
- 严禁在根目录存放临时的 `.py`, `.js`, `.sh` 等脚本。
- `tests/integration/`: 用于集成测试和数据验证。
- `tests/scripts/`: 用于任务管理、状态修复等实用工具。
- `tests/manual/`: 用于手动验证 API 或通信的工具。

### 3. 核心开发目录
- `backend/`: FastAPI 后端代码。
- `frontend/`: Next.js 前端代码。
- `back_test_data_service/`: 独立回测数据服务。

### 4. 系统环境与配置
- `nginx/`, `monitoring/`, `systemd/`, `scripts/`: 分别存放代理、监控、服务和系统安装脚本。

---
*以后新增任何文件，请根据上述规则放入对应目录。*
