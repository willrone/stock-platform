# 🚀 股票预测平台 - 快速启动指南

## 最简单的启动方式 ⭐

```bash
# 1. 进入项目目录
cd stock-prediction-platform

# 2. 一键启动
./start.sh

# 3. 访问应用
# 前端: http://localhost:3000
# API: http://localhost:8000/api/v1/docs
```

## 启动选项

| 命令 | 说明 |
|------|------|
| `./start.sh` | 启动前端+后端（推荐）⭐ |
| `./start.sh backend-only` | 仅启动后端 |
| `./stop.sh` | 停止所有服务 |

## ✅ 成功启动的标志

看到以下信息说明启动成功：

```
[成功] 后端服务启动成功 (PID: xxxxx)
[成功] 前端服务启动成功 (PID: xxxxx)

🌐 服务访问地址：
  前端应用: http://localhost:3000
  后端API: http://localhost:8000
  API文档: http://localhost:8000/api/v1/docs
```

## 常见问题解决

### ❓ Python环境问题

```bash
# 检查Python版本（需要3.9+）
python3 --version

# 如果没有Python3
sudo apt install python3 python3-pip python3-venv  # Ubuntu/Debian
brew install python3  # macOS
```

### ❓ Node.js环境问题

```bash
# 检查Node.js版本（需要18+）
node --version

# 如果没有Node.js（可选，仅前端需要）
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs  # Ubuntu/Debian

# 或使用nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
```

### ❓ 端口被占用

```bash
# 检查端口占用
lsof -i :8000  # 后端端口
lsof -i :3000  # 前端端口

# 杀死占用进程
kill -9 <PID>
```

### ❓ 权限问题

```bash
# 给脚本执行权限
chmod +x start.sh stop.sh
chmod +x scripts/*.sh
```

### ❓ 依赖安装失败

```bash
# 使用国内源安装Python依赖
pip install -r backend/requirements-minimal.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 使用国内源安装Node.js依赖
cd frontend
npm install --registry=https://registry.npmmirror.com
```

### ❓ 服务启动失败

```bash
# 查看日志
tail -f data/logs/backend.log   # 后端日志
tail -f data/logs/frontend.log  # 前端日志

# 手动启动后端调试
cd backend
source venv/bin/activate
python run.py

# 手动启动前端调试
cd frontend
npm run dev
```

## 文件结构说明

```
stock-prediction-platform/
├── start.sh                    # 一键启动脚本
├── stop.sh                     # 一键停止脚本
├── scripts/
│   ├── simple-start.sh         # 简单启动脚本
│   ├── stop-simple.sh          # 简单停止脚本
│   └── install_deps.sh         # 依赖安装脚本
├── backend/
│   ├── venv/                   # Python虚拟环境（自动创建）
│   ├── requirements-minimal.txt # 最小化依赖
│   └── run.py                  # 后端启动文件
├── frontend/
│   ├── node_modules/           # Node.js依赖（自动创建）
│   └── package.json            # 前端配置
└── data/
    ├── backend.pid             # 后端进程ID
    ├── frontend.pid            # 前端进程ID
    └── logs/                   # 日志文件
```

## 高级选项

### 仅使用后端API

如果你只需要API服务，不需要前端界面：

```bash
./start.sh backend-only
```

### 使用完整功能

如果需要机器学习模型等完整功能：

```bash
# 使用Docker（推荐）
./scripts/quick-start.sh

# 或安装完整依赖
cd backend
pip install -r requirements.txt
```

### 开发模式

```bash
# 后端开发
cd backend
source venv/bin/activate
python run.py

# 前端开发
cd frontend
npm run dev
```

## 性能优化建议

1. **首次启动较慢**：需要下载和安装依赖，后续启动会很快
2. **内存使用**：最小化模式约占用500MB内存，完整模式约2GB
3. **网络优化**：使用国内镜像源加速依赖下载
4. **存储空间**：最小化安装约需要1GB空间

## 技术支持

如果遇到问题：

1. 查看日志文件：`data/logs/`
2. 检查进程状态：`ps aux | grep python`
3. 重新启动：`./stop.sh && ./start.sh`
4. 清理重装：删除`backend/venv`和`frontend/node_modules`后重新启动

---

**提示**：这个快速启动方式使用最小化依赖，适合快速体验和开发。如需完整的机器学习功能，建议使用Docker方式启动。