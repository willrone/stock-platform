# 本地四服务开发架构

本文记录当前仓库推荐的本地开发拓扑。默认入口是 `make dev`，由 tmux 编排原生进程；不要优先使用历史 Docker compose 或杀端口式脚本。

## 服务一览

| 服务        | tmux window |        默认监听 | 入口脚本                  | 说明                                                        |
| ----------- | ----------: | --------------: | ------------------------- | ----------------------------------------------------------- |
| Backend API |   `backend` | `0.0.0.0:18082` | `scripts/dev-backend.sh`  | FastAPI 主应用、任务生命周期、WebSocket、Prometheus metrics |
| Frontend    |  `frontend` | `0.0.0.0:13000` | `scripts/dev-frontend.sh` | Next.js 开发服务，默认连接 backend                          |
| Data API    |  `data-api` |  `0.0.0.0:5002` | `scripts/dev-data-api.sh` | 独立股票数据 API，读取共享 Parquet 数据目录                 |
| Worker      |    `worker` |          无端口 | `scripts/dev-worker.sh`   | 当前仍是占位窗口；任务调度仍随 backend 生命周期运行         |

相关默认值集中在 `scripts/env.sh`：

- `DEFAULT_DEV_BACKEND_PORT=18082`
- `DEFAULT_DEV_FRONTEND_PORT=13000`
- `DEFAULT_DEV_DATA_API_PORT=5002`
- `DEFAULT_DEV_METRICS_PORT=19090`
- `DEV_SESSION_NAME=stock-platform-dev`

## 数据流

```text
Browser
  │
  ▼
Frontend :13000
  │ NEXT_PUBLIC_API_URL / NEXT_PUBLIC_WS_URL
  ▼
Backend :18082 ──► Data API :5002 ──► data/parquet
  │
  ├─ SQLite / local data under backend/data or data/
  ├─ model artifacts under data/models
  └─ metrics on :19090 when enabled

Worker window: placeholder only; no independent queue process yet.
```

Backend 默认通过 `REMOTE_DATA_SERVICE_URL=http://127.0.0.1:5002` 访问 Data API。Data API 默认使用 `PARQUET_DATA_DIR=$PROJECT_ROOT/data/parquet`。

## 启动、状态和日志

首次准备：

```bash
make doctor
make setup
```

启动开发会话：

```bash
make dev
# 等价于 ./scripts/start-dev.sh，也可用顶层 ./start.sh 兼容入口
```

检查状态：

```bash
./status.sh
# 或 make status
```

真实运行态 smoke：

```bash
make smoke-local
# 检查 frontend /data /monitoring、backend health/data/monitoring、Data API、metrics
# 如 :19090 未单独监听，会自动回退检查 backend /metrics
```

查看日志：

```bash
./scripts/logs.sh backend
./scripts/logs.sh frontend
./scripts/logs.sh data-api
./scripts/logs.sh worker
./scripts/logs.sh all
```

停止：

```bash
./stop.sh
# 或 make stop
```

进入 tmux：

```bash
tmux attach -t stock-platform-dev
```

## 健康检查

启动后优先看这几个地址：

| 检查项           | 地址                                  |
| ---------------- | ------------------------------------- |
| 前端页面         | http://127.0.0.1:13000                |
| Backend health   | http://127.0.0.1:18082/api/v1/health  |
| Backend API docs | http://127.0.0.1:18082/api/v1/docs    |
| Data API health  | http://127.0.0.1:5002/api/data/health |
| Metrics          | http://127.0.0.1:19090/metrics        |

命令行快速验证：

```bash
curl -fsS http://127.0.0.1:18082/api/v1/health
curl -fsS http://127.0.0.1:5002/api/data/health
```

如果只想验证端口和进程，使用：

```bash
./status.sh
```

如果要验证真实运行链路，使用：

```bash
make smoke-local
```

`make smoke-local` 会检查：

- frontend `/`、`/data`、`/monitoring` 返回 2xx 且无明显 Next/React fatal marker
- backend `/api/v1/health`
- backend `/api/v1/data/status` 中 `data.is_connected=true`
- backend `/api/v1/monitoring/health` 中 `data.overall_healthy=true`
- Data API `/api/data/health` 中 `storage_available=true`
- metrics：优先 `:19090/metrics`，失败时回退到 backend `/metrics`

## 环境覆盖

可通过环境变量覆盖默认端口：

```bash
STOCK_PLATFORM_BACKEND_PORT=18083 \
STOCK_PLATFORM_FRONTEND_PORT=13001 \
STOCK_PLATFORM_DATA_API_PORT=5003 \
make dev
```

也可以分别编辑：

- `backend/.env`
- `frontend/.env.local`

注意：`scripts/env.sh` 会在缺少上述文件时按开发默认值创建，但不会主动 kill 已占用端口。如果端口被占用，启动脚本会报错并打印占用者。

## 常见问题

### 1. `make dev` 提示 tmux 会话已存在

说明开发环境已经启动。直接进入：

```bash
tmux attach -t stock-platform-dev
```

或先停掉再重启：

```bash
./stop.sh
make dev
```

### 2. 端口被占用

当前脚本会保护多项目环境，不会自动杀进程。查看占用者：

```bash
./status.sh
lsof -nP -iTCP:18082 -sTCP:LISTEN
lsof -nP -iTCP:13000 -sTCP:LISTEN
lsof -nP -iTCP:5002 -sTCP:LISTEN
```

然后手动决定是否停止对应进程，或用环境变量换端口。

### 3. 前端能打开但接口失败

检查：

1. `./status.sh` 中 backend 是否健康。
2. `frontend/.env.local` 中 `NEXT_PUBLIC_API_URL` 是否指向当前 backend。
3. `backend/.env` 中 `CORS_ORIGINS` 是否包含当前 frontend 地址。

### 4. `/api/v1/data/status` 显示数据服务不可用

检查 Data API：

```bash
curl -fsS http://127.0.0.1:5002/api/data/health
./scripts/logs.sh data-api
```

确认 `backend/.env` 中 `REMOTE_DATA_SERVICE_URL` 指向 Data API，例如：

```text
REMOTE_DATA_SERVICE_URL="http://127.0.0.1:5002"
```

### 5. Worker window 看起来没有任务输出

这是当前预期行为。`scripts/dev-worker.sh` 目前只是占位窗口，任务调度仍由 backend 承担；后续如果拆出独立 worker，再在该脚本接入真实 worker 入口。

## CI 与本地验证

后端全量测试：

```bash
cd backend
.venv-py313/bin/python -m pytest tests/ -q
```

前端质量门：

```bash
cd frontend
npm run lint
npm run type-check
npm run format:check
```

提交前至少运行与改动范围相关的最小验证；涉及启动链路时，优先再跑 `./status.sh` 和 `make smoke-local`。
