# stock-platform 启动方式总览

这是当前仓库的统一启动入口说明。目标只有两个：

- 本地开发：优先用 tmux + 原生进程
- 本机常驻：优先用 systemd

不要再把历史 Docker/简单脚本当作默认开发入口。

## 1. 开发态（默认）

适用场景：

- 本地开发
- 联调
- 查日志
- 临时重启 backend / frontend

默认端口：

- backend: 127.0.0.1:18082
- frontend: 127.0.0.1:13000
- data-api: 127.0.0.1:5002
- metrics: 127.0.0.1:19090

四服务拓扑、健康检查和常见问题见：`docs/guides/LOCAL_DEVELOPMENT_SERVICES.md`。

推荐命令：

```bash
make doctor
make setup
make dev
./status.sh
./stop.sh
```

日志：

```bash
./scripts/logs.sh backend
./scripts/logs.sh frontend
./scripts/logs.sh data-api
./scripts/logs.sh worker
./scripts/logs.sh all
```

## 2. 生产态（当前推荐的常驻托管路径）

适用场景：

- 这台机器本地常驻运行
- 需要开机自启
- 希望通过 systemctl 统一管理

前提：

- backend/.venv 已准备好
- frontend 依赖已安装
- systemd 可用

推荐顺序：

```bash
make setup
make prod-build
sudo ./scripts/install-systemd.sh
./scripts/prod-up.sh
./scripts/prod-status.sh
```

常用命令：

```bash
make prod-build
make prod-up
make prod-down
make prod-status
```

说明：

- backend 服务读取 `backend/.env`
- frontend 服务读取 `frontend/.env.local`
- frontend systemd 运行的是 `npm run start`，所以在 `prod-up` 前要先完成 `prod-build`
- 当前 worker unit 仍是占位服务，因为任务生命周期还主要由 backend 承担
- 当前 production 已按“局域网可访问”配置：frontend 监听 `0.0.0.0:13000`，backend 监听 `0.0.0.0:18082`
- 推荐局域网入口是 frontend：`http://本机局域网IP:13000`

## 3. 历史脚本状态

以下脚本仍保留，但已经标记为 deprecated：

- `scripts/simple-start.sh`
- `scripts/stop-simple.sh`
- `scripts/quick-start.sh`
- `scripts/start.sh`
- `scripts/stop.sh`

它们现在只作为兼容层：

- 老的开发入口会转发到新的 `./start.sh` / `./stop.sh`
- 老的生产入口会提示改用 `prod-build` / `install-systemd.sh` / `prod-up.sh`

## 4. 一眼看懂该用哪个入口

- 我要开发：`make dev`
- 我要看状态：`./status.sh`
- 我要停开发环境：`./stop.sh`
- 我要准备常驻运行：`make prod-build`
- 我要安装 systemd：`sudo ./scripts/install-systemd.sh`
- 我要启动常驻服务：`./scripts/prod-up.sh`
- 我要检查常驻服务：`./scripts/prod-status.sh`

## 5. 当前不推荐的路径

以下不是当前默认方案：

- 旧 Docker compose 启动链路
- 杀端口式“万能启动脚本”
- 默认回退到 8000 / 3000

原因：

- 这台机器是多项目并存环境
- 8000/3000 容易撞别的服务
- 原生进程 + tmux / systemd 更容易定位问题
