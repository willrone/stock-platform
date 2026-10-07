# 股票平台代码架构评审报告

> **评审日期**: 2026-07-29
> **评审范围**: 后端 (FastAPI)、前端 (Next.js)、数据服务 (Flask)、Docker/部署、用户认证系统
> **代码总量**: ~162,000 行 Python + ~122 个 TypeScript/TSX 源文件

---

## 1. 摘要

**总体评价**: 一个功能全面的量化交易平台，后端回测引擎和 Qlib 集成深度扎实，但架构层面存在**安全漏洞、模块边界模糊、代码组织混乱**等问题，特别是在新引入的认证系统中尤为突出。

**健康度评分**: **B-** (B 减)

| 维度 | 评分 | 说明 |
|------|------|------|
| 安全性 | **C** | JWT 密钥硬编码兜底、API 无认证保护、明文 Token 存 localStorage |
| 可维护性 | **B-** | 部分模块超大 (>1000行)，双框架 (FastAPI/Flask)，双错误系统 |
| 可测试性 | **B** | 测试覆盖较全 (有 unit/integration/property 测试)，但 CI 依赖特殊兜底分支 |
| 可扩展性 | **B** | 服务层级划分清晰，但依赖注入不彻底 |
| 部署质量 | **B** | Docker 多阶段构建、healthcheck 齐全，但 env 管理和 Nginx 配置有隐患 |

---

## 2. 发现的问题

### 🔴 Critical (严重 — 应立即修复)

#### C1. JWT 密钥硬编码兜底，生产环境使用不安全默认值

- **位置**: `backend/app/core/security.py` 第 51-52 行、第 59-60 行
```python
secret = getattr(settings, "JWT_SECRET", "dev-secret-change-in-production")
```
- **影响**: 如果 `settings.JWT_SECRET` 未配置（当前设置中没有此字段），所有 JWT Token 使用公开字面量签名，任何人都可以伪造 Token。
- **建议**: 
  1. 在 `Settings` 模型中添加 `JWT_SECRET: str` 字段
  2. 在 `model_validator` 中增加 `assert JWT_SECRET != "dev-secret-change-in-production"` 校验
  3. 生产环境通过 `.env` 设置随机 256-bit 密钥

#### C2. API 路由未统一应用认证中间件，存在无认证访问路径

- **位置**: 
  - `backend/app/api/v1/api.py` — 引入了 auth.router，但未对其余 22 个路由模块施加 auth 保护
  - `backend/app/api/v1/dependencies.py` 第 28-50 行 — `get_current_user()` 默认回退到 `DEFAULT_USER_ID`
- **影响**: 所有业务 API (回测/数据/模型/预测等) 均可通过缺失 Authorization 头或 `X-User-ID: admin` 绕过认证。新增的 auth 系统实际上没有保护任何资源。
- **建议**:
  1. 在 `api.py` 中通过 `include_router(..., dependencies=[Depends(get_current_user)])` 统一施加认证
  2. 移除 `get_current_user()` 中的 `DEFAULT_USER_ID` 兜底逻辑，改为抛出 401
  3. 静态资源/健康检查路径单独放行

#### C3. 前端 Token 存储命名不一致，认证拦截器失效

- **位置**: 
  - `frontend/src/app/login/page.tsx` 第 32 行: `localStorage.setItem('token', data.access_token)`
  - `frontend/src/services/api.ts` 第 59 行: `const token = localStorage.getItem('auth_token')`
- **影响**: 登录页面将 Token 写入 `token` 键，但 API 拦截器读取 `auth_token` 键，导致所有经过 `api.ts` 发出的请求均不携带 Token，拦截器形同虚设。
- **建议**: 统一使用同一个键名（如 `access_token`），或抽离为常量模块。

#### C4. Tushare API Token 明文硬编码在多处 .env 文件

- **位置**: 
  - `/.env` 第 10 行: `TUSHARE_TOKEN=3bb09d7c81ac1f83a90e57b505626391739a93bd02c717bdcb987da4`
  - `/back_test_data_service/.env` 第 1 行: `TUSHARE_TOKEN=...`
- **影响**: API Token 是敏感凭证。尽管 .env 通常被 gitignore，但 .env 文件可能被意外提交、备份泄露或共享。违背最小权限和安全第一原则。
- **建议**:
  1. 使用 secret 管理工具（如 Vault、Docker secrets 或 1Password CLI）
  2. 在 Docker Compose 中通过 `secrets:` 挂载，而非 env_file
  3. 对 `Dockerfile` 中的 `COPY . .` 增加 `.dockerignore` 排除 .env

#### C5. 双错误处理体系共存，使用混乱

- **位置**: 
  - `backend/app/core/errors.py` — 定义 AppError、ValidationError、DomainError、InfraError、UserFacingError
  - `backend/app/core/error_handler.py` — 定义 BaseError、ErrorType、ErrorSeverity、ErrorRecoveryManager
- **影响**: 两个模块定义了重叠的错误类型体系。前者以 `AppError` 为基类，后者以 `BaseError` 为基类。开发者在编写新代码时无法确定该继承哪个基类。部分代码混用两个系统。
- **建议**: 合并为一个错误层次体系，推荐保留 `error_handler.py` 的 ErrorType/ErrorSeverity 体系（更细粒度），废弃 `errors.py`。

---

### 🟡 Warning (警告 — 建议尽快修复)

#### W1. dependencies.py 是一个 1163 行的"万能模块"

- **位置**: `backend/app/api/v1/dependencies.py` (1163 行)
- **内容混杂**:
  - 用户认证依赖 (get_current_user)
  - 任务队列管理器全局实例
  - 任务执行函数 (execute_prediction_task_simple, execute_backtest_task_simple, execute_qlib_precompute_task_simple)
  - 配置规范化 (正常该属于 service 层)
  - Repository 工厂函数
- **影响**: 严重违反单一职责原则。`dependencies.py` 混入了本应在 `services/tasks/` 中的进程池执行函数，导致这个文件同时做依赖注入、业务逻辑、配置转换三件事。
- **建议**: 将任务执行函数拆入 `services/tasks/task_executors.py`，配置规范化移入 `services/backtest/utils/`，保留 `dependencies.py` 仅做 FastAPI Depends 函数。

#### W2. 数据服务使用 subprocess.Popen 启动调度器

- **位置**: `back_test_data_service/main.py` 第 53-57 行
```python
service_process = subprocess.Popen(
    [sys.executable, str(service_script)],
    stdout=open('logs/data_service.log', 'a'),
    stderr=subprocess.STDOUT
)
```
- **影响**: 服务进程生命周期无法被 Docker/systemd 正确管理。日志文件句柄泄漏、僵尸进程风险、缺乏健康检查机制。
- **建议**: 使用 APScheduler 的 AsyncIOScheduler（与 Flask 集成）或者直接在一个进程中用多线程方式运行数据和 API 服务。

#### W3. Nginx proxy_pass 会错误剥离 /api/ 前缀

- **位置**: `nginx/nginx.conf` 第 62 行
```nginx
location /api/ {
    proxy_pass http://backend/;  # 这里结尾斜杠会剥离 /api/ 前缀
}
```
- **影响**: 后端 FastAPI 期望路径为 `/api/v1/...`。Nginx 将 `/api/v1/health` 转发为 `/v1/health`，导致后端 404。
- **同时**，前端 Next.js rewrites 也做了 `/api/v1` → `http://backend:8000/api/v1` 的代理，生产环境中 Nginx 和 Next.js 会做出双重代理。
- **建议**: 
  1. Nginx proxy_pass 去掉结尾斜杠: `proxy_pass http://backend;`
  2. 决定一个代理策略：要么全部通过 Nginx，要么全部通过 Next.js rewrites，不要两重。

#### W4. Auth 路由手动管理 Session 生命周期，不统一

- **位置**: `backend/app/api/v1/auth.py` 多处 `SessionLocal()`/`session.close()` 模式
- **影响**: 与其他路由使用 `get_async_session` 依赖注入的方式不一致。手动管理 session 容易因异常路径导致连接泄漏（当前使用 try/finally 避免了泄漏，但仍是一致性风险）。
- **建议**: auth 模块也统一使用 `SessionLocal` 的上下文管理器或 FastAPI Depends。至少抽取一个 `get_sync_session` 依赖。

#### W5. 迁移脚本未使用 Alembic

- **位置**: `backend/migrations/` 中全部是原始 Python 脚本，直接执行 ALTER TABLE
- **影响**: 迁移不可回滚、无依赖链、无法在多环境中复现一致的结构。当前数据库 (SQLite) 对 ALTER TABLE 支持有限，未来切到 PostgreSQL 后脚本可能不兼容。
- **建议**: 引入 Alembic 管理数据库迁移，将所有历史迁移脚本转化为 Alembic revision。

#### W6. Rate limiting 基于内存，无法跨进程

- **位置**: `backend/app/middleware/rate_limiting.py` — 使用内存 SlidingWindowCounter
- **影响**: 当部署多 worker 或多实例时，每个进程有自己的限流计数器，客户端可以轮询不同 worker 绕过限流。
- **建议**: 使用 Redis 作为集中式限流后端，或者接受单进程限流的局限性并文档化。

#### W7. Backend Dockerfile 中 pip --user 安装路径可能不可用

- **位置**: `backend/Dockerfile` 第 19 行: `RUN pip install --user -r requirements.txt`
- **影响**: `--user` 安装到 `/root/.local/lib/python3.13/site-packages`，但运行时未显式设置 `PATH` 或 `PYTHONPATH`。某些 CLI 工具或可执行文件可能 find 不到。
- **建议**: 在 builder 阶段使用 `pip install`（不带 --user）到系统目录，或使用虚拟环境，并在运行时阶段激活。

#### W8. 双框架并行 (FastAPI + Flask)

- **位置**: 
  - `backend/` — FastAPI
  - `back_test_data_service/` — Flask
- **影响**: 两个框架有完全不同的路由注册方式、中间件体系和部署范式。团队需要维护两套知识体系，增加认知负荷。数据服务的 Flask 路由缺乏 OpenAPI 支持。
- **建议**: 中长期逐步将数据服务的 API 功能迁移到 FastAPI 后端中，Flask 只保留旧版兼容。

#### W9. Frontend 将用户信息明文存入 localStorage

- **位置**: `frontend/src/app/login/page.tsx` 第 33 行: `localStorage.setItem('user', JSON.stringify(data.user))`
- **影响**: 用户邮箱、订阅等级等个人数据在浏览器中以明文存储，可被同域下的 XSS 脚本窃取。
- **建议**: Token 使用 httpOnly Cookie，用户信息尽量从 `/me` 端点每次获取，或使用内存状态管理。

---

### 🔵 Suggestion (建议 — 改进性)

#### S1. User 模型未注册到 `init_db()`

- **位置**: `backend/app/core/database.py` 第 176-178 行
```python
from app.models import backtest_detailed_models  # noqa: F401
from app.models import strategy_config_models  # noqa: F401
from app.models import task_models  # noqa: F401
```
- **影响**: `user_models.py` 中的 User 表未在此处导入，如果数据库重建，User 表不会被创建。当前只有在路由首次 import auth.py 时才间接加载 User 模型。
- **建议**: 添加 `from app.models import user_models  # noqa: F401`

#### S2. OpenAPI 未注册安全方案

- **位置**: `backend/app/main.py` — FastAPI 应用构建函数
- **影响**: Swagger UI 不会显示"Authorize"按钮，开发者无法直接在文档页面测试需要认证的 API。
- **建议**: 在 `create_application()` 中添加：
```python
from fastapi.security import HTTPBearer
...
security_scheme = HTTPBearer()
app.add_middleware(...)  # 或通过 OpenAPI 配置添加安全方案
```

#### S3. Backend pyproject.toml mypy 配置引用了不存在的 Python 版本

- **位置**: `backend/pyproject.toml` 第 27 行: `python_version = "3.11"`
- **影响**: 实际运行环境是 Python 3.13（Dockerfile 和 `.venv-py313` 均使用了 3.13），类型检查可能不准确。
- **建议**: `python_version = "3.13"` 同时更新 `.pre-commit-config.yaml` 中的 `python: python3.11`

#### S4. `health.py` 路由路径前缀与实际不匹配

- **位置**: `backend/app/api/v1/health.py` 第 11 行: `router = APIRouter(prefix="/health", ...)`
- **影响**: 在 `api.py` 中 `api_router.include_router(health.router)`，并且 `api_router` 挂载在 `settings.API_V1_PREFIX` (`/api/v1`) 下，所以健康检查路径为 `/api/v1/health`。但 Nginx 和 docker-compose healthcheck 中用的是 `/health`（没有 `/api/v1` 前缀），造成健康检查 404。
- **建议**: 统一健康检查路径，或者将 health router 直接挂载在根路径上。

#### S5. 多个路由文件超过 1000 行

- **位置**: 
  - `backend/app/api/v1/models.py` (1945 行)
  - `backend/app/api/v1/tasks.py` (1670 行)
  - `backend/app/api/v1/data.py` (1295 行)
  - `backend/app/api/v1/dependencies.py` (1163 行)
  - `backend/app/api/v1/backtest.py` (1162 行)
- **影响**: 超大文件难以导航、代码审查困难、合并冲突概率高。
- **建议**: 按功能子模块拆分路由文件（如 `models/training.py`、`models/management.py`）。

#### S6. Frontend 没有全局状态同步 auth 状态

- `useAppStore.ts` 从 `localStorage` 读取用户信息，但登录页面直接写 `localStorage` 而不刷新 store，导致其他组件无法立即感知用户状态变更。
- **建议**: login 成功后调用 `useAppStore.getState().setUser()` 同步更新 store。

#### S7. Backend service container 管理的服务太少

- 容器只管理 3 个服务，但实际应用有 backtest、prediction、task、data、model、monitoring 等数十个服务层级。大部分服务仍在使用 `直接实例化`。
- **建议**: 扩展容器的服务注册能力，或者使用成熟的 DI 框架（如 `dependency-injector`）替代手写容器。

---

## 3. 亮点

### ✅ 架构层面

1. **回测引擎设计良好** — `backtest/services/core/` + `execution/` + `analysis/` + `reporting/` 分层清晰，策略工厂模式使用得当，易于扩展新策略。
2. **Qlib 集成深度高** — `enhanced_qlib_provider.py`、`unified_qlib_training_engine.py` 等模块对 Qlib 的封装完整，支持官方复现流程。
3. **WebSocket 连接管理器** — `websocket.py` 中的 `ConnectionManager` 设计干净，支持任务订阅和系统广播。

### ✅ 测试层面

4. **测试覆盖充分** — 后端有 100+ 测试文件，包含 unit/integration/property 测试，还引入了 property-based testing。
5. **pre-commit 配置完整** — 包含 black、isort、flake8、bandit、prettier、commitizen，工程规范化做得很好。

### ✅ 部署层面

6. **Docker 多阶段构建** — 后端和前端 Dockerfile 均使用多阶段构建，合理利用缓存层，镜像尺寸控制良好。
7. **Healthcheck 配置齐全** — 所有服务都配置了 healthcheck，Docker Compose 使用 `condition: service_healthy` 控制启动顺序。
8. **监控系统集成** — Prometheus + Grafana + AlertManager 配置完备。

### ✅ 代码质量

9. **错误恢复机制** — `error_handler.py` 的 `ErrorRecoveryManager` 和结构化日志体系设计先进，具备生产级别的事故响应能力。
10. **数据库连接池优化** — `database.py` 对 SQLite 的 WAL 模式、超时、重试机制做了充分考虑。

---

## 4. 架构建议（中长期）

### 4.1 渐进式统一认证体系

1. 在 `config.py` 中新增 `AUTH_ENABLED: bool = True`，允许开发环境关闭认证
2. 实现 FastAPI `HTTPBearer` 中间件作为全局安全依赖
3. 将认证 DTO (RegisterRequest/LoginRequest) 从 `auth.py` 提取到独立的 `schemas/auth_schemas.py`
4. 实现 Token 刷新机制（目前 Token 过期后只能重新登录）
5. 考虑使用 httpOnly Cookie 替代 Bearer header + localStorage

### 4.2 数据库迁移策略

引入 Alembic 并建立迁移基线：
```bash
alembic init alembic
alembic revision --autogenerate -m "initial"
# 将现有手动迁移脚本转为 Alembic revision
```

### 4.3 统一代理策略

选择**一种**代理策略，避免 Nginx + Next.js rewrites 两层代理：
- **推荐选项**: Next.js 仅为 SSR/静态资源服务，所有 API 调用走 Nginx → Backend
- 前端 `api.ts` 的 `baseURL` 应为相对路径（当前已配 `/api/v1`），由 Nginx 统一转发

### 4.4 拆解巨型文件

优先拆解：`dependencies.py` (1163行) → `models.py` (1945行) → `tasks.py` (1670行)。建议按功能模块拆分，每个文件不超过 300-500 行。

### 4.5 统一数据服务到 FastAPI

将 `back_test_data_service/` 的数据获取和 DAO 能力引入 FastAPI 后端，废弃 Flask 服务。或者至少引入 OpenAPI 支持 (Flask + flasgger 或 APISpec)。

### 4.6 Secret 管理

- 使用 `python-decouple` 或 `python-dotenv-secure`
- 敏感 token 通过 Docker secrets 或环境变量注入，不随代码仓库分发
- 为 `.env` 文件建立严格的访问控制 (600 权限)

---

## 5. 修改优先级

| 优先级 | 编号 | 问题 | 预计工时 |
|--------|------|------|----------|
| **P0 - 立即** | C1 | JWT 密钥硬编码兜底 | 30 min |
| **P0 - 立即** | C2 | API 无认证保护 | 1-2 h |
| **P0 - 立即** | C3 | 前端 Token 命名不一致 | 10 min |
| **P0 - 立即** | C4 | Tushare Token 明文存储 | 30 min |
| **P0 - 立即** | S1 | User 模型未注册到 init_db | 5 min |
| **P1 - 尽快** | C5 | 双错误处理体系合并 | 2-4 h |
| **P1 - 尽快** | W1 | dependencies.py 拆解 | 2-3 h |
| **P1 - 尽快** | W4 | Nginx proxy_pass 前缀问题 | 30 min |
| **P1 - 尽快** | S4 | health 路由路径不匹配 | 15 min |
| **P2 - 本周** | W2 | 数据服务 subprocess Popen | 1-2 h |
| **P2 - 本周** | W5 | Alembic 迁移 | 3-4 h |
| **P2 - 本周** | S2 | OpenAPI 安全方案 | 1 h |
| **P2 - 本周** | S6 | Frontend auth 状态同步 | 1 h |
| **P3 - 优化期** | W3 | Auth session 管理统一 | 1-2 h |
| **P3 - 优化期** | W7 | Docker pip --user 路径 | 30 min |
| **P3 - 优化期** | W8 | Flask → FastAPI 迁移 | 2-3 天 |
| **P3 - 优化期** | W9 | Token 存储安全性 | 2-3 h |
| **P4 - 中长期** | W6 | Redis 集中限流 | 1-2 天 |
| **P4 - 中长期** | S5 | 大文件拆分 | 2-3 天 |
| **P4 - 中长期** | S7 | 扩展 DI 容器 | 2-3 天 |

---

## 统计信息

| 指标 | 数值 |
|------|------|
| 后端 Python 源文件 | ~100,916 行 (backend/app/) |
| 总 Python 代码 (含脚本/测试) | ~162,016 行 |
| 前端 TS/TSX 源文件 | 122 个 |
| 后端测试文件 | 80+ (unit/integration/property) |
| Dockerfile 数量 | 4 (backend/frontend/data-service + dev) |
| 配置文件 (.env/.yml/.yaml) | 15+ |
| 路由模块 (API v1) | 22 个 |
| 服务模块 | ~60 个 |
| 数据库迁移脚本 | 5 个 (均手动) |
