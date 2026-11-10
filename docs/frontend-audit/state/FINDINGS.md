# 股票平台前端审计报告 —— 状态管理 / 错误处理 / 加载态 / 空态 / 表单校验 / API 契约一致性

- 审计范围：`/Users/rone/Projects/stock-platform/frontend`（Next.js 14 App Router + TS strict + zustand + react-query v3 + axios 1.6.2）
- 参照后端：`/Users/rone/Projects/stock-platform/backend`（FastAPI，`app/api/v1/*`）
- 实机环境：前端 dev `http://localhost:3000`、后端 `http://127.0.0.1:18082`
- 账号：`ronghui`（`ronghui@example.com`，`subscription_tier=enterprise`、`is_admin=1`）
- 审计方式：静态代码阅读 + 后端 FastAPI 路由表逐条对拍 + 真实 HTTP 探测 + 浏览器实机复现
- 截图目录：`docs/frontend-audit/state/`
- **只读审计，未修改任何源码。**

> 说明：后端有速率限制（`RateLimitMiddleware`，60 次/分钟 + burst 10，按客户端 IP 计）。部分截图中的
> 「请求过于频繁，请稍后再试」正是触发该限流的真实表现，**不是**后端业务故障。后端探测使用了
> `X-Test-Environment: true` 头（该头是后端既有测试旁路，非本次新增）以获得干净信号。

---

## 0. 结论速览

共 **23 条**问题：P0 = 8、P1 = 8、P2 = 5、P3 = 2。

**最致命的一类不是「错误处理写得糙」，而是「后端返回成功，前端却判定为失败」**——
后端有一批端点（`/commerce/*`、`/trading/*`、`/billing/*`、`/auth/*`）**不套** `{success,message,data}`
信封，直接返回裸对象；而前端 `apiRequest` 的响应拦截器把「没有 `success` 字段」一律判为业务失败并
抛 `Error('请求失败')`。这导致**钱包页显示 ¥0.00、今日信号页整页报错、订阅页把 enterprise 用户显示成
FREE**——即计费与付费信号这两条商业主链路在当前代码下**完全不可用**。

---

## P0 问题清单

### P0-1　`/commerce/*`、`/trading/*` 返回裸对象，`apiRequest` 一律判失败 → 钱包 / 今日信号 / 订阅全链路瘫
- **严重度**：P0
- **文件**：`frontend/src/services/api.ts:194-198`（拦截器判定）、`frontend/src/services/api.ts:306-309`（`apiRequest.get` 二次判定）、`frontend/src/app/wallet/page.tsx:120-127`、`frontend/src/app/today-signal/page.tsx:186,208`
- **现象**：后端 `commerce.py` / `trading.py` / `billing.py` 的路由**不声明** `response_model=StandardResponse`，直接返回裸 JSON：
  - `GET /api/v1/commerce/balance` → `{"user_id":…,"balance_cents":1000,"balance_yuan":10.0,…}`（无 `success` 字段）
  - `GET /api/v1/trading/strategies` → `{"strategies":[…],"tier":…}`
  - `GET /api/v1/trading/signal?strategy=honest_42f` → `{"strategy":…,"buys":[…],"holds":[…]}`
  - `GET /api/v1/billing/subscription` → `{"subscription":null,"plan":null,"subscription_tier":"enterprise"}`
  前端拦截器 `if (!data.success) { reject(new Error(data.message || '请求失败')) }` → 裸对象 `data.success === undefined` → 必然走进 reject 分支。
- **复现步骤**：
  1. 登录（邮箱 `ronghui@example.com`）。
  2. 打开 `/wallet` → 显示「当前余额 0.00 元 / 账户状态 余额不足 / 加载数据失败」（**后端实际余额 ¥10.00**）。
  3. 打开 `/today-signal` → 整页红色 Alert「请求失败」，信号表格完全不出。
  4. 打开 `/settings/subscription` → 顶部显示「FREE 免费方案」（**后端 `subscription_tier=enterprise`**）。
  5. 浏览器内复现判定：`fetch('/api/v1/commerce/balance')` 返回 200 且 `hasSuccess=false`；
     套用 `apiRequest.get` 判定逻辑 → `REJECTED by apiRequest: 请求失败`。
- **改法**（二选一，**推荐 A**）：
  - **A（治本，统一后端信封）**：给 `commerce.py` / `trading.py` / `billing.py` / `auth.py` 的每个路由补
    `response_model=StandardResponse`，把裸返回体包进 `data`。同时把 `auth.py` 的 `AuthResponse` 场景单独处理
    （登录/注册/刷新是前端唯一直接读 `data.access_token` 的地方，见 P0-2）。
  - **B（治标，前端兼容）**：在 `api.ts` 拦截器里改成「有 `success` 字段才按业务失败判定；无 `success` 字段
    且 HTTP 2xx → 视为成功，把整个 body 当 `data`」，并让 `apiRequest.get` 同步放宽。
  - 无论 A/B，都要给 `/commerce/*`、`/trading/*`、`/billing/*` 补一组契约测试，断言响应信封。
- **截图**：`02-wallet-commerce-contract-break.png`、`12-today-signal-request-failed.png`、`11-subscription-dashboard-500.png`

---

### P0-2　登录/注册读 `data.access_token`，但后端 `AuthResponse` 是裸体 → 登录态字段错位
- **严重度**：P0
- **文件**：`frontend/src/app/login/page.tsx:44-50,86-90`、`frontend/src/app/register/page.tsx:47-53`
- **现象**：后端 `auth.py:57-61` `class AuthResponse(BaseModel): access_token; refresh_token; token_type; user: dict`，
  返回**裸体**。前端 `login/page.tsx:46` 读 `data.access_token`（顶层）——**当前恰好正确**；
  但 `register/page.tsx:54` 写 `setUser(data.user as unknown as Parameters<typeof setUser>[0])`，
  而 `useAppStore.User` 的字段是 `{id, name, email, role, is_admin?, subscription_tier?}`，
  后端 `user` 是 `{id, email, username, subscription_tier, is_admin}` — **没有 `name`、没有 `role`**。
  于是走 `/register` 注册的用户，`useAppStore.user.name` 与 `.role` 恒为 `undefined`，
  顶栏头像/角色展示退化。
  更要命的是：登录页 `handleLogin`（:53-60）**手工**做了 `username → name`、`subscription_tier → role` 的映射，
  而注册页（:54）**没有做**——两处对同一份后端 payload 的解释不一致，靠 `as unknown as` 硬压过去。
- **复现步骤**：
  1. 打开 `/register` → 提交合法表单 → 成功跳转 `/dashboard`。
  2. 读 `useAppStore.getState().user` → `{id, email, username, subscription_tier, is_admin}`，
     **无 `name`、无 `role`**（`as unknown as` 把编译期检查也一并关掉了）。
- **改法**：
  1. 在 `useAppStore` 之外新增一个唯一的 `mapAuthUser(raw: AuthUserPayload): User` 归一函数，
     登录页与注册页**都**调它，删掉 `as unknown as`。
  2. `User` 接口改为直接对齐后端字段（`username` / `subscription_tier`），或保留 `name` 但在归一层做唯一映射。
  3. 归一函数对 `subscription_tier` 缺失的情况给显式兜底（`?? 'free'`），避免 `role` 变 `undefined`。
- **截图**：`01-login-username-blocked.png`

---

### P0-3　退出登录删错 key → token 残留，退出后仍可访问受保护页
- **严重度**：P0
- **文件**：`frontend/src/components/layout/AppLayout.tsx:202`
- **现象**：
  ```ts
  localStorage.removeItem('auth_token');   // ← 这个 key 全项目从未被写入过
  router.push('/login');
  ```
  实际存 token 的 key 是 `access_token` / `refresh_token` / `user_info`
  （`login/page.tsx:46-50`、`services/api.ts:101-104`）。`auth_token` 是幽灵 key，删了等于没删。
- **复现步骤**：
  1. 登录 → 右上角头像 → 「退出登录」→ 跳到 `/login`。
  2. 读 `localStorage` → `['refresh_token','user_info','access_token']`，**`access_token` 仍在**。
  3. 直接访问 `/wallet` → 页面**完整加载**（后端真实鉴权通过，余额/订单接口照常返回），只弹了一个「加载数据失败」toast。
  4. 浏览器后退 → 回到 `/dashboard`，仍以登录态渲染。
  - 公共机/共享电脑场景下等于**没有退出登录**。
- **改法**：
  1. `AppLayout.tsx:202` 改为调用 `services/api.ts` 已有的 `clearStoredAuth()`（它已经正确删了三个 key），
     导出该函数供 UI 复用，不要各页面各写一份。
  2. 顺带清 `useAppStore` 里的 `user` / `isAuthenticated`（当前 `setUser(null)` 从没被调用过）。
  3. 补一条 e2e：登录 → 退出 → 断言 `localStorage.getItem('access_token') === null` 且 `/wallet` 重定向到 `/login`。
- **截图**：`03-logout-menu.png`（退出入口）、`04-logout-token-persists.png`

---

### P0-4　管理端「充值订单确认到账」4 个接口全部 404 → 收入确认链路断
- **严重度**：P0
- **文件**：`frontend/src/app/admin/payments/page.tsx:84,116,129,143`
- **现象**：前端打的是 `/api/v1/admin/payment/orders`（少了一层 `/commerce`），后端实际路由是
  `/api/v1/commerce/admin/payment/orders`、`/api/v1/commerce/admin/payment/orders/{order_no}/confirm`、
  `…/cancel`、`/api/v1/commerce/admin/payment/config`。
  **4 个调用全部 404**，而这 4 个正是「用户转账 → 管理员确认到账 → 权益生效」这条变现链路的全部环节。
- **复现步骤**：
  ```bash
  curl -X GET  'http://127.0.0.1:18082/api/v1/admin/payment/orders'                 -H 'Authorization: Bearer <tok>'
  # → 404 {"success":false,"message":"Not Found"}
  curl -X POST 'http://127.0.0.1:18082/api/v1/admin/payment/orders/PO-X/confirm'    # → 404
  curl -X POST 'http://127.0.0.1:18082/api/v1/admin/payment/orders/PO-X/cancel'     # → 404
  curl -X POST 'http://127.0.0.1:18082/api/v1/admin/payment/config'                 # → 404
  ```
  正确路径 `GET /api/v1/commerce/admin/payment/orders` → **200**，返回
  `{"orders":[…,"order_no":"PO-JJJJC2M7","status":"paid",…],"count":N}`。
- **改法**：4 处 URL 补 `/commerce` 前缀：
  - `page.tsx:84` → `/commerce/admin/payment/orders`
  - `page.tsx:116` → `/commerce/admin/payment/orders/${orderNo}/confirm`
  - `page.tsx:129` → `/commerce/admin/payment/orders/${orderNo}/cancel`
  - `page.tsx:143` → `/commerce/admin/payment/config`
  改完注意 P0-1：`/commerce/*` 是裸体，`apiRequest.get` 同样会判失败，需一并按 P0-1 的方案处理。
- **截图**：`09-admin-payments-404-route.png`

---

### P0-5　参数调优页 URL 双前缀 `/api/v1/api/v1/*` → 3 个调用全 404，功能整体不可用
- **严重度**：P0
- **文件**：`frontend/src/app/tasks/tuning/page.tsx:170,267,305`
- **现象**：`apiRequest` 的 `baseURL` 已经是 `'/api/v1'`（`services/api.ts:135`），但这 3 处传入的 URL
  又带了 `/api/v1` 前缀，实际请求变成 `/api/v1/api/v1/tasks?...` → 404。
  `page.tsx:170` 的 `.catch(() => { /* fallback: 可能是列表响应变成了分页结构 */ })` 把 404 静默吞成空列表，
  所以 UI 上**看不到任何报错**，只是「选择已完成回测任务」下拉框永远是空的。
- **复现步骤**：
  1. 打开 `/tasks/tuning` → 「① 选择基准任务」下拉框为空，任务数 0。
  2. `curl 'http://127.0.0.1:18082/api/v1/api/v1/tasks?limit=200&status=completed'` → **404**。
  3. 正确路径 `/api/v1/tasks?limit=200&status=completed` → **200**。
- **改法**：3 处去掉重复前缀 → `'/tasks?limit=200&status=completed'`、
  `'/tasks/tuning/preview-config'`、`'/tasks/tuning/submit'`。
  并把 `page.tsx:171-183` 的空 `catch` 改成至少 `setError('加载任务列表失败')`。
  建议在 `apiRequest` 里加一条开发期断言：`url.startsWith('/api/')` 时直接 `throw`，
  从根上杜绝这类双前缀。
- **截图**：`13-tasks-tuning-double-prefix-404.png`

---

### P0-6　创建任务失败被完全吞掉（`logger.error` only）→ 用户不知情，且可重复提交重复计费
- **严重度**：P0
- **文件**：`frontend/src/app/tasks/create/page.tsx:415-419`
- **现象**：
  ```ts
  } catch (error) {
    logger.error('创建任务失败:', error);   // ← 只有这一行
  } finally { setLoading(false); setCreating(false); }
  ```
  后端返回 400/402/500/配额不足时，**页面上没有任何提示**：按钮恢复可点、表单保持原样、
  用户以为「点了没反应」就再点一次。
  商业影响：回测按 **¥0.5/次** 计费（`backtest_basic`），重复提交 = 重复建任务 + 重复扣费；
  配额不足（free tier `monthly_backtest_limit`）这类**最需要告知用户**的场景恰恰一个字都不显示。
- **复现步骤**：
  1. 打开 `/tasks/create?type=backtest`。
  2. 不填任何东西直接点「创建任务」→ 就地校验会拦（`validateForm`，:322-357）——**这一层是好的**。
  3. 填全字段后提交，在请求发出前把网络断开 / 让后端返回 500
     → 页面无任何错误提示，按钮恢复可用。
  4. 代码路径见 `page.tsx:415-419`。
- **改法**：
  1. `catch` 里加用户可见提示：优先 `useSnackbarStore().showError(可读中文消息)`，并把后端 `detail`/`message` 透出。
  2. 提交按钮在 `loading` 时 `disabled`（**当前 `:935` 已经是 `disabled={loading}`，这点是对的**），
     但 `setLoading(false)` 发生在 catch 之后 → 用户可立即重提。需在 catch 分支**保持表单内容**
     并提示「可重试」，避免重复建单。
  3. 若后端有幂等键，提交时带 `Idempotency-Key`（可防连点产生的重复任务）。
- **截图**：`08-create-task-form.png`

---

### P0-7　`/dashboard/stats` 后端 500 → 订阅页 tier 回落到硬编码 `'free'`，enterprise 用户被显示成 FREE
- **严重度**：P0
- **文件**：`frontend/src/app/settings/subscription/page.tsx:61-82`
- **现象**：
  - 后端 `GET /api/v1/dashboard/stats` → **500**，`message` 为
    `用量统计获取失败: 获取用户任务总数失败: 'AsyncSession' object has no attribute 'query'`
    （后端 bug，但前端把它**静默降级**成了「免费方案」）。
  - 前端 `page.tsx:71-80` 的 `catch` 分支读 `localStorage.user_info` 兜底；
    但兜底路径只取 `user.subscription_tier`，而 `localStorage.user_info` 在**登录/注册时**写入的
    是后端 `user` payload（有 `subscription_tier`）——一旦 `user_info` 缺失/被写成字符串 `"undefined"`
    （见 P2-3），`JSON.parse` 抛错 → `catch` → `setTier('free')`。
  - 结果：**enterprise 用户在订阅页看到「FREE 免费方案」**，且页面无任何错误提示，
    商业上等于「付费用户看不到自己的权益」。
  - 另外，500 响应把后端内部异常原文（`'AsyncSession' object has no attribute 'query'`）
    直接透给了客户端——既是信息泄露，也是无用的用户文案。
- **复现步骤**：
  1. 登录 `ronghui@example.com`（`subscription_tier=enterprise`）。
  2. `curl -H 'Authorization: Bearer <tok>' http://127.0.0.1:18082/api/v1/dashboard/stats` → **500**。
  3. 打开 `/settings/subscription` → 「当前套餐 FREE / 免费方案 / 本月用量 0/50 / 0/20 / 0/3」。
  4. 正确值应为企业版（`/api/v1/auth/me` 返回 `subscription_tier: "enterprise"`）。
- **改法**：
  1. **后端**：修 `dashboard.py` 的 `AsyncSession.query` 调用（改用 `await session.execute(...)`），
     并把 500 的 `message` 换成用户可读文案，内部异常只进日志。
  2. **前端**：`catch` 分支里 tier 兜底**不要静默降级成 `'free'`**——
     拿不到真实 tier 就显示「套餐信息加载失败，点击重试」，而不是谎报 FREE。
  3. 订阅页加一个「刷新」按钮（当前只有 `loadSubscription` 在 `useEffect` 里跑一次，失败后无法重试）。
- **截图**：`11-subscription-dashboard-500.png`

---

### P0-8　`useTaskStore` 全局持有服务端任务列表 + 分页元数据 → 跨页污染与陈旧数据
- **严重度**：P0
- **文件**：`frontend/src/stores/useTaskStore.ts:59-68,80-93`、`frontend/src/app/backtest/page.tsx:56,96-98`、`frontend/src/app/tasks/page.tsx:77,98-99`
- **现象**：`useTaskStore` 是**模块级单例**，持有 `tasks` / `total` / `currentPage` / `pageSize` / `statusFilter`。
  - `/tasks` 页 `loadTasks` 调 `setTasks(result.tasks, result.total)`，写入**当前页**的任务（默认 10 条）。
  - `/backtest` 页 `loadTasks` 调 `TaskService.getTasks(undefined, 1000, 0)`，把**全量 1000 条**塞进**同一个** `tasks` 数组，
    再用 `tasks.filter(t => t.task_type === 'backtest')` 在**前端**筛出回测任务（`backtest/page.tsx:64-89`）。
  - 后果 A：**前端过滤 = 数据不全**。后端 `GET /tasks` 默认分页，返回的 1000 条可能**不包含**全部 backtest 任务；
    真正带 `task_type` 过滤的是后端 `?status=` / `?task_type=` 能力，前端没用到。
  - 后果 B：**跨页污染**。`/tasks` → `/backtest` → `/tasks` 来回切，`tasks` 数组残留上一次的量级与内容，
    而 `total` 又来自不同请求 → 表格里的「共 N 条」和实际渲染条数对不上。
  - 后果 C：**分页状态跨页残留**。`/tasks` 设的 `statusFilter` / `currentPage` 会带进 `/backtest`。
  - 后果 D：`updateTask`（:80-93）用 `task.task_id === taskId` 匹配，若后端某次返回的 `task_id` 缺失，
    整批任务都更新不到。
- **复现步骤**：
  1. 登录 → `/tasks`（看到 1 条任务，`total=1`）→ 点进 `/backtest`（触发 `getTasks(limit=1000)` 覆盖 store）→ 返回 `/tasks`。
  2. 观察 `/backtest` 上「回测任务」列表：它来自前端 `filter`，而非后端按 `task_type` 过滤的结果。
  3. 代码：`backtest/page.tsx:64-89` + `taskService.ts:28-39`。
- **改法**：
  1. **服务端数据不要进全局 store**。用 react-query v3（本项目已装 `react-query@3.39.3`，但**全项目 0 处使用**）
     或至少 `useState` + `useEffect` 局部持有；`useTaskStore` 只保留真正跨页的 UI 态（筛选条件、当前选中项）。
  2. `/backtest` 改用后端参数过滤：`GET /tasks?task_type=backtest&limit=…`，去掉前端 `filter`。
  3. 若必须保留 store，给 `setTasks` 加 `source` 维度或改用 `useTaskStore` 的**切片 selector**
     （`useTaskStore(s => s.tasks)` 已用，但数组本身仍共享）——按页面拆 store 或给每次加载打 `requestId`。
  4. 分页元数据（`total`/`currentPage`/`pageSize`）必须与数据**同源同批次**写入。
- **截图**：`06-tasks-list-three-states.png`、`10-tasks-batch-delete.png`

---

## P1 问题清单

### P1-1　`tasks/page.tsx` 加载失败被吞成「暂无任务」——错误态与空态不可区分
- **严重度**：P1
- **文件**：`frontend/src/app/tasks/page.tsx:100-104`
- **现象**：`catch` 只 `logger.error('加载任务失败:', error)`，**没有 `setError`、没有错误 UI**。
  页面继续走 `filteredTasks.length === 0` 分支渲染「暂无任务」（`:446-451` / 桌面端同款）。
  用户看到的是「这平台没有任务」，而不是「后端挂了/网络断了」——**运维与用户都被误导**。
- **复现步骤**：
  1. 登录 → `/tasks`。
  2. 页面内注入 XHR 补丁，让 `/api/v1/tasks*` 返回 500（见 P1-2 复现脚本）→ 点「刷新」。
  3. 页面显示「暂无任务」，**无任何错误提示**。
- **改法**：
  1. 加 `const [loadError, setLoadError] = useState<string|null>(null)`，`catch` 里 `setLoadError(msg)`。
  2. 渲染时区分：`loadError ? <错误 Alert + 重试按钮> : tasks.length === 0 ? <空态> : <列表>`。
  3. 抽一个共用的 `<DataState loading error empty onRetry/>` 组件，所有列表页统一用（见 P1-3）。
- **截图**：`07-tasks-api-500-error-state.png`

---

### P1-2　`apiRequest` 的失败信息丢失后端原文 → 用户只看到「请求失败」
- **严重度**：P1
- **文件**：`frontend/src/services/api.ts:194-198,300-328`
- **现象**：
  ```ts
  if (!data.success) {
    apiLogger.error('[API] 业务错误:', data.message);
    apiLogger.error(data.message || '请求失败');
    return Promise.reject(new Error(data.message || '请求失败'));  // ← 消息丢了
  }
  ```
  - 后端 `StandardResponse.message` 是唯一的中文可读原因，被 `new Error(...)` 塞进 `.message` 后，
    页面若用 `e.message` 展示尚可，但很多页面**根本没读** `e.message`（如 `tasks/page.tsx:101` 只写日志）。
  - 更关键：`apiRequest.get` 的 `.catch`（:316-327）**只对 GET 做了错误增强**（补 `status`），
    `post` / `put` / `delete` / `patch`（:330-364）**完全没有 `.catch`** →
    这 4 个方法的错误是**裸 AxiosError**，`error.message` = `"Request failed with status code 400"`，
    `error.status` **不存在**。
  - 后果：`taskService.ts:161-171` 的 `getTaskConfig` 靠 `error?.status === 404` 判断「旧任务没有快照」——
    用的是 `get`（有增强）所以侥幸能工作；但任何用 `post` 的地方要判断状态码就会拿到 `undefined`。
- **复现步骤**：
  1. `apiRequest.post('/tasks', {...})` 传一个后端会 422 的 body → catch 到的 error 上 `error.status` 是 `undefined`，
     `error.message` 是英文的 `"Request failed with status code 422"`。
  2. 后端实际给的是 `{"success":false,"message":"…","detail":[…]}`。
- **改法**：
  1. 抽一个 `normalizeError(error): EnhancedError` 统一函数，**所有 5 个方法都套**，
     把 `response.data.message`（或 `detail`）提为 `error.message`，`error.status = response.status`。
  2. `new Error(message)` 时补 `name`/`cause`/`code` 字段，保留原始 error 便于上报。
  3. 422（FastAPI 校验错误）的 `detail` 是**数组**不是字符串，要单独展开成
     「日期格式不正确」这类可读文案。
- **截图**：`07-tasks-api-500-error-state.png`

---

### P1-3　三态（加载/空/失败）没有统一组件，各页各写各的
- **严重度**：P1
- **文件**：`frontend/src/app/`（`tasks` / `models` / `data` / `backtest` / `signals` / `monitoring` / `predictions` / `optimization` / `settings`）
- **现象**：项目里有 `LoadingSpinner` / `EmptyState` / `PageSkeleton` / `ErrorBoundary` 四个"半成品"通用组件，
  但**没有**一个统一的「数据三态」容器。各页自己手搓，行为不一致：

  | 页面 | 加载中 | 空态 | 失败态 |
  |---|---|---|---|
  | `tasks/page.tsx` | ✅ `:290-292` | ✅ `:446-451` | ❌ 无（吞成空态） |
  | `models/page.tsx` | ✅ `:410-411` | ✅ `:455` | ❌ 无 |
  | `data/page.tsx` | ⚠️ 分区块各自 `LoadingSpinner` | ✅ `:990,1084` | ⚠️ 只有 `console.error` |
  | `backtest/page.tsx` | ✅ `:224-225` | ✅ `:324` | ✅ `:228`（唯一做对的） |
  | `signals/page.tsx` | ✅ | ✅ `:711,775,783` | ✅ `:657` |
  | `monitoring/page.tsx` | ✅ `:134-135` | ✅ `:356,381,510,598` | ❌ 无（`:92-95` 只 console） |
  | `predictions/page.tsx` | ✅ `:180` | ✅ `:423` | ✅ `:254` |
  | `optimization/page.tsx` | ✅ `:92` | ⚠️ | ❌ 无（`:45-48` 只 console） |
  | `settings/subscription/page.tsx` | ✅ | ⚠️ | ❌ 静默降级（P0-7） |

  `EmptyState`（`components/common/EmptyState.tsx`）几乎没被引用（`icon` 默认是 emoji `📭`，`actionLabel` 与 `onAction` 必须同时传才渲染按钮）。
- **复现步骤**：逐页访问 `/tasks`、`/models`、`/monitoring`、`/optimization`，在控制台把对应接口打成 500，
  观察是否有可重试的错误 UI。
- **改法**：
  1. 新建 `components/common/DataState.tsx`：
     ```tsx
     <DataState loading={loading} error={error} empty={!loading && !error && data.length===0}
                onRetry={refetch} emptyTitle="..." emptyActionLabel="..." />
     ```
  2. 所有列表页换成它，删掉各自的 `if (loading)` / `length===0` 分支。
  3. 统一错误文案：技术细节进 `console`，用户文案只留一句「加载失败，点击重试」+ 可选的具体原因。
- **截图**：`07-tasks-api-500-error-state.png`、`06-tasks-list-three-states.png`

---

### P1-4　`alert()` 用于业务反馈（15 处）——阻塞 UI、样式不统一、无法批量关闭
- **严重度**：P1
- **文件**：`frontend/src/app/models/page.tsx:277,280,304,307,386,389`、
  `frontend/src/components/optimization/CreateOptimizationTaskForm.tsx:230,297`、
  `frontend/src/components/models/FeatureSelector.tsx:309,314`、
  `frontend/src/components/models/TrainingReportModal.tsx:223,225,309`
- **现象**：用浏览器原生 `alert()` 做成功/失败/校验反馈：
  - 阻塞主线程，样式由浏览器决定，与 MUI/antd 主题完全脱节；
  - 移动端体验差，且**无法截图断言**、无法埋点、无法做「多错误同时展示」；
  - 项目同时装了 `useSnackbarStore`（`stores/useSnackbarStore.ts`）+ `GlobalSnackbar`（MUI Snackbar），
    机制齐全却**没在这些地方用**。
- **复现步骤**：
  1. `/models` → 删除一个模型 → 弹出浏览器原生「模型删除成功」白框（非 MUI Alert）。
  2. `/optimization` → 不填必填项提交 → 原生「请填写所有必填字段」。
- **改法**：
  1. 15 处 `alert()` 全部替换为 `useSnackbarStore().showSuccess/showError/showWarning`。
  2. 表单校验类（`FeatureSelector.tsx:309,314`、`CreateOptimizationTaskForm.tsx:230`）改为**就地错误提示**
     （字段下方 `FormHelperText error`），不要用 toast——toast 不指向具体字段。
  3. 加 ESLint 规则禁止 `alert`/`confirm`（`no-restricted-globals`），防回归。
- **截图**：`08-create-task-form.png`

---

### P1-5　`getStockData` 把异常吞成空数组 → 「查无数据」与「查询失败」不可区分
- **严重度**：P1
- **文件**：`frontend/src/services/dataService.ts:205-216`
- **现象**：
  ```ts
  } catch (error: unknown) {
    dataLogger.error('[DataService] getStockData 失败:', error);
    return { stock_code: stockCode, data: [], last_updated: new Date().toISOString() };
  }
  ```
  注释还写着「返回空数据而不是抛出错误，让组件可以处理」——
  但组件**无法区分**「该股票无数据」和「后端 500 / 网络断了」。
  更糟的是返回的 `last_updated` 是**当前时间**，看起来像刚拉过数据。
- **复现步骤**：
  1. 打开 `/predictions` 或任何走 `getStockData` 的图表页。
  2. 断开网络或让 `/api/v1/stocks/data` 返回 500 → 图表区域显示「无数据」，无错误提示。
- **改法**：
  1. 不要在这里吞异常。要么抛出，要么返回一个**带 `error` 字段的显式结果类型**：
     ```ts
     type StockDataResult = { ok: true; data: StockData } | { ok: false; error: string };
     ```
  2. 页面按 `ok` 分支渲染「无数据」或「加载失败 + 重试」。
  3. `last_updated` 不要在失败路径上伪造。
- **截图**：`08-create-task-form.png`

---

### P1-6　任务导出：方法错（GET vs POST）+ 格式不支持（csv vs pdf/excel）+ 错误被 `// noop` 吞
- **严重度**：P1
- **文件**：`frontend/src/services/taskService.ts:116-124`、`frontend/src/app/tasks/[id]/useTaskDetailPage.ts:264-278`
- **现象**：三重不匹配：
  1. **方法错**：前端 `apiRequest.get('/tasks/{id}/export')`；后端只有 `POST /tasks/{task_id}/export`
     （`tasks_impl.py:1016`）→ `GET` 返回 **405 Method Not Allowed**。
  2. **格式错**：前端传 `format: 'csv'`；后端 `BacktestExportRequest.format` 只接受 `pdf` / `excel`，
     传 `csv` → **400「不支持的导出格式」**。
  3. **响应类型错**：后端返回 `StandardResponse`（信封 + `data` 是文件路径），
     前端却 `new Blob([response], {type:'text/csv'})` —— 即使前面都对，**下载到的也会是 JSON 字符串而不是 CSV**。
  4. **错误被吞**：`useTaskDetailPage.ts:277-279` 是 `catch (_error) { // noop }` —— 用户点「导出」后
     **什么都不会发生，也没有任何提示**。
- **复现步骤**：
  ```bash
  curl -X GET  'http://127.0.0.1:18082/api/v1/tasks/xxx/export' -H 'Authorization: Bearer <tok>'  # → 405
  curl -X POST 'http://127.0.0.1:18082/api/v1/tasks/xxx/export' -H 'Content-Type: application/json' \
       -d '{"format":"csv"}' -H 'Authorization: Bearer <tok>'                                     # → 400 不支持的导出格式
  ```
  4. 打开任一已完成回测任务的详情页 → 点「导出」→ 无反应、无提示。
- **改法**：
  1. `exportTaskResults` 改 `apiRequest.post`，body 传 `{format:'excel'}`（或按后端支持的枚举取值）。
  2. 后端返回的是文件路径，不是文件流 → 前端应 `window.open(res.download_url)` 或对路径再发一次下载请求，
     不能 `new Blob([json])`。
  3. `catch (_error) { // noop }` 必须补上用户提示（snackbar）。
- **截图**：`10-tasks-batch-delete.png`

---

### P1-7　批量删除任务调后端不存在的端点（405）+ 错误信息被丢弃
- **严重度**：P1
- **文件**：`frontend/src/services/taskService.ts:129-131`、`frontend/src/app/tasks/page.tsx:258-270`
- **现象**：`POST /api/v1/tasks/batch-delete` 在后端**不存在**（路由表里没有）→ 命中
  `POST /tasks/{task_id}/...` 的模式不匹配，返回 **405 Method Not Allowed**。
  页面侧 `tasks/page.tsx:267-269` 是 `catch (error) { logger.error('批量删除失败'); }` ——
  错误信息被丢弃，用户只看到「勾选消失了但任务还在」。
- **复现步骤**：
  1. `curl -X POST 'http://127.0.0.1:18082/api/v1/tasks/batch-delete' -H 'Content-Type: application/json' -d '{"task_ids":["x"]}'` → **405**。
  2. `/tasks` 页勾选任务 → 批量删除 → 列表未变，无提示。
- **改法**：
  1. 要么后端补 `POST /tasks/batch-delete`，要么前端改成循环调已存在的 `DELETE /tasks/{task_id}`
     （`taskService.deleteTask` 已实现，`force` 参数后端支持）。
  2. `catch` 里保留并展示后端 message（配合 P1-2 的 `normalizeError`）。
  3. 批量操作要有部分成功的反馈（成功 N 条 / 失败 M 条）。
- **截图**：`10-tasks-batch-delete.png`

---

### P1-8　`/monitoring/health` 的 `services` 是对象字典，前端当字符串数组比较 → 健康状态恒为「正常」
- **严重度**：P1
- **文件**：`frontend/src/app/dashboard/page.tsx:73-93`
- **现象**：前端类型声明 `services?: Record<string, string>`，判定
  `Object.values(services).some(s => s === 'unhealthy' || s === 'error')`。
  但后端实际返回：
  ```json
  "services": {
    "remote_data_service": {"healthy": true, "status_label": "正常", "response_time_ms": 3.03, ...},
    "indicators_service": {...}, "parquet_manager": {...}
  }
  ```
  值是**对象**不是字符串 → `s === 'unhealthy'` **恒为 false** → `hasError` 恒 false →
  `systemHealth` **永远是 `'good'`**。后端任何一个服务挂了，仪表板照样显示「系统状态 正常」。
- **复现步骤**：
  1. `curl -H 'Authorization: Bearer <tok>' http://127.0.0.1:18082/api/v1/monitoring/health` →
     `data.services.remote_data_service.healthy = true`（对象）。
  2. 把某个服务打挂（`healthy:false`）→ 仪表板仍显示「正常」。
  3. 代码：`dashboard/page.tsx:83-85`。
- **改法**：
  1. 改前端类型为 `Record<string, {healthy:boolean; status_label?:string; response_time_ms?:number; error_message?:string|null}>`。
  2. 判定改为 `Object.values(services).some(s => !s.healthy)`，
     并把 `status_label` 展示出来（后端已经给了中文标签，前端没用）。
  3. 顺带修 `dashboard/page.tsx:188` 的 `dataFiles: 156` —— **硬编码假数据**，
     正常路径 `:92` 写的是 `dataFiles: 0`。重试按钮点下去会显示 156 个数据文件，纯属编造。
- **截图**：`11-subscription-dashboard-500.png`

---

## P2 问题清单

### P2-1　`apiRequest` 的 401 刷新流程可无限循环刷新 / 无并发上限
- **严重度**：P2
- **文件**：`frontend/src/services/api.ts:227-256`
- **现象**：
  - `_retry` 标记挂在 `error.config` 上，刷新成功后用 `instance(originalRequest)` 重放；
    但 `originalRequest._retry = true` 是在**第一次 401** 时设的，重放请求若**再 401**，
    拦截器里 `!originalRequest._retry` 为 false → 走 `case 401` 分支 → 清 token 跳登录。**这一段是对的**。
  - 真正的问题：`refreshAccessToken` 用**裸 `axios.post('/api/v1/auth/refresh')`**（:92），
    绕过了 `api` 实例的 `baseURL`，靠浏览器同源解析。开发环境下前端在 3000，
    `/api/v1/auth/refresh` 会被 **Next.js rewrite** 代理到后端 —— 恰好能通，但这是巧合，
    一旦部署路径变化（比如前端挂在 `/dashboard` 子路径）就会 404。
  - `isRefreshing` / `refreshQueue` 是**模块级变量**，在 SSR（`typeof window === 'undefined'`）下
    也会被创建并共享 —— 跨请求串味。
  - 没有刷新**重试上限**：`flushRefreshQueue` 失败后所有排队请求一起 reject，
    但 `isRefreshing = true` 期间新到的请求会 push 进队列，如果刷新请求本身超时（axios 默认无超时），
    队列会**永久挂起**。
- **改法**：
  1. 刷新请求也走 `api` 实例（用 `api.post` 并带 `skipAuthRefresh` 标志位），
     或显式用 `axios.post(`${api.defaults.baseURL}/auth/refresh`)`。
  2. 给刷新加 `timeout: 15000` 和**最多重试 1 次**。
  3. 队列 reject 时清空 `refreshQueue`（当前 `flushRefreshQueue` 会清，但 `catch` 分支的
     `flushRefreshQueue(refreshError)` 之后 `finally` 里的 `isRefreshing = false` 之后仍可能有残留）。
  4. 401 时给用户一个「登录已过期，正在重新登录」的可见提示，别静默跳转。

---

### P2-2　`/data/sync*` 系列 5 个接口后端不存在（404），同步历史/进度弹窗必挂
- **严重度**：P2
- **文件**：`frontend/src/services/dataService.ts:663,751,769,800,818`
- **现象**：以下前端调用在后端**全部无对应路由**：

  | 前端调用 | 后端实际 | 结果 |
  |---|---|---|
  | `POST /data/sync` | 无（只有 `POST /data/sync/remote`） | **404** |
  | `DELETE /data/files?...` | 只有 `GET /data/files` | **405** |
  | `GET /data/sync/{id}/progress` | 无 | **404** |
  | `GET /data/sync/history` | 无 | **404** |
  | `POST /data/sync/{id}/retry` | 无 | **404** |

  实际用到的组件：`components/monitoring/SyncProgressModal.tsx:69,101`、
  `components/monitoring/SyncHistoryModal.tsx:78,95`。
- **复现步骤**：后端路由表逐条对拍（`app.main.app.routes`）；`curl` 验证均 404/405。
- **改法**：
  1. **后端补齐** `POST /data/sync`、`GET /data/sync/{id}/progress`、`GET /data/sync/history`、
     `POST /data/sync/{id}/retry`、`DELETE /data/files` 五个端点（`data_impl.py` 里已有
     `sync_data_from_remote` 之类的服务方法，只是没暴露路由）。
  2. 前端同步删掉无用的 `syncDataFromRemote` / `deleteDataFiles` / `getSyncProgress` /
     `getSyncHistory` / `retrySyncFailed`，或标注 `@deprecated` 等待后端补齐。
  3. 补完后加契约测试，避免再漂移。

---

### P2-3　`localStorage.user_info` 是**无版本**的持久化快照，可被写成字符串 `"undefined"`
- **严重度**：P2
- **文件**：`frontend/src/app/login/page.tsx:50,90`、`frontend/src/app/register/page.tsx:53`、
  `frontend/src/app/settings/subscription/page.tsx:69`、`frontend/src/app/pricing/page.tsx:142-151`
- **现象**：
  - 写入方：`login/page.tsx:50` `localStorage.setItem('user_info', JSON.stringify(data.user))`。
    若 `data.user` 是 `undefined`，`JSON.stringify(undefined)` 返回 **`undefined`**（不是字符串），
    `setItem` 会把它**强制转成字符串 `"undefined"`** 存进去。
  - 读取方：`pricing/page.tsx:146` `JSON.parse(storedUser)` → `JSON.parse("undefined")` **抛异常** →
    `catch` → `setCurrentTier('free')`；`subscription/page.tsx:73` 同理 → tier 静默变 FREE（见 P0-7）。
  - 实测中确实观察到 `localStorage.getItem('user_info') === "undefined"`。
  - **没有版本迁移机制**：`featureConfigs.ts:279-360` 的 `feature_configs_custom` 也是裸 JSON，
    字段一改就整份丢弃（`getCustomConfigs` catch 返回 `[]`），用户自定义特征配置静默消失。
- **复现步骤**：
  1. 手动 `localStorage.setItem('user_info','undefined')` → 刷新 `/pricing` 或 `/settings/subscription`
     → 套餐显示 FREE。
  2. 读 `localStorage.getItem('user_info')` 确认存的是字符串 `"undefined"` 而非 JSON。
- **改法**：
  1. **写入前校验**：`if (data?.user) localStorage.setItem('user_info', JSON.stringify(data.user))`，
     杜绝写 `"undefined"`。
  2. **读取时校验**：`JSON.parse` 后 `typeof user === 'object' && user?.id` 才认，否则当未登录处理并清除。
  3. **加版本字段**：`localStorage.setItem('user_info', JSON.stringify({v:1, user}))`，
     读取时 `v !== CURRENT_VERSION` 就丢弃并重新拉 `/auth/me`。
  4. `feature_configs_custom` 同样加 `{v:1, items:[…]}` 结构 + 版本判定。
  5. 根本解法：**别把服务端用户态持久化在 localStorage** —— 每次启动调 `GET /api/v1/auth/me`
     （该端点存在且返回完整 `UserInfo`），localStorage 只放 token。

---

### P2-4　`/api/v1/version`、`/api/v1/data/remote/summary` 等端点契约不匹配
- **严重度**：P2
- **文件**：`frontend/src/services/dataService.ts:539,571`
- **现象**：
  - `getApiVersion()` 打 `GET /api/v1/version` → **404**；后端实际是 `GET /api/v1/system/version` → 200。
  - `getRemoteDataSummary()` 打 `GET /api/v1/data/remote/summary` → 200 但
    `{"success":false,"message":"数据服务未连接，无法获取汇总统计","data":null}`；
    前端类型 `DataServiceSummary` 声明 `total_stocks: number`（必填），
    实际 `data` 为 `null` → `apiRequest.get` 走到 `:310-313` 的 `return null as T`，
    **把 `null` 冒充成 `DataServiceSummary` 返回**，调用方 `setDataSummary(summary)` 拿到 null。
    页面 `data/page.tsx:145-151` 有 `setDataSummary(null)` 的分支，但那是 catch 分支；
    走的是「成功但 data=null」路径，**行为不确定**。
- **改法**：
  1. `dataService.ts:539` 改为 `'/system/version'`。
  2. `apiRequest.get` 的 `return null as T`（:310-313）不要用类型断言糊弄 ——
     改成抛错，或让返回类型变成 `T | null` 强制调用方处理。
  3. `getRemoteDataSummary` 的返回类型改 `DataServiceSummary | null`。

---

### P2-5　`as unknown as` / `any` 绕过类型系统，掩盖真实契约错误
- **严重度**：P2
- **文件**：`frontend/src/app/register/page.tsx:54`、`frontend/src/app/predictions/page.tsx:131`、
  `frontend/src/types/task.ts:13,14,18,25,26,29,35,48,50,88`、
  `frontend/src/services/api.ts:23-24`（`LooseApiPayload = any`）、
  `frontend/src/services/dataService.ts:47-48`（`LooseJsonValue = any`）
- **现象**：
  - `register/page.tsx:54` `setUser(data.user as unknown as Parameters<typeof setUser>[0])` ——
    后端 user 是 `{id,email,username,subscription_tier,is_admin}`，
    `User` 是 `{id,name,email,role,is_admin?,subscription_tier?}`，
    `as unknown as` 直接把 `name`/`role` 缺失这件事**从编译器视野里抹掉**（详见 P0-2）。
  - `predictions/page.tsx:131` `setPredictions(response as unknown as PredictionResult[])` ——
    同样是绕过。
  - `types/task.ts` 里 `result?: any` / `backtest_results?: any` / `[key: string]: any` ——
    **索引签名是 `any`**，等于该类型下的任何字段访问都不报错，回测结果字段拼错也不会被发现。
  - `services/api.ts:24` `type LooseApiPayload = any` 是 `apiRequest` 全部方法的**默认泛型** ——
    写 `apiRequest.get('/xxx')` 不传泛型时返回 `any`，等于没有类型。
  - `npx tsc --noEmit` **通过**（exit 0）—— 说明这些 `any` 正是类型系统失效的「合法」逃逸口。
- **改法**：
  1. `apiRequest` 的默认泛型从 `any` 改成 `unknown`，强制每个调用点显式声明返回类型。
  2. `types/task.ts` 的 `[key: string]: any` 改成 `unknown`，
     `result`/`backtest_results` 指向具体的 `BacktestResults` 接口。
  3. 删掉 2 处 `as unknown as`，改成显式映射函数（见 P0-2 的改法）。
  4. `tsconfig.json` 打开 `noImplicitAny` 已是 strict；再考虑开
     `"noUncheckedIndexedAccess": true` 抓索引签名问题。
  5. ESLint 加 `@typescript-eslint/no-explicit-any: error`（当前只有 2 处 disable 注释，属于打地鼠）。

---

## P3 问题清单

### P3-1　`react-query@3.39.3` 装了但全项目 0 处使用
- **严重度**：P3
- **文件**：`frontend/package.json:57`（依赖）、全项目无 `react-query` import
- **现象**：`grep -rn 'react-query\|useQuery\|useMutation\|QueryClient' src/` → **0 结果**。
  所有数据获取都是手写 `useState` + `useEffect` + `useCallback`，各自实现 loading/error/data 三态 ——
  这正是 P1-3「三态不统一」和 P1-8「跨页污染」的根因。
- **改法**：要么引入 react-query v3 统一数据层（`useQuery` 自带 staleTime / retry / 缓存 / 跨组件共享），
  要么把它从 `package.json` 删掉。当前状态是「装了不用」，给后续开发造成「已经有数据层了」的错觉。

### P3-2　`backendConfig.ts` 是死代码 + WebSocket URL 推导逻辑三处重复且不一致
- **严重度**：P3
- **文件**：`frontend/src/utils/backendConfig.ts:11-27,33-50`、`frontend/src/stores/useAppStore.ts:60-73`、
  `frontend/src/services/websocket.ts:79-93`
- **现象**：
  - `getBackendWebSocketUrl()` / `getBackendHttpUrl()` **零调用点**（grep 全项目只有定义处）。
  - 同一段「从 `window.location` 推导后端地址」的逻辑在 **3 处**各写一遍，
    且不一致：`useAppStore.ts:63,71` 默认端口 `8000`；`websocket.ts:88` 默认端口 `8000`；
    `backendConfig.ts:21,44` 也是 `8000`；但**实际后端在 18082**，
    全靠 `.env.local` 的 `NEXT_PUBLIC_WS_URL=ws://127.0.0.1:18082` 兜住。
    换一台机器、忘了拷 `.env.local`，WebSocket 就会去连 `:8000` —— 而 `websocket.ts:79-93`
    的 URL 拼接逻辑是**另一份**（会补 `/ws` 后缀），`useAppStore` 那份**不会**。
- **改法**：
  1. 删掉 `backendConfig.ts`（或真正用起来并删掉另两处）。
  2. 抽一个 `getWebSocketUrl()` 单一实现，`useAppStore` / `websocket.ts` 共用。
  3. `.env.example` 里把 `NEXT_PUBLIC_API_URL` 注释掉的默认值写清楚（当前示例是
     `http://<backend-host>:8000/api/v1`，而实际后端是 18082，且 `api.ts` 用的是相对路径
     `/api/v1` —— 示例会误导人）。

---

## 先修清单（按 P0 / P1 排序，12 条）

| # | 严重度 | 一句话 | 主要位置 | 截图 |
|---|---|---|---|---|
| 1 | **P0** | **统一响应信封**：后端 `/commerce/*` `/trading/*` `/billing/*` 补 `StandardResponse`，或前端兼容裸体 —— 这一条修好，钱包/今日信号/订阅三个页面同时复活 | `services/api.ts:194-198,306-309` | `02`, `12`, `11` |
| 2 | **P0** | **退出登录删对 key**：`auth_token` → `access_token`（复用 `clearStoredAuth()`），并清 `useAppStore` | `AppLayout.tsx:202` | `04` |
| 3 | **P0** | **管理端充值 4 个接口补 `/commerce` 前缀** —— 收入确认链路 | `admin/payments/page.tsx:84,116,129,143` | `09` |
| 4 | **P0** | **参数调优页去掉 `/api/v1` 双前缀**（3 处）+ 空 catch 要有提示 | `tasks/tuning/page.tsx:170,267,305` | `13` |
| 5 | **P0** | **创建任务失败必须给用户看**（snackbar + 保留表单 + 防重复提交/幂等键） | `tasks/create/page.tsx:415-419` | `08` |
| 6 | **P0** | **`/dashboard/stats` 后端 500 要修**；前端 tier 兜底不许谎报 FREE | `settings/subscription/page.tsx:61-82` + 后端 `dashboard.py` | `11` |
| 7 | **P0** | **登录/注册统一 user 归一函数**，删 `as unknown as`，`User` 字段对齐后端 | `login/page.tsx:53-60`、`register/page.tsx:54` | `01` |
| 8 | **P0** | **服务端数据搬出全局 store**：`useTaskStore` 只留 UI 态；`/backtest` 改后端 `task_type` 过滤 | `useTaskStore.ts`、`backtest/page.tsx:64-98` | `06`, `10` |
| 9 | **P1** | **抽 `<DataState>` 统一三态**（loading / empty / error+重试），替换 9 个列表页的手写分支 | `components/common/DataState.tsx`（新建） | `07`, `06` |
| 10 | **P1** | **`apiRequest` 错误归一**：5 个方法都补 `.catch`，把后端 `message`/`detail` 提成可读文案 + `status` | `services/api.ts:300-365` | `07` |
| 11 | **P1** | **导出/批量删除对齐后端**：导出改 POST + `format:excel` + 真实文件流；批量删除改循环 DELETE；两处错误都要提示 | `taskService.ts:116-131`、`useTaskDetailPage.ts:277` | `10` |
| 12 | **P1** | **15 处 `alert()` 换成 snackbar**；表单校验改就地提示；加 ESLint 禁 `alert` | `models/page.tsx` 等 4 文件 | `08` |

**建议顺序**：#1 单独就能救活 3 个页面 → #2/#3 立刻消除「以为退出了其实没退」和「收入确认不了」两个商业级风险 →
#4/#5 修两个功能整体不可用的页面 → #6~#8 处理数据正确性 → #9~#12 做一致性收口。

---

## 附：其他观察（未计入严重度）

- **`api.ts` 内部日志体系重复**：`services/api.ts:108-120`、`dataService.ts:135-147`、`websocket.ts:40-57`
  各写了一份 `debug/warn/error` 包装器，而项目里已经有 `utils/logger.ts`。三份实现行为还不一致
  （`logger.ts` 的 `error` 在生产环境也打，`api.ts` 的 `error` 无条件打）。
- **`api.ts` 的 `apiLogger.error` 在生产环境不打点任何监控**（`:117-119` 无条件 `console.error`），
  `next.config.js` 的 `removeConsole` 只 exclude `error`/`warn` —— 即生产环境**会**保留这些 error 日志。
  配合后端 60 次/分钟的限流，一次页面访问可能打十几条 console.error，噪音掩盖真问题。
- **后端限流（60/min）对前端页面级并发不友好**：单个页面并行打 3~5 个接口，
  多标签页同时操作很容易撞线；后端返回 429 时前端没有「退避重试」，
  只是把「请求过于频繁，请稍后再试」原样弹给用户。建议后端把限流按**用户**（JWT sub）而非 IP 计，
  或前端加指数退避。
- **后端 500 的 `message` 泄露了内部异常**（如 `'AsyncSession' object has no attribute 'query'`），
  建议 `ErrorHandlingMiddleware._handle_general_error` 已有的「不暴露详细错误」策略
  同样覆盖到**业务层 `HTTPException(500, detail=f"…{exc}")`** 这类写法上。
- **`next.config.js` 的 `typescript.ignoreBuildErrors: true`**：构建时跳过了 TS 检查，
  所以契约错误不会被构建拦住。修完 #1~#8 后建议关掉这个开关。
