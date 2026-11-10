# 平台运营侧前端审计 — 数据管理 / 监控 / 管理端支付 / 设置 / 模型与模板

- 审计日期：2026-10-27
- 审计范围：`frontend/src/app/{monitoring,data,admin/payments,settings,models,templates}`、`frontend/src/components/{monitoring,data,providers}/*`
- 方法：真实浏览器访问 `http://localhost:3000`（Chromium，1280×577 桌面 + 375×812 窄屏），登录 `ronghui@example.com`（管理员 `is_admin=true`），配合后端 `http://127.0.0.1:18082` 直连探测交叉验证。所有问题均在浏览器中实测复现，未修改任何源码。
- 截图目录：`docs/frontend-audit/data/`

---

## 问题清单

### P0

#### P0-1 管理端支付页：订单列表接口路径错误，页面永远空列表且报错 "Not Found"

- **严重度**：P0
- **位置**：`frontend/src/app/admin/payments/page.tsx:84`（`apiRequest.get('/admin/payment/orders...')`）、`:116`（confirm）、`:129`（cancel）、`:143`（config 保存）；文件头注释 `:11-15` 也写的是错误路径
- **现象描述**：前端把管理端接口挂在 `/api/v1/admin/payment/*`，而后端实际把 commerce 路由挂在 `/commerce` 前缀下（`backend/app/api/v1/commerce.py:29` `APIRouter(prefix="/commerce")`，经 `api.py:80` 挂载）。实测：
  - `GET /api/v1/admin/payment/orders?limit=200` → **404** `{"success":false,"message":"Not Found"}`
  - `GET /api/v1/commerce/admin/payment/orders?limit=200` → **200**，返回真实订单数组
  - 页面渲染出红色 `Not Found` 错误条，表格区显示"暂无充值订单"，待确认订单数恒为 0、待确认金额恒为 ¥0.00
- **复现步骤**：管理员登录 → 顶栏 `充值订单` → 页面顶部出现红色 "Not Found" 错误条 → 订单表为空 → 点「刷新」错误不变。
- **具体改法**：把 4 处调用统一改为 `/commerce/admin/payment/orders`、`/commerce/admin/payment/orders/{order_no}/confirm`、`/commerce/admin/payment/orders/{order_no}/cancel`、`/commerce/admin/payment/config`，并同步修正文件头 `:11-15` 的接口注释。
- **截图**：`01-admin-payments.png`

#### P0-2 管理端支付页：收款方式配置表单完全不渲染（响应信封不匹配）

- **严重度**：P0
- **位置**：`frontend/src/app/admin/payments/page.tsx:95`（`apiRequest.get<PayConfig>('/commerce/payment/config')`）→ `frontend/src/services/api.ts:301-315`
- **现象描述**：`/api/v1/commerce/payment/config` 实测返回 **裸对象**（顶层就是 `method_name/payee_account/...`，**没有** `success`/`data` 包装）：
  `{"method_name":"微信转账","payee_account":"ronghui-wx",...}`
  但 `api.ts` 的响应拦截器（`:194`）判定 `if (!data.success) reject(...)`，`apiRequest.get`（`:306`）又要求 `res.data.success` 与 `res.data.data` 存在，因此**任何 commerce 系裸响应都会被判为业务失败**。页面 `loadConfig` 的 `catch` 又把错误吞掉（`:97-99`），导致 `config` 恒为 `null`，`:296` 的 `{config && (...)}` 整块表单不渲染。
  实测页面只有"收款方式配置"标题和说明文字，**0 个 label、0 个 input**（`document.querySelectorAll('label').length === 0`）。管理员**根本无法修改收款账号/收款码/支付说明/启用状态**——这正是该页面的核心职责。
  **影响面不止本页**：实测 `/api/v1/commerce/*` 全系（`balance`、`usage/summary`、`billing/history`、`payment/orders`、`payment/config`）**一律返回裸对象**，全部会被前端 `apiRequest` 判为失败。`/wallet` 充值页因此整体失效——实测页面底部连续弹出两个"加载数据失败"红色错误条，余额/用量/账单/收款指引全部为空：
  - `frontend/src/app/wallet/page.tsx:121-123`（balance / usage/summary / billing/history）
  - `frontend/src/app/wallet/page.tsx:168`（payment/orders）
  - `frontend/src/app/wallet/page.tsx:177`（payment/config）
  用户侧"充值"链路（读收款指引 → 下单 → 查订单）**完全走不通**。
- **复现步骤**：管理员登录 → `/admin/payments` → 滚到页面底部"收款方式配置"，只有一段说明文字，没有任何表单字段。
- **具体改法**：给 commerce 系接口加统一响应包装（后端 `StandardResponse`），或在 `api.ts` 增加一个 `apiRequest.getRaw<T>()` 旁路用于裸响应端点，`wallet/page.tsx` 的 `payment/orders`（`:168`）同样需要处理。同时 `loadConfig` 的 `catch` 不应静默吞错——配置读不到时必须显示错误提示，而不是静默留白。
- **截图**：`01-admin-payments.png`、`16-admin-payments-nonadmin.png`

#### P0-3 数据页本地股票列表：5594 行一次性全量渲染，页面高度 29.8 万像素，无分页/虚拟滚动，浏览器标签页 OOM 崩溃

- **严重度**：P0
- **位置**：`frontend/src/app/data/page.tsx:1120`（`localStocks.map(...)` 全量渲染）、`:1101`（移动端 `MobileStockCard` 全量渲染）、`:1013`/`:1107`（`overflowX:'auto'` 包裹但无高度约束）
- **现象描述**：切到「本地股票列表」Tab（实测 5594 条）：
  - 桌面端：`document.body.scrollHeight = 297,979px`，表格包裹层 `clientHeight = 296,626px`，`maxHeight: none`；切 Tab 时主线程冻结 **>5 秒**（`Runtime.evaluate` 5s 超时）
  - 窄屏 375px：切 Tab 后 `document.body.scrollHeight = 510,086px`，JS 堆从 68MB 涨到 **292MB**；实测渲染进程直接崩溃，tab 变为 `about:blank`（`Target.getTargets` 只剩 `about:blank`）
  - 表格无 `position: sticky` 表头（`stickyCells: 0`），滚动 30 万像素后表头与列名完全不可见
  - 无分页控件、无搜索框、无排序（`clickableTh: 0`）
- **复现步骤**：登录 → `/data` → 点「本地股票列表」Tab → 页面卡死数秒；窄屏下渲染进程崩溃，浏览器标签页变空白。
- **具体改法**：桌面端改为分页（`TablePagination` 或后端 `limit/offset`）或虚拟滚动（`react-window`/`@tanstack/react-virtual`）；移动端 `MobileStockCard` 同样必须分页/虚拟化；同时给表格容器加 `maxHeight` + `position: sticky` 表头，让列名在长列表中始终可见。
- **截图**：`03-data-local-table-5594rows.png`、`15-data-narrow-375-localcards.png`

### P1

#### P1-1 管理员刷新页面后，「充值订单」导航入口消失（Zustand store 未持久化）

- **严重度**：P1
- **位置**：`frontend/src/components/layout/AppLayout.tsx:254` 与 `:358`（`.filter(item => !item.adminOnly || user?.is_admin)`）、`frontend/src/stores/useAppStore.ts:52-109`（store 未使用 `persist` 中间件）、`frontend/src/app/login/page.tsx:46-60`（只有登录时 `setUser`）
- **现象描述**：`useAppStore` 里的 `user` 只在登录页 `setUser()` 时赋值，没有 `persist` 中间件，因此**任何一次硬刷新都会把 `user` 清空**，`user?.is_admin` 变成 `undefined`，`adminOnly` 菜单项被过滤掉。实测（管理员账号）：
  - 表单登录后：导航有「充值订单」✔
  - `location.reload()` 硬刷新后：导航按钮列表变为 `["仪表板","任务管理",...,"系统设置"]`，**「充值订单」消失** ✘，而 `localStorage.user_info` 里 `is_admin` 仍在
  - 顶栏头像下拉菜单里也不显示当前用户名/角色，用户无法确认自己是不是管理员
- **复现步骤**：管理员登录 → 观察顶栏有「充值订单」→ 按 F5 → 该入口消失；想进管理页只能手敲 URL。
- **具体改法**：给 `useAppStore` 加 `persist` 中间件（存 `user`/`isAuthenticated`，键名与已有的 `localStorage.user_info` 统一），或在 `AppLayout` 挂载时从 `localStorage.user_info` 水合 `user`；更稳妥的做法是在 `AppLayout` 统一用 `/api/v1/auth/me` 拉一次当前用户，保证 `is_admin` 与服务端一致。
- **截图**：`04-monitoring-overview.png`（刷新后顶栏无「充值订单」）、`19-usermenu.png`

#### P1-2 管理端支付页：「确认到账」/「取消」点击即生效，无二次确认

- **严重度**：P1
- **位置**：`frontend/src/app/admin/payments/page.tsx:251-269`（两个按钮直接 `onClick={() => confirm(o.order_no)}` / `cancel(o.order_no)`）
- **现象描述**：真实收款链路上的不可逆操作——「确认到账」会**给用户余额加钱并自动升级套餐**（后端 `commerce.py:293-330` `admin_confirm_order` → `PaymentService().confirm_order` → 余额入账 + 套餐升级），「取消」会作废订单。两者均**无任何 Dialog 二次确认、无金额回显、无撤销入口**，误点一次即生效。对比同仓库的模型删除操作有完整的确认弹窗（`frontend/src/app/models/page.tsx:532-561`），标准不一致。
- **复现步骤**：管理端支付页（当前因 P0-1 无数据；修复 P0-1 后即可复现）→ 直接点击订单行右侧「确认到账」→ 立即发出 POST。
- **具体改法**：为 `confirm` / `cancel` 加 `<Dialog>` 二次确认，确认弹窗中回显订单号 + 金额 + 用户 ID + 收款账号，让管理员能核对打款信息后再放行；确认后用 Snackbar 提示「已确认到账，余额已入账」并刷新列表。
- **截图**：`01-admin-payments.png`

#### P1-3 设置页：所有「保存设置」都是假保存——只弹成功提示，不发任何请求，刷新即丢

- **严重度**：P1
- **位置**：`frontend/src/app/settings/page.tsx:107-153`（`handleSaveGeneral` / `handleSaveData` / `handleSaveNotification` / `handleSavePerformance` 全部是 `await new Promise(r => setTimeout(r, 500))` 后 `showMessage('...已保存','success')`）
- **现象描述**：实测拦截 `fetch` + `XMLHttpRequest`，点击「保存设置」后**出站请求数为 0**，但页面弹出绿色 Snackbar「通用设置已保存」。把「系统名称」改成 `AUDIT-TEST-平台` 保存后刷新，值恢复为默认的「股票预测平台」。四个 Tab 全部如此。后端也**根本没有** settings 相关接口（`grep -rn 'settings' backend/app/api/v1/*.py` 无匹配）。这是一个"看起来成功、实际什么也没做"的假反馈，运营侧会误以为配置已生效。
- **复现步骤**：登录 → `/settings` → 通用设置 → 改「系统名称」→ 点「保存设置」→ 弹出「通用设置已保存」绿色提示 → 刷新页面 → 值已丢失。
- **具体改法**：要么接后端（新增 `GET/PUT /api/v1/settings` 并把当前值从后端加载，而非 `useState` 硬编码初值），要么在页面顶部明确标注「本页为前端演示占位，设置不会保存」，并把按钮置灰，杜绝误导。
- **截图**：`06-settings-save-toast.png`、`06-settings-general.png`

#### P1-4 设置页「测试连接」永远失败：直连数据服务受 CORS 阻断

- **严重度**：P1
- **位置**：`frontend/src/app/settings/page.tsx:155-169`（`fetch(\`${dataSettings.dataServiceUrl}/health\`)`）、`:301`（IconButton 无 tooltip/文案）
- **现象描述**：该函数从浏览器直接 fetch 数据服务 `http://127.0.0.1:5002/health`，跨源被 CORS 拦截，catch 后统一弹「无法连接到数据服务」。实测：
  - 浏览器 fetch → `TypeError: Failed to fetch`（CORS）
  - 服务端直连 `http://127.0.0.1:5002/health` → **404**（该 Flask 服务根本没有 `/health` 路由）
  - 而平台自己的代理 `/api/v1/data/status` 报 `is_connected: true, response_time: 2.855ms`
  即：**服务明明是通的，按钮却报不通**。而且这个 IconButton 没有 tooltip、没有文字，用户根本不知道它是干什么的。
- **复现步骤**：`/settings` → 数据设置 Tab → 点数据服务地址右侧的圆形箭头图标 → 弹出「无法连接到数据服务」。
- **具体改法**：改为调用平台代理 `apiRequest.get('/data/status')`（该接口已能正确探活并返回响应时间），把 `/health` 换成后端真实存在的探活路径；同时给该 IconButton 加 `Tooltip title="测试连接"`。
- **截图**：`20-settings-test-connection-fail.png`、`07-settings-data.png`

#### P1-5 监控页：三套独立轮询叠加，刷新反馈缺失（无 loading 态、无时间戳），且系统可用性为硬编码假数据

- **严重度**：P1
- **位置**：`frontend/src/app/monitoring/page.tsx:99-105`（页面级 60s 轮询）、`frontend/src/components/monitoring/SystemHealthCard.tsx:67`（30s 轮询）、`frontend/src/components/monitoring/PerformanceMetricsCard.tsx:60`（60s 轮询）、`monitoring/page.tsx:293-295`（硬编码 `99.9%`）
- **现象描述**：
  1. **轮询叠加**：实测 65 秒内监控页发出 6 个请求（`/monitoring/health`×2、`/monitoring/errors`、`/monitoring/anomalies`、`/monitoring/quality`、`/monitoring/metrics`），三个定时器各自独立、刷新时刻不同步，数据在页面上是"半新半旧"的。后端限流是 60 次/分钟/IP（`backend/app/middleware/rate_limiting.py`，实测 `x-ratelimit-limit-minute: 60`），监控页一个标签页就吃掉约 5 次/分钟，多开标签页很容易触发 429。
  2. **刷新反馈缺失**：点顶栏刷新按钮（`:171`）后，实测无任何 loading 指示（按钮无 spinner，只是 `disabled`），页面也不显示"最后更新时间"。唯一的时间戳藏在 `SystemHealthCard` 底部（"最后更新 / 自动刷新: 30秒"），而页面自身 60s 刷新与"错误分析/异常检测/数据质量"三个 Tab 的数据完全没有刷新提示——用户无法判断看到的数字是什么时候的。
  3. **假数据**：系统可用性卡片写死 `99.9%`（`monitoring/page.tsx:294`），与后端数据无关。
- **复现步骤**：登录 → `/monitoring` → 点顶栏刷新图标 → 页面无任何反馈；观察"系统概览"最右卡片"系统可用性 99.9% 过去30天"，改后端数据也不变。
- **具体改法**：（a）把三个定时器合并为页面级单一调度，或至少统一刷新时刻；（b）刷新按钮加 spinner，页面头部加"最后更新 HH:mm:ss + 下次刷新倒计时"；（c）系统可用性改为后端真实指标，无数据时显示 `--` 而非写死的 99.9%。
- **截图**：`04-monitoring-overview.png`、`05-monitoring-anomalies.png`

#### P1-6 模板页「一键使用」完全失效：跳转参数无人接收，创建页是空白表单

- **严重度**：P1
- **位置**：`frontend/src/app/templates/page.tsx:150-161`（拼 `templateId/strategy/startDate/endDate/initialCash/config` 六个 query 参数）、`frontend/src/app/tasks/create/page.tsx:146-252`（只解析 `type/mode/model_id/rebuild` 四个参数）
- **现象描述**：模板页承诺"选择一个模板，系统会自动填入策略参数"，但创建页的 URL 参数解析里**根本没有** `templateId/strategy/startDate/endDate/initialCash/config` 这几个 key（`grep -rn 'templateId' frontend/src/` 全仓只有模板页自己那一处）。实测点「一键使用」后：
  - URL 变成 `/tasks/create?type=backtest&templateId=ma_cross&strategy=ma_cross_strategy&startDate=2020-01-01&endDate=2024-12-31&initialCash=1000000&config={...}`
  - 但表单实际值：交易策略 = `moving_average`（默认值，非模板的 `ma_cross_strategy`）、回测开始日期 = **空**、结束日期 = **空**、初始资金 = `100000`（非模板的 `1000000`）、策略参数（fast_ma=5/slow_ma=20/ma_type=SMA）**全部未填**
  - 且模板里的策略名 `ma_cross_strategy` / `momentum_strategy` / `multi_factor_strategy` 在后端**不存在**（`grep -rn 'ma_cross_strategy' backend/app/` 无匹配），后端实际策略 key 是 `moving_average` 等
- **复现步骤**：登录 → `/templates` → 任意卡片点「一键使用」→ 跳转到创建页 → 观察表单：日期为空、初始资金 100000、策略为 moving_average、模板参数一个都没进来。
- **具体改法**：在 `tasks/create/page.tsx` 的 `useEffect([searchParams])` 中补上 `templateId/strategy/startDate/endDate/initialCash/config` 的解析（并对 `strategy` 做白名单校验，映射到后端真实存在的策略 key）；或改为把模板定义为后端可查的 `strategy_configs` 记录，用 `config_id` 引用，避免前端硬编码一份与后端脱节的策略清单。
- **截图**：`12-templates-oneclick-result.png`、`11-templates.png`

### P2

#### P2-1 监控页「数据质量」Tab 直接把原始 JSON 字符串糊在卡片里

- **严重度**：P2
- **位置**：`frontend/src/app/monitoring/page.tsx:584-597`（`Object.entries(dataQuality).map(...)`，值为对象时直接 `JSON.stringify(value)`）
- **现象描述**：实测该 Tab 渲染出 `checks {"data_completeness":{"score":1,"status":"pass","message":"发现 5594 个数据文件"},...}`、`issues []`、`recommendations []`、`storage {"root":"/Users/.../parquet/stock_data","total_files":5594,...}` 这种未加工的 JSON 字符串，并且 key 是英文原样（`overall score` / `checks` / `issues` / `recommendations` / `storage`）。同时**服务端绝对路径被暴露到 UI**。空数组 `[]` 也被原样显示，运营看不懂。
- **复现步骤**：`/monitoring` → 点「数据质量」Tab → 看到整屏 JSON 字符串。
- **具体改法**：为 `dataQuality` 定义明确的 TS 接口，按 `checks` 的四个子项（data_completeness / data_freshness / storage_coverage / stock_coverage）渲染成带状态色（pass/excellent/warn/fail）的指标卡；`issues` / `recommendations` 为空时显示"无异常"文案而非 `[]`；服务端路径不应出现在 UI。
- **截图**：`18-monitoring-quality-rawjson.png`

#### P2-2 数据页：数据服务失败时静默显示 "--" 与"暂无数据"，与页头"已连接"自相矛盾

- **严重度**：P2
- **位置**：`frontend/src/app/data/page.tsx:275-289`（汇总卡片）、`:990-1002`（远端列表空态）、`:208-228`（`loadRemoteServiceLogs` 把错误写进 `serviceLog.content`）
- **现象描述**：页头卡片显示「连接状态：已连接 / 响应时间 3.598ms」，但下面的汇总卡片 5 个指标全是 `--`（后端返回 `success:false, message:"数据服务未连接，无法获取汇总统计"`），「远端股票列表」标签页计数为 0 且提示"暂无股票数据 / 请检查远端服务连接状态"——而连接状态明明是"已连接"。日志区只显示一行红字"数据存储初始化失败"，没有任何原因或重试入口。用户在"已连接"和"未连接"两种矛盾提示之间无法判断真实状态。
- **复现步骤**：`/data` → 对比顶部"连接状态：已连接"与下方"股票数 --"、远端列表"暂无股票数据 / 请检查远端服务连接状态"。
- **具体改法**：把"远端服务健康"和"远端数据可用性"拆成两个独立状态；汇总卡片在 `success:false` 时显示后端 `message` 而不是 `--`；远端列表空态按实际原因文案区分（服务未连接 / 服务已连接但无数据 / 数据存储初始化失败），并提供"重试"按钮。
- **截图**：`02-data-top.png`

#### P2-3 数据页：表格无排序、无搜索、无分页，数值列左对齐难以比较

- **严重度**：P2
- **位置**：`frontend/src/app/data/page.tsx:1014-1022`（远端表头）、`:1108-1117`（本地表头）、`:1147-1153`（文件数/记录数/文件大小单元格）
- **现象描述**：本地股票列表 6 列（股票代码/股票名称/数据范围/文件数/记录数/文件大小）实测**全部 `text-align: left`**，包括三个数值列（`文件数`、`记录数`、`文件大小`）——同类数字无法按位对齐，5594 行里比较大小极其吃力。表头无排序控件（`clickableTh: 0`）、无搜索框、无分页。列宽平均分配（165-237px），"数据范围"这类长文本被压到 196px。
- **复现步骤**：`/data` → 本地股票列表 Tab → 观察数值列左对齐、无排序/搜索/分页。
- **具体改法**：给 `文件数/记录数/文件大小` 加 `align="right"`；表头加排序（按记录数/文件大小/代码）；加代码或名称搜索框；配合 P0-3 的分页一起做。
- **截图**：`03-data-local-table-5594rows.png`

#### P2-4 用户菜单「个人资料」「账户设置」指向不存在的路由，均落到 404 页

- **严重度**：P2
- **位置**：`frontend/src/components/layout/AppLayout.tsx:194-199`（`router.push('/profile')` / `router.push('/account-settings')`）、`:310-318`（两个 MenuItem）
- **现象描述**：实测点击顶栏头像 →「个人资料」或「账户设置」，分别跳到 `/profile` 和 `/account-settings`，两个目录在 `frontend/src/app/` 下**都不存在**，页面渲染通用的"404 / 页面不存在 / 您访问的页面不存在或已被移除"（`frontend/src/app/not-found.tsx`）。管理员/用户无法进入个人资料或账户设置。
- **复现步骤**：登录 → 点右上角头像 → 选「个人资料」→ 404 页面。
- **具体改法**：要么补建这两个路由页面（`/profile` 展示账号信息，`/account-settings` 复用 `settings/subscription` 的能力），要么先从菜单中移除，避免暴露死链。
- **截图**：`19-usermenu.png`

#### P2-5 管理端支付页窄屏横向溢出，且表头列宽挤压

- **严重度**：P2
- **位置**：`frontend/src/app/admin/payments/page.tsx:219-228`（6 列表格）、`:179-194`（两个 `minWidth: 180` 的统计卡）
- **现象描述**：375px 窄屏下实测 `document.body.scrollWidth = 412 > clientWidth = 375`，出现横向滚动条。表格 6 列（订单号/金额/下单时间/状态/备注/操作）无 `overflowX` 包裹、无窄屏卡片替代方案，操作列两个按钮会挤爆布局。
- **复现步骤**：窄屏（375px）打开 `/admin/payments` → 页面出现横向滚动条。
- **具体改法**：给表格外包 `<TableContainer sx={{overflowX:'auto'}}>`；参照仓库既有做法（`monitoring/page.tsx:355`、`:370`）补一套 `xs` 断点的卡片列表；统计卡改为 `xs:12, sm:6` 的栅格布局。
- **截图**：`22-admin-payments-narrow-375.png`

#### P2-6 监控页异常/错误数据为空，"异常检测"Tab 的角标数量不可信

- **严重度**：P2
- **位置**：`frontend/src/app/monitoring/page.tsx:205-207`（Tab 角标 `anomalies?.total_anomalies > 0`）、`:260-267`（概览卡片）、`:510-517`（异常详情空态）
- **现象描述**：实测后端 `/monitoring/anomalies` 返回 `total_anomalies: 0`、`anomalies: []`，页面各处都显示 0，异常详情表格显示"暂无异常检测"。后端 `/monitoring/metrics` 返回 `services: {}`（空对象），导致性能指标卡片全部显示 0ms / 0 请求 / 0 错误 / 0 活跃服务——**"0 错误"和"没有数据"在 UI 上无法区分**，用户可能误判为系统健康。同时错误统计 `/monitoring/errors` 也是 0 条，"错误类型 0" 与"未采集到错误"混淆。
- **复现步骤**：`/monitoring` → 系统概览 → 性能指标卡片全 0 → 错误分析 Tab 全 0。
- **具体改法**：区分"无数据"与"值为 0"：后端返回空集合时前端显示"暂无采集数据"而非 0；给 `/monitoring/metrics` 补一个"数据尚未上报"的空态；异常为 0 时明确写"未检测到异常"。
- **截图**：`04-monitoring-overview.png`、`05-monitoring-anomalies.png`

### P3

#### P3-1 `GlobalSnackbar` 从未挂载，全局 Toast 机制形同虚设

- **严重度**：P3
- **位置**：`frontend/src/components/providers/Providers.tsx:17,25`（定义了 `GlobalSnackbar`）vs `frontend/src/app/layout.tsx:24-28`（root layout 直接内联 `MUIThemeProvider/ErrorBoundary/AppLayout`，**没有用** `Providers`）
- **现象描述**：`Providers.tsx` 是个从未被引用的孤儿组件（`grep -rn 'Providers' frontend/src/` 只命中它自己）。`useSnackbarStore` 的 `showError/showSuccess` 被 `app/data/page.tsx:239,255`、`app/dashboard/page.tsx:99,194`、`app/backtest/page.tsx:103,200` 大量调用，但页面里实测 `document.querySelectorAll('.MuiSnackbar-root').length === 0`——**这些错误提示一条都不会显示给用户**。数据页远端/本地列表加载失败时，用户只看到空列表，没有任何文字说明。
- **复现步骤**：任意页面上触发一次 `showError`（例如数据页远端列表加载失败路径）→ 页面无任何 toast 出现。
- **具体改法**：二选一——把 `layout.tsx` 改为 `<Providers>{children}</Providers>`，或直接在 `layout.tsx` 里挂 `<GlobalSnackbar />`；并删除未被使用的 `Providers.tsx` 避免误导。

#### P3-2 `components/data/*` 与 `components/monitoring/{SyncHistoryModal,SyncProgressModal}` 是死代码

- **严重度**：P3
- **位置**：`frontend/src/components/data/EnhancedDataFileTable.tsx`（120 行起）、`frontend/src/components/monitoring/SyncHistoryModal.tsx`、`frontend/src/components/monitoring/SyncProgressModal.tsx`
- **现象描述**：`EnhancedDataFileTable` 实现了完整的文件级管理（搜索/状态筛选/大小筛选/日期筛选/批量选择/**删除文件**/**删除选中**/预览弹窗），但 `grep -rn 'EnhancedDataFileTable' frontend/src/` 只命中它自己和 `components/data/index.ts` 的再导出，**没有任何页面使用它**。同理 `SyncHistoryModal` / `SyncProgressModal` 也只被 `components/monitoring/index.ts` 导出，无人引用。后果有两层：一是数据页因此**完全没有文件级的删除/清理能力**（唯一能删文件的组件是死代码）；二是这些组件里的"删除文件/删除选中"属于危险操作，一旦被接上，需确认它们有二次确认（当前 `EnhancedDataFileTable.tsx` 的 `isDeleteOpen` 弹窗逻辑需复核）。
- **复现步骤**：`grep -rn 'EnhancedDataFileTable' frontend/src/` → 仅 2 处（自身 + index.ts）；`/data` 页无任何文件列表/删除入口。
- **具体改法**：把 `EnhancedDataFileTable` 接入数据页（作为第二个 Tab"数据文件"，承接文件浏览/筛选/删除），并为删除操作补二次确认与影响范围提示；`SyncHistoryModal`/`SyncProgressModal` 若短期不用则删除，避免与 `app/data/page.tsx` 里自建的进度条逻辑重复。

#### P3-3 模型管理页缺少编辑与启停能力，35 条模型无分页

- **严重度**：P3
- **位置**：`frontend/src/app/models/page.tsx:180-290`（操作区仅"发起回测/查看报告/删除"）、`frontend/src/components/models/ModelListTable.tsx:185-233`
- **现象描述**：模型列表每行只有三个操作（就绪态：发起回测 / 查看报告；任意态：删除）。**没有编辑模型**（改名称/描述/超参）、**没有启停/停用模型**（后端 `models` 表也无 enabled 字段）。列表 35 条全部渲染，页高 3936px，无分页/搜索/筛选，35 个同名 `official-alpha360-smoke`（多个同名同类型同状态）无从区分——列表里存在大量重名 smoke 模型，运营无法判断哪个是自己要用的。删除操作虽有确认弹窗（`models/page.tsx:532-561`），但确认文案只有"您确定要删除此模型吗？"，**不回显模型名称**（只有一串 model_id），重名场景下极易误删。
- **复现步骤**：`/models` → 观察列表 35 条大量重名 `official-alphaXXX-smoke` → 每行操作只有 3 个按钮。
- **具体改法**：给模型加 `enabled` 字段与启停开关（后端补字段 + 接口），并补"编辑模型"入口；确认删除弹窗回显模型名称 + 类型 + 创建时间，重名时要求额外确认；列表加搜索与分页。
- **截图**：`10-models.png`、`17-models-create-dialog.png`

#### P3-4 监控页顶部 15 个导航项全部平铺，1280px 下已显拥挤

- **严重度**：P3
- **位置**：`frontend/src/components/layout/AppLayout.tsx:63-145`（15 个 `menuItems`）、`:253-270`（顶栏 `justifyContent:'center'` 平铺渲染）
- **现象描述**：顶栏平铺 15 个带图标的文字按钮（仪表板/任务管理/模型管理/数据管理/系统监控/预测分析/策略回测/策略模板/参数调优/策略信号/今日信号/纸面信号/多策略纸面盘/超参优化/系统设置），且"策略信号/今日信号/纸面信号/多策略纸面盘"是 4 个同族信号页、"策略模板/参数调优/超参优化"是 3 个同族参数页。实测 1280px 宽下顶栏已占满整行，字号被压小；且这些入口**未按角色裁剪**——普通用户与管理员看到完全相同的 15 项（只有"充值订单"一项受 `is_admin` 控制，见 P1-1），免费用户也能看到所有需要付费策略的功能入口。
- **复现步骤**：任意页 → 观察顶栏 15 项平铺；对比普通用户登录后的顶栏（除无"充值订单"外完全一致）。
- **具体改法**：把同族页面收进"下拉分组"（信号类 / 参数类），顶栏只保留 6-8 个一级入口 + 用户菜单；按 `subscription_tier` 裁剪付费功能入口，让不可用的入口显示锁标并说明升级路径，而不是照常展示。

---

## 先修清单（P0 / P1，按优先级）

| # | 严重度 | 问题 | 位置 |
|---|--------|------|------|
| 1 | **P0** | 管理端支付页 4 个接口路径漏了 `/commerce` 前缀 → 订单列表恒空 + 红色 "Not Found" | `app/admin/payments/page.tsx:84,116,129,143` |
| 2 | **P0** | `/api/v1/commerce/*` 全系返回裸对象，前端 `apiRequest` 要求 `{success,data}` 信封 → 收款配置表单与整个 `/wallet` 充值链路（余额/用量/账单/收款指引）全部失效 | `services/api.ts:194,306`；`app/admin/payments/page.tsx:95`；`app/wallet/page.tsx:121-177` |
| 3 | **P0** | 本地股票列表 5594 行全量渲染，桌面页高 29.8 万 px、窄屏页高 51 万 px 且渲染进程 OOM 崩溃 → 必须分页或虚拟滚动 | `app/data/page.tsx:1120,1101` |
| 4 | **P1** | `useAppStore` 未持久化，管理员刷新后「充值订单」入口消失 | `components/layout/AppLayout.tsx:254,358`；`stores/useAppStore.ts:52` |
| 5 | **P1** | 「确认到账」（真实加钱 + 套餐升级）点击即生效，无二次确认、无金额回显 | `app/admin/payments/page.tsx:251-269` |
| 6 | **P1** | 设置页 4 个「保存设置」是假保存：0 个出站请求，只弹成功提示，刷新即丢 | `app/settings/page.tsx:107-153` |
| 7 | **P1** | 设置页「测试连接」直连数据服务被 CORS 拦截，且 `/health` 在服务端不存在 → 服务正常也报不通 | `app/settings/page.tsx:155-169` |
| 8 | **P1** | 模板「一键使用」跳转的 6 个参数创建页一个都不解析 → 跳转后是空白表单，且模板策略名后端不存在 | `app/templates/page.tsx:150-161`；`app/tasks/create/page.tsx:146-252` |
| 9 | **P1** | 监控页 3 套独立轮询叠加（65s/6 请求），无 loading 态、无刷新时间戳，"系统可用性 99.9%" 是硬编码假数据 | `app/monitoring/page.tsx:99-105,294`；`components/monitoring/SystemHealthCard.tsx:67` |

---

## 总体评价

这组运营侧页面的**骨架是完整的**（表单分组清晰、状态用 Chip 区分、移动端有卡片降级方案、模型删除有确认弹窗），但**"看起来能用"和"真的能用"之间存在系统性落差**：管理端支付页因两处接口契约错误（路径漏 `/commerce` 前缀 + 响应信封不匹配）实际上完全不可用——订单看不到、收款配置改不了，而这两处恰恰是资金链路最不容出错的地方；同一信封缺陷还让整个 `/wallet` 充值链路（余额/用量/账单/收款指引）整体失效；数据页 5594 行全量渲染已实测导致浏览器标签页崩溃；设置页的四个"保存成功"是纯前端假反馈。叠加 Zustand store 未持久化导致管理员入口刷新即消、模板"一键使用"跳转参数无人接收这两处断链，运营侧的实际可用性明显低于页面所呈现的完成度。建议按 P0 → P1 顺序修复，其中 P0-1 与 P0-2 属于资金链路阻断，应最优先处理。
