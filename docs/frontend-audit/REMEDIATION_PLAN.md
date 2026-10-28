# 股票平台前端整改 · 诊断与方案

审计时间：2026-10-27 07:50 ~ 10:31
审计方式：10 个子代理并行只读审计（165 张实机截图）+ 父会话独立复验
报告位置：`docs/frontend-audit/{layout,landing,signal,backtest,design,mobile,state,data}/FINDINGS.md`

---

## 一、核心结论

**这不是"排版不好看"的问题。平台的付费转化链路和核心赚钱页面，目前大面积处于"完全不可用"状态。**

子代理共报出 40+ 条 P0。父会话逐条独立复验（真实浏览器 + 真实 API + 真实 token），确认四条最关键的 P0 **全部复现**：

| 页面 | 实测结果 | 复验方式 |
|---|---|---|
| `/today-signal` | 显示"请求失败"，付费用户被标成"体验版" | 浏览器登录后访问，抓 DOM |
| `/wallet` | 余额恒为 0.00 元，账单/用量全空 | 同上 |
| `/admin/payments` | 红色 "Not Found"，订单列表恒空 | 同上 |
| `/tasks/tuning` | 基准任务下拉框永远为空 | 同上 |

---

## 二、根因：三条独立的契约断裂

### 根因 A —— 响应信封不统一（影响面最大）

前端 `src/services/api.ts:194` 的响应拦截器：
```ts
if (!data.success) { return Promise.reject(new Error(data.message || '请求失败')); }
```
**硬性要求** `{success, data, message, timestamp}` 信封。

后端实际情况：

| 模块 | 路由数 | 返回形式 |
|---|---|---|
| `commerce.py` | 15 | 8 个 `response_model=`（裸 Pydantic）、7 个 RAW dict |
| `trading.py` | 4 | **全部 RAW dict** |
| `dashboard.py` | 2 | StandardResponse ✅ |
| 其余 40 个模块 | — | StandardResponse ✅ |

即 `apiRequest` 打的 `/commerce/*`、`/trading/*` 共 19 个端点，**响应全部是裸对象 → `data.success` 为 undefined → 拦截器一律 reject**。
`/trading/strategies` 实测返回 `{strategies, count, tier, tier_label, ...}`，无 `success` 字段 → 前端判定失败。

### 根因 B —— 路由路径前缀丢失

后端管理端路由实际挂载在 `/commerce/admin/payment/*`（`commerce.py:281-356`，`APIRouter(prefix="/commerce")`）。

前端 `admin/payments/page.tsx:84,116,129,143` 调用的是 `/admin/payment/orders` —— **漏了 `/commerce` 前缀**，4 个接口全部 404。

同理 `tasks/tuning/page.tsx` 出现 `/api/v1/api/v1/...` 双前缀。

### 根因 C —— 登录响应形态不统一

后端 `/auth/login` 返回**裸** `{access_token, refresh_token, token_type, user}`，前端 `api.ts` 大部分地方读 `data.data.access_token`（信封路径），只有登录处读 `data.access_token`。
结果：登录后 token 存的是正确值，但**任何依赖信封路径读 token 的刷新/校验逻辑都会拿到 undefined**。
`user_info` 里 `subscription_tier` / `is_admin` 字段命名也前后端不一致。

### 根因 D —— dashboard/stats 500

实测 `/api/v1/dashboard/stats` 返回 **HTTP 500**，导致订阅页 tier 回落到硬编码 `'free'` → enterprise 用户在界面上被显示为 FREE，且付费墙误判。

---

## 三、P0 清单（按"能不能收到钱 / 能不能让用户赚钱"排序）

| # | 严重度 | 问题 | 位置 | 修复方向 |
|---|---|---|---|---|
| 1 | P0 | `/trading/*` 裸对象 → 今日信号页 100% 报错 | `trading.py:47,85,133,165` | 后端 4 个端点包 `StandardResponse` |
| 2 | P0 | `/commerce/*` 裸对象 → 钱包/充值/订阅全链路瘫 | `commerce.py` 8 个 `response_model=` 端点 | 统一包信封 |
| 3 | P0 | 管理端 4 接口漏 `/commerce` 前缀 → 收入确认链路断 | `admin/payments/page.tsx:84,116,129,143` | 前端补前缀 |
| 4 | P0 | 充值金额校验永远不通过，任何金额提交不了 | `wallet/page.tsx` 校验逻辑 | 修校验规则 |
| 5 | P0 | 登录响应裸体 → token 字段错位 | `api.ts` vs `auth.py` | 后端包信封，前端兼容读取 |
| 6 | P0 | `dashboard/stats` 500 → enterprise 被显示成 FREE | `dashboard.py:31` | 修后端 500 |
| 7 | P0 | 退出登录删错 key（`auth_token` vs `access_token`）→ 会话未真正登出 | `AppLayout.tsx:200-204` | 改用 `clearStoredAuth()` |
| 8 | P0 | 顶栏 16 项平铺，1200–1440px 顶栏 159px、标签竖切 4 行 | `AppLayout.tsx:63-145,245-271` | 改侧边栏 + 分组 |
| 9 | P0 | `/profile`、`/account-settings` 路由不存在，点击必 404 | `AppLayout.tsx:195,198` | 建页或移除 |
| 10 | P0 | 管理员入口刷新后消失（zustand 未 persist） | `useAppStore.ts:52` | 加 persist |
| 11 | P0 | `hsl(var(--border))` 引用从未定义的变量 → 全站 `*` 边框退化为 currentColor | `globals.css:5-8` | 补 `--border` 定义 |
| 12 | P0 | 涨跌色绿涨红跌 vs 红涨绿跌同屏自相矛盾 | `muiTheme.tsx:26-35` 等 9 处 | 新增 `palette.semantics.up/down` |
| 13 | P0 | 本地股票列表 5594 行全量渲染，页面高 29.8 万 px，渲染进程 OOM | `data/page.tsx:1120,1101` | 分页/虚拟滚动 |
| 14 | P0 | 交易策略下拉框空白，提交时静默用 `moving_average` | `tasks/create/page.tsx:126,91` | 裸 fetch → apiRequest + 校验 |
| 15 | P0 | 回测列表收益/夏普列永远是 `-`（读 `task.result`，API 给 `result_summary`） | `backtest/page.tsx:210-222,432-441` | 改读 `result_summary` |
| 16 | P0 | 参数调优台 URL 双前缀 `/api/v1/api/v1/*` | `tasks/tuning/page.tsx:170,267,305` | 去重复前缀 |
| 17 | P0 | `/admin/payments` 与 `/today-signal` 移动端 8 列表被 `overflow:hidden` 硬裁 | `paper-signal:64`、`today-signal:524` | 加 `overflowX:auto` |
| 18 | P0 | 全站零 `safe-area-inset`，刘海屏遮挡顶栏/footer | `globals.css`、`AppLayout.tsx` | 加 `--safe-*` 变量 |
| 19 | P0 | `/trading/*` 缺信封 → 付费用户被误判"体验版"并推去重复付费 | `today-signal/page.tsx:172-173` | tier 初始值改 null |

---

## 四、整改分阶段方案

### 阶段 1：修契约（后端为主，前端兼容为辅）— 收入链路，最高优先级
1. 后端 `trading.py` 4 个端点 + `commerce.py` 8 个裸端点统一包 `StandardResponse`
2. 后端 `auth.py` 登录/注册响应包信封
3. 修 `dashboard/stats` 的 500
4. 前端 `api.ts` 响应拦截器改为**双向兼容**（有 `success` 用信封路径，无 `success` 视作裸 payload），并集中到 `normalizeResponse`
5. 前端 `admin/payments/page.tsx` 4 处补 `/commerce` 前缀
6. 前端 `tasks/tuning` 去双前缀
7. 前端 `tasks/create` 策略下拉改用 `apiRequest` + 非空校验
8. 前端 `backtest` 改读 `result_summary`
9. 钱包充值金额校验规则修正
10. 退出登录改用 `clearStoredAuth()`

**验收**：浏览器实测 4 个 P0 页面全部有真实数据；`/today-signal` 显示 enterprise 档位；`/admin/payments` 能看到订单并能确认到账。

### 阶段 2：导航壳重构 + 会话 — 可用性骨架
- 16 项顶栏 → 侧边栏（分组：概览 / 研究 / 信号 / 运营 / 账户）
- zustand persist
- 面包屑
- 补/删 `/profile`、`/account-settings`
- 公开页（landing/login/pricing）拆出 `(app)` 路由组

### 阶段 3：设计系统统一
- 补 `--border`，建立 `src/theme/tokens.ts` 唯一真相源
- 新增 `palette.semantics.up/down`，全站涨跌色统一
- 移除 HeroUI（0 引用，128KB 死 CSS）、移除 recharts（仅 1 处引用）
- antd → MUI 迁移（4 个页面）
- 统一 loading / empty / error 三态组件
- 统一圆角 3 档、阴影 2 档

### 阶段 4：移动端 + 性能
- `safe-area-inset` + 底部 Tab 栏
- 表格 `overflowX:auto` + 移动端卡片视图
- 路由级 `loading.tsx` 骨架屏
- 5 个核心路由加 segment loading
- 虚拟滚动（5594 行）
- 焦点可见性、`aria-label`

---

## 五、当前基线数据（实测）

- `npm run type-check`：0 错误
- `npm run lint`：59 warning / 0 error（15 文件）
- `npm run test:ci`：15 suites / 139 tests 全通过
- 覆盖率：**All files Stmts 12.54% / Branch 10.98%**（88 个文件 0% 覆盖）
- `next build` First Load JS：`/wallet` **372 kB**（最重）、`/signals` 278 kB、`/settings/subscription` 256 kB
- 重复代码：`SignalHistoryTable` 与 `TradeHistoryTable` **562 行完全相同**
- HeroUI 注入 128KB 死 CSS，占全局 CSS **56%**

---

## 六、整改后验收门禁

```bash
cd frontend
npm run type-check    # 必须 0 错误
npm run lint          # 必须 0 error，warning 收敛
npm run test:ci       # 必须全通过
npm run build:check   # 必须成功
# 浏览器实测（4 条 P0 全部有真实数据）
# 移动端 390×844 / 360×800 无横向溢出
```
