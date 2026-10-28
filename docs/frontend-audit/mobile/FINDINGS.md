# 移动端适配 / 可访问性 / 交互便捷性 —— 只读审计报告

- 审计对象：`~/Projects/stock-platform/frontend`
- 审计方式：browser_exec + CDP `Emulation.setDeviceMetricsOverride`（390×844 / 360×800 两档）+ `Emulation.setTouchEmulationEnabled`（`pointer:coarse` 生效，MUI 的 `@media (pointer: coarse)` 触摸尺寸规则才会真实命中）
- 环境：前端 dev `http://localhost:3000`，后端 `http://127.0.0.1:18082`
- 截图目录：`docs/frontend-audit/mobile/`（编号前缀见各条）
- 本次未修改任何源码

> 重要前置事实（影响多条结论）：后端登录接口是 **email** 字段，账号 `ronghui` 的 email 是 `ronghui@example.com`（`backend/app/api/v1/auth.py:172` 按 `User.email` 精确匹配，不接受 username）。用任务里给的 `ronghui` 作为 email 登录会 401。

---

## 一、问题清单

### P0

---

#### [P0-1] `/trading/*` 接口缺少 `{success, data}` 响应信封，前端 `apiRequest.get` 必然抛错 → 今日信号页在移动端 100% 不可用

- **文件路径:行号**
  - 前端消费方：`frontend/src/app/today-signal/page.tsx:186`（`/trading/strategies`）、`:208`（`/trading/signal`）、`:212`（`/trading/signal-quality`）
  - 前端契约校验：`frontend/src/services/api.ts:194-198`（响应拦截器 `if (!data.success) reject`）、`:306-312`（`apiRequest.get` 再判一次 `!res.data.success`）
  - 后端返回：`backend/app/api/v1/trading.py:75-80`（`list_signal_strategies` 直接 `return {...}`，未包 `StandardResponse`）
- **现象描述**
  `/api/v1/trading/strategies` 实测返回的顶层 key 是 `['strategies','count','tier','tier_label','allowed_strategies','delay_days']`，**没有 `success` 字段也没有 `data` 字段**；`/api/v1/trading/signal?strategy=honest_42f` 同理（顶层是 `strategy/display_name/...`）。而 `api.ts:194` 的拦截器判定 `!data.success` 即 reject，`api.ts:306` 二次判定 `!res.data.success` 再 throw。两条链路都必然失败，前端 `catch` 里 `setError(e.message)` 把后端的业务消息丢掉，最终在页面上渲染成通用文案 **「请求失败」**。
  对比验证：同一后端的 `/api/v1/tasks`、`/api/v1/models`、`/api/v1/backtest/tasks`、`/api/v1/monitoring/overview` 都返回 `['success','message','data','timestamp']`——**只有 `/trading/*` 系列是裸返回**，这是一处前后端契约不一致，不是网络问题。
  后果：今日信号页（移动端主入口之一）在任何视口下都只显示「刷新 / 请求失败 / 体验版 / 升级解锁全部信号线 / 策略」这一行空壳，**买入/卖出明细表、股数/金额/费用/分数、因子 chips 全部不渲染**。移动端用户因此完全看不到「为什么买它」。
- **复现步骤**
  1. 视口 390×844（`pointer:coarse` 开），登录后打开 `http://localhost:3000/today-signal`
  2. 等待 3~7s
  3. 观察：页面主体只有 `今日信号 / 收盘后自动生成… / 刷新 / 请求失败 / 体验版 / 升级解锁全部信号线 / 策略`，**无任何表格**
  4. 佐证：`curl -H "Authorization: Bearer $TOK" http://127.0.0.1:18082/api/v1/trading/strategies | python3 -c "import sys,json;print(list(json.load(sys.stdin).keys()))"` → `['strategies','count',...]`，无 `success`
- **具体改法**
  二选一，优先改后端（不破坏其他 20+ 个已符合契约的接口）：
  - 后端：在 `backend/app/api/v1/trading.py` 的 `list_signal_strategies` / `get_daily_signal` / `signal_quality` 三个 handler 上套 `StandardResponse(success=True, data=<原返回值>)`，与 `error_handling.py:67` 保持一致；
  - 前端（若后端不便改）：在 `frontend/src/services/api.ts:301` 的 `apiRequest.get` 里增加「若响应体无 `success` 字段则视为裸 payload 直接返回」的兼容分支，并把 `today-signal/page.tsx` 的 catch 改为优先展示 `e.message` 而非通用「请求失败」。
  另外 `api.ts:194` 的拦截器 reject 时把 `data.message` 丢掉了（只用 `data.message || '请求失败'`），建议把后端返回的 `detail` 一并透出，便于弱网/降级时定位。
- **截图文件名**：`03-today-signal-390.png`、`08-today-signal-360.png`

---

#### [P0-2] 7~8 列明细表在 390px 下被 Card 的 `overflow:hidden` 直接裁掉，且不可横向滚动 —— 分数/费用列永久不可见

- **文件路径:行号**
  - `frontend/src/app/paper-signal/page.tsx:64-73`（`<Table size="small">`，7 列，无 `overflowX` 包裹、无 `MuiTableContainer-root`）
  - `frontend/src/app/paper-portfolio/page.tsx:220`（`<Table size="small">`，7 列，同样无包裹）
  - `frontend/src/app/today-signal/page.tsx:524`、`:602`、`:640`（3 张 8 列表，同样无包裹）
- **现象描述**
  实测（390px）：`<table>` 实际渲染宽度 **432px**（paper-signal）/ **380px、385px**（paper-portfolio 两张表），而外层 `MuiPaper-root`（Card）的 `overflow` 计算值是 **`hidden`**、宽度仅 **318px**。MUI Card 默认 `overflow:hidden`，于是表格右侧 **131px（paper-signal）**、**78px / 83px（paper-portfolio）** 被永久裁掉。
  关键点是**没有 `overflowX:'auto'` 的祖先**，所以 `scrollWidth > clientWidth` 也不可滚——用户既看不到、也拖不动。裁掉的正是最右侧的 **「分数」列**（paper-signal 最后一列表头实测 `right=485` vs 裁切边界 `354`）。
  同时，文档级 `document.documentElement.scrollWidth` 仍是 390，**不会触发页面级横向滚动条**——这意味着这个问题在「有没有横向滚动条」这一常见检查项里完全隐形，但数据是真的丢了。
  列宽被压缩到无法阅读：实测「名称」列总宽仅 **30px**（内容区约 14px），表头「名称」两字被压成 **4 行**（表头行高 73px），数据行「萃华退」3 个汉字每行只放得下 2 个字、折成 2 行，整行行高 **85px**。列宽分配严重失衡：代码列 92px、名称列 30px。
- **复现步骤**
  1. 视口 390×844（`pointer:coarse` 开）
  2. 打开 `http://localhost:3000/paper-signal`（有真实数据）
  3. 观察表格：最右侧「分数」列被裁掉一半，表格右边缘硬生生切在 Card 边界上，无法横向拖动
  4. 复现 `http://localhost:3000/paper-portfolio` → 「卖出(6)」「买入(18)」两张表同样右侧被裁
  5. DevTools 验证：选中 table，Console 执行
     `[...document.querySelectorAll('table')].map(t=>({w:t.getBoundingClientRect().width, parentOv:getComputedStyle(t.parentElement).overflowX, hasScrollAncestor: !!t.closest('[style*=overflow]')}))`
     → 宽度 432 / `visible` / 无滚动祖先
- **具体改法**
  1. **首选**：为这三类交易明细表补移动端卡片视图。仓库里已有成熟范式——`frontend/src/components/mobile/MobileSignalCard.tsx`（`signals/page.tsx:703` 用 `display:{xs:'block',md:'none'}` + `display:{xs:'none',md:'block'}` 桌面表格双轨）。按同样模式新增 `MobileTradeCard`（代码+名称做标题行，价格/股数/金额/费用/分数做两列 key-value 网格，因子 chips 放在底部），在 `paper-signal`、`paper-portfolio`、`today-signal` 三页加 `xs` 卡片 / `md` 表格双轨。
  2. **兜底（必须同时做）**：所有 `<Table>` 外层包 `<Box sx={{ overflowX:'auto', WebkitOverflowScrolling:'touch' }}>` 或改用 MUI 的 `<TableContainer>`，确保任何断点下表格都能横向滚动，不被 Card 的 `overflow:hidden` 静默裁切。
  3. 表格加 `sx={{ minWidth: 640 }}`（`paper-portfolio` 的定价对照表 `frontend/src/app/pricing/page.tsx:366` 已是这个做法，可直接照抄），配合 2 的滚动容器，避免浏览器把列压到 30px。
  4. 「名称」列加 `sx={{ minWidth: 72, whiteSpace:'nowrap' }}`，「代码」列 `sx={{ fontFamily:'monospace', whiteSpace:'nowrap' }}`。
- **截图文件名**：`24-paper-signal-390.png`、`12-paper-portfolio-table-390.png`、`19-paper-portfolio-table-crop-390.png`

---

#### [P0-3] 全站零 `safe-area-inset` 处理 —— 刘海屏/灵动岛机型上顶部吸顶栏与底部 footer 会被系统栏遮挡

- **文件路径:行号**
  - `frontend/src/app/globals.css`（全文 155 行，无任何 `safe-area`）
  - `frontend/src/components/layout/AppLayout.tsx:216-219`（`AppBar position="sticky"`，无 `paddingTop: env(safe-area-inset-top)`）
  - `frontend/src/components/layout/AppLayout.tsx:405-420`（`footer`，`py: 2` 固定 16px，无 `env(safe-area-inset-bottom)`）
  - `frontend/src/theme/mobileOverrides.ts:181-193`（`MuiAppBar` 覆盖里只有 `minHeight:56` / `paddingLeft/Right`，同样没有 top inset）
- **现象描述**
  实测 `document.documentElement.outerHTML.includes('safe-area')` → **false**；`getComputedStyle(footer).paddingBottom` → `16px`，不含 `env()`；`header` 的 `paddingTop` → `0px`。
  影响：
  - iPhone 14/15/16 Pro 等机型浏览器顶部有 **安全区（~47-59px）**，横向滚动时 `position:sticky` 的 AppBar 会顶到刘海/灵动岛下面，「仪表板」标题被挖掉；
  - 底部 home indicator 区域（~34px）会压住 footer 的版权行。因为 footer 是文档流末尾、页面通常可滚到底，用户滚动到底时 footer 文字正好落在 home indicator 上。
  - 全站**没有底部导航/底部操作条**，所以「底部导航遮挡内容」这一项不成立（见 [P2-9]）。
- **复现步骤**
  1. 真机 Safari / Chrome 打开任意页面，390×844 逻辑宽（iPhone 12/13/14 模拟器亦可）
  2. 上下滚动到顶/到底，观察吸顶栏顶部与 footer 底部
  3. 代码级验证：`grep -rn "safe-area" frontend/src/` → **无结果**
- **具体改法**
  在 `frontend/src/app/globals.css` 增加：
  ```css
  :root {
    --safe-top: env(safe-area-inset-top, 0px);
    --safe-bottom: env(safe-area-inset-bottom, 0px);
    --safe-left: env(safe-area-inset-left, 0px);
    --safe-right: env(safe-area-inset-right, 0px);
  }
  ```
  然后：
  - `AppLayout.tsx:218` 的 AppBar `sx` 加 `pt: 'var(--safe-top)'`；`mobileOverrides.ts` 的 `MuiAppBar` 覆盖里 `minHeight: 56` 改为 `minHeight: 'calc(56px + var(--safe-top))'`；
  - `AppLayout.tsx:410` 的 footer `py: 2` 改为 `py: 2` + `pb: 'max(16px, var(--safe-bottom))'`；
  - `AppLayout.tsx:395` 的 main `p: { xs: 1.5, sm: 2, md: 3 }` 改为 `px: 'max(12px, var(--safe-left))'`（横向同理，避免横屏时贴边）；
  - 底部若有新增固定条（[P2-9] 建议加），必须 `paddingBottom: var(--safe-bottom)`。
  另外 `frontend/src/app/layout.tsx:13-18` 的 viewport meta 建议加 `viewportFit: 'cover'`，否则 `env()` 在部分浏览器里恒为 0。
- **截图文件名**：`02-dashboard-390.png`（顶栏）、`07-dashboard-360.png`（底栏）

---

### P1

---

#### [P1-1] 全站无可见焦点指示器（focus outline 恒为 0px）—— 键盘/辅助技术用户无法定位当前焦点

- **文件路径:行号**
  - `frontend/src/theme/muiTheme.tsx:89-105`（`components` 只覆盖了 `MuiButton` / `MuiCard`，**没有任何 `focusVisible` / `outline` 相关覆盖**）
  - `frontend/src/theme/mobileOverrides.ts`（全文 205 行，同样没有 focus 相关规则）
  - `frontend/src/app/globals.css`（无 `:focus-visible` 规则）
- **现象描述**
  逐个 Tab 遍历后 `getComputedStyle(activeElement)` 实测：
  - `outlineWidth: "0px"`、`outlineStyle: "none"`、`boxShadow: "none"`——**所有可聚焦元素都是如此**；
  - `:focus-visible` 匹配为 `true`（即浏览器确实把元素标记为键盘焦点），但**没有任何视觉样式响应它**；
  - 页面上确实存在 MUI 注入的 `.Mui-focusVisible` 样式表规则，但那些规则只对 `MuiButton-root.Mui-focusVisible` 生效且给的是 `box-shadow` Elevation 阴影（视觉上与 hover 难以区分）；**`MuiIconButton` 完全没有 `focusVisible` 规则**——顶部导航的汉堡菜单、铃铛、头像三个 IconButton 键盘焦点时是**完全不可见的**。
  实测 Tab 序列（390px）：`menu`(IconButton) → `(无名称 IconButton)` → `(无名称 IconButton)` → 循环。第 2、3 个是**既没有可访问名也没有焦点框**的元素。
  Tailwind 的 `focus-visible:outline-2` 工具类在产物里存在，但没有任何组件用到它们。
- **复现步骤**
  1. 视口 390×844，打开 `http://localhost:3000/dashboard`
  2. 用键盘（Tab）逐个移动焦点
  3. 观察：焦点落在顶部三个图标按钮上时，屏幕上**看不到任何变化**
  4. 验证：Console 执行 `[...document.querySelectorAll('button')].map(b=>({fv:b.matches(':focus-visible'), o:getComputedStyle(b).outlineWidth}))` → 全部 `0px`
- **具体改法**
  1. 在 `frontend/src/theme/muiTheme.tsx` 的 `createTheme` 中增加全局默认：
     ```ts
     components: {
       MuiButtonBase: {
         styleOverrides: {
           root: {
             '&.Mui-focusVisible, &:focus-visible': {
               outline: '2px solid',
               outlineColor: 'primary.main',
               outlineOffset: '2px',
             },
           },
         },
       },
       // ...
     }
     ```
     这一处同时解决 Button / IconButton / Switch / Chip 的焦点框。
  2. 对 MUI 覆盖不到的第三方组件（antd，见 [P1-2]）在 `globals.css` 补：
     ```css
     .ant-btn:focus-visible, .ant-input:focus-visible, .ant-select:focus-visible {
       outline: 2px solid #1976d2; outline-offset: 2px;
     }
     ```
  3. 深色底（`pricing/page.tsx:371` 的 `#0f172a` 表格头）上焦点框需换成 `outlineColor: '#90caf9'`。
  4. 若要更彻底，可在 `AppLayout.tsx` 顶部加一个「跳到主内容」skip link（`<a href="#main-content" className="sr-only-focusable">`），同时给 `main` 加 `id="main-content"`。
- **截图文件名**：`15-drawer-390.png`（顶部 IconButton 区域）、`18-signals-form-390.png`

---

#### [P1-2] antd 组件完全绕开 MUI 主题：充值弹窗输入框 `font-size:14px` → iOS Safari 聚焦时自动放大，输入法键盘顶飞整个弹窗

- **文件路径:行号**
  - `frontend/src/app/wallet/page.tsx:378`（`<Input type="number" ... addonAfter="元" />`）
  - `frontend/src/app/wallet/page.tsx` 全文（`antd` 的 `Button`/`Input`/`Table`/`Descriptions` 混排）
  - `frontend/src/theme/mobileOverrides.ts:109-119`（`MuiTextField` 的 `fontSize:16px` 只对 MUI 组件生效，antd 不受影响）
  - `frontend/src/app/layout.tsx`（**无 `ConfigProvider`**，antd 走默认 token）
- **现象描述**
  实测 wallet 页 `.ant-input` 的 `fontSize` = **14px**（MUI 的 `TextField` 实测是 16px）。
  iOS Safari 的行为是：**任何 `font-size < 16px` 的输入框获得焦点时，页面会自动放大（zoom in）且用户无法手动缩回**，直到输入框失焦。实际后果是：用户在「充值」弹窗里点金额输入框 → 页面整体放大 + 软键盘弹出 → 弹窗里的「创建充值订单」按钮和「取消」按钮被顶出视口，用户需要手动滚动/缩放才能提交。
  同页的 antd `Button` 实测高度 **32px**（刷新 / 充值 / 查看全部），远低于 44px 触摸目标标准。
- **复现步骤**
  1. 视口 390×844（`pointer:coarse` 开），打开 `http://localhost:3000/wallet`
  2. 点顶部「刷新」/「充值」/「查看全部」三个按钮 → 实测高度均 32px
  3. 点「充值」→ 弹窗里点金额输入框（placeholder「例如：100」）→ 实测 `font-size: 14px`
  4. 真机（iPhone Safari）复现：聚焦输入框时页面自动放大、键盘弹出后提交按钮不可见
  5. 验证：`document.querySelector('.ant-input')` → `getComputedStyle(el).fontSize` = `14px`
- **具体改法**
  1. 在 `frontend/src/app/layout.tsx` 包一层 antd `ConfigProvider`，统一 token（这是根治，优先做）：
     ```tsx
     <ConfigProvider theme={{
       token: { fontSize: 16, controlHeight: 44, borderRadius: 8 },
       components: { Input: { controlHeight: 44 }, Button: { controlHeight: 44 } },
     }}>
       <AntdApp>{children}</AntdApp>
     </ConfigProvider>
     ```
  2. 若暂时不改布局，至少给 `wallet/page.tsx:378` 的 `Input` 加 `style={{ fontSize: 16 }}`，并给该页所有 antd `Button` 加 `style={{ minHeight: 44 }}`。
  3. 更彻底的做法（推荐）：把 wallet / register / reset-password / settings-subscription 四页的 antd 组件统一迁到 MUI，消灭双设计系统（现状是 MUI v7 + antd v6 + HeroUI + Tailwind 四套并存，`frontend/package.json:33-43`）。
- **截图文件名**：`06-wallet-390.png`、`11-wallet-360.png`、`28-wallet-recharge-modal-390.png`

---

#### [P1-3] 弱网无进度反馈：`/signals` 全市场信号生成是「while 循环串行翻页」，移动端可能长时间无任何反馈

- **文件路径:行号**
  - `frontend/src/app/signals/page.tsx:318-347`（`while (hasMore)` 循环逐页 `fetchPage`，`pageSize` 默认 200，最坏情况翻数十页）
  - `frontend/src/app/signals/page.tsx:704-710`（移动端 loading 只有 `<CircularProgress size={24}/>` + 文案「计算中...」，**无页码/百分比进度**）
  - `frontend/src/app/signals/page.tsx:768-773`（桌面端文案「计算中（全市场建议分页/分批）...」）
  - `frontend/src/app/signals/page.tsx:550`（按钮文案 `loading ? '生成中...' : '生成信号'`）
- **现象描述**
  `generateAll` 默认 **true**（`signals/page.tsx:72`），点「生成信号」后前端会**串行**把全市场信号一页页拉完。移动端只有一句「计算中...」和一个小转圈：
  - 没有「已拉取 3/12 页」这类进度，用户无法判断是在跑还是卡了；
  - 没有超时/取消按钮，用户只能干等或刷新页面（刷新会丢掉已经拉到的数据）；
  - 弱网下（TTFB 0.95s~23.4s）单页就要几十秒，整体耗时不可预期。
  结合已知的国内访问 pages.dev 的 TTFB 剧烈波动，这是移动端最容易劝退的场景。
  其他页面同样缺「长任务进度」：`backtest`、`optimization` 提交回测/调优任务后只有 spinner。
- **复现步骤**
  1. 视口 390×844，打开 `http://localhost:3000/signals`
  2. 选一个策略 → 点「生成信号」
  3. 观察：转圈 + 「计算中...」，无进度、无取消
  4. 弱网复现：DevTools → Network → Slow 3G，耗时成倍放大
- **具体改法**
  1. 在 `fetchLatest` 的 `while` 循环里累加 `donePages` / `fetchedCount`，写入 state；移动端 loading 区（`signals/page.tsx:704-710`）改为
     `计算中：已获取 {fetched} / {total}（第 {donePages} 页）` + `<LinearProgress variant="determinate" value={pct} />`。
  2. 加一个「停止」按钮，置 `cancelledRef.current = true`，循环内 `if (cancelled) break`，并保留已拉到的部分结果。
  3. 加超时兜底：单页 `apiRequest.get` 传入 `timeout: 15000`，超限给出可重试的错误提示。
  4. 抽一个通用 `<TaskProgress />` 组件（`src/components/common/`），供 backtest / optimization / models 复用。
- **截图文件名**：`04-signals-390.png`、`09-signals-360.png`

---

#### [P1-4] 错误态缺少重试入口：`today-signal` / `signals` / `wallet` 的失败提示只有一句文案，用户被困在死路

- **文件路径:行号**
  - `frontend/src/app/today-signal/page.tsx:252-256`（`<Alert severity="error" onClose={() => setError(null)}>{error}</Alert>`——**只有关闭按钮，没有「重试」**）
  - `frontend/src/app/signals/page.tsx:657-661`（同样只有 `<Typography color="error">{error}</Typography>`，无重试）
  - 对比：`frontend/src/app/dashboard/page.tsx:202`、`frontend/src/app/backtest/page.tsx:252` **有**「重试」按钮（说明团队已意识到这个问题，只是没推广）
- **现象描述**
  用户遇到「请求失败 / 请求过于频繁 / Network Error」时：
  - 页面顶部虽然有一个「刷新」按钮，但它是**整页重新拉数据**的语义（`signals/page.tsx:545` / `today-signal/page.tsx:247`），在弱网下会再次失败，用户反复点击形成「刷新风暴」→ 更容易触发后端限流；
  - 错误 Alert 自身没有就近的「重试」按钮，用户得往上滚到顶栏找「刷新」；
  - **限流场景特别糟**：后端 `backend/app/middleware/rate_limiting.py:22` 默认 `requests_per_minute=60 / requests_per_hour=1000 / burst_size=10`，前端 429 时的 `api.ts:274-276` 只 `apiLogger.error('请求过于频繁，请稍后再试')`——**既不重试也不提示倒计时**，用户看到「请求过于频繁，请稍后再试」却不知道要等多久。
- **复现步骤**
  1. 视口 390×844，打开 `http://localhost:3000/today-signal`
  2. 断网（DevTools → Offline）后点「刷新」
  3. 观察：出现红色 Alert「请求失败」，Alert 上**只有 X 关闭按钮**，没有任何「重试」
  4. 对比打开 `http://localhost:3000/dashboard` 断网点「重试」→ 有按钮
- **具体改法**
  1. 抽一个 `<ErrorState error={...} onRetry={...} />` 到 `src/components/common/`，包含：图标 + 错误文案 + 「重试」按钮（≥44px 触摸高度）+ 若是 429 额外显示「请在 N 秒后重试」倒计时。
  2. 在 `today-signal/page.tsx:252`、`signals/page.tsx:657`、`wallet/page.tsx` 的错误分支统一替换。
  3. `api.ts:274` 的 429 分支加上 `Retry-After` 读取，并把 429 归类为「可重试」，在 `apiRequest` 层做**指数退避自动重试**（当前 `_retry` 机制只用于 401）。
  4. 顺带修 `today-signal/page.tsx:220` 的 `setError(e instanceof Error ? e.message : '加载信号失败')`——当前会把后端的具体 `detail` 丢掉（见 [P0-1]）。
- **截图文件名**：`03-today-signal-390.png`、`08-today-signal-360.png`

---

#### [P1-5] 表格/图表无数据替代物：K 线图、净值曲线、热力图全是 canvas，屏幕阅读器与色盲用户拿不到任何数据

- **文件路径:行号**
  - `frontend/src/components/charts/TradingViewChart.tsx:481-496`（`lightweight-charts` → canvas，无 `role`/`aria-label`/替代表格）
  - `frontend/src/components/charts/EquityCurveChart.tsx:144`（`echarts.init(chartRef.current)` → canvas，同样无替代）
  - `frontend/src/components/charts/DrawdownChart.tsx:470`、`frontend/src/components/charts/MonthlyHeatmapChart.tsx:444`（同）
  - `frontend/src/components/charts/InteractiveChartsContainer.tsx`（`grep aria-label|role=|srOnly|<table` → **零命中**）
  - `frontend/src/components/charts/EquityCurveChart.tsx`（`grep aria-label|role=|srOnly|visuallyHidden|<table` → **零命中**）
- **现象描述**
  4 个图表组件里**没有一处** `aria-label` / `role="img"` / `aria-describedby` / 隐藏数据表 / `visuallyHidden` 摘要。实测 dashboard 页 `document.querySelectorAll('canvas').length` = 0（该页当时无图表数据），但 `svg` 有 21~27 个，且 **MUI 图表卡片上没有 `role="figure"` + `aria-label`**。
  后果：
  - **屏幕阅读器**：图表区域被读成「image / group」或直接跳过，用户完全不知道净值涨了多少、回撤多深；
  - **色盲用户**：涨跌用红/绿区分（`today-signal/page.tsx:337` `color={... 'success.main' : 'error.main'}`，`mobileOverrides` 的 `success.main='#2e7d32'` / `error.main='#d32f2f'`），红绿色觉障碍者难以分辨买卖方向，且**没有图标/文字冗余编码**（`MobileSignalCard.tsx:22-59` 有 `TrendingUp/TrendingDown` 图标，good，但表格里的 `TableCell` 只有裸色块）。
  - 同时 `globals.css:140-155` 定义了 `.task-status-running` 等纯颜色状态类（`#06b6d4` / `#10b981` / `#ef4444`），同样只有颜色区分。
- **复现步骤**
  1. 视口 390×844，打开 `http://localhost:3000/backtest` → 打开一个回测详情（净值曲线 / 回撤图 / 月度热力图）
  2. macOS「旁白」或 Chrome DevTools → Elements → Accessibility 面板，选中图表容器
  3. 观察：无可访问名、无 role、非文本节点被忽略
  4. 验证：`document.querySelectorAll('canvas').length` 与 `[...document.styleSheets]` 里检索 `role="img"` → 零命中
- **具体改法**
  1. 每个图表容器加语义包裹：
     ```tsx
     <Box role="img" aria-label={`${title}：${summaryText}`}>
       <Chart ... />
     </Box>
     ```
     `summaryText` 用真实数据动态生成，如「净值曲线：起始 100 万，期末 98.4 万，期间最大回撤 -27.1%」。
  2. 图表下方加「查看数据表」可展开区（`<Table>` + 同样的数据源），既是屏幕阅读器/色盲的兜底，也方便核对具体数值。
  3. 颜色状态补冗余编码：信号 `TableCell` 内在色块外加文字/图标（复用 `MobileSignalCard.tsx:50-59` 的 `TrendingUp/TrendingDown` 逻辑）；`globals.css:140-155` 的状态类补 `::before` 图标。
  4. `today-signal/page.tsx:556-575` 的因子 chips 已用文字+颜色，补上 `aria-label`（含贡献度数值）即可。
- **截图文件名**：`05-backtest-390.png`、`10-backtest-360.png`

---

#### [P1-6] `/admin/payments` 在 390px 与 360px 下均产生**页面级横向滚动条**（docW 419 / 412）

- **文件路径:行号**
  - `frontend/src/app/admin/payments/page.tsx:178-194`（`<Stack direction="row" spacing={2}>` 内两个 `<Card sx={{ minWidth: 180 }}>`，**固定 row 方向、无 `flexWrap`、无响应式 `direction` 切换**）
- **现象描述**
  实测：`document.documentElement.scrollWidth` = **419**（390 视口）/ **412**（360 视口），`clientWidth` = 390 / 360 → 确认存在页面级横向滚动。
  溢出元素链：`div.MuiPaper-root`（Card，right=412, w=180, 文本「待确认金额 ¥0.00」）→ `div.MuiCardContent-root` → `h5`（「¥0.00」，right=396）。
  算术：主区 padding `xs:1.5` = 12px×2 = 24px，剩 366px 可用；两个 `minWidth:180` 的 Card + `spacing:2`（16px）= 376px > 366px，**必然溢出**。第二个 Card 的右边界被推到 412px，超出视口 22px。
  同类问题在 `today-signal/page.tsx:329/344/371/386/397/410`（6 个 `minWidth` 150~190 的 Card）、`paper-signal/page.tsx:157`（`minWidth:150`）也存在，只是那些页恰好被更宽的内容撑开而没暴露。
- **复现步骤**
  1. 视口 390×844，打开 `http://localhost:3000/admin/payments`（需 admin 账号）
  2. 观察：页面底部出现横向滚动条；左右拖动时「待确认金额 ¥0.00」卡片被推到视口外
  3. 360×800 下同样复现（docW 412）
  4. 验证：`document.documentElement.scrollWidth` = 419 vs `clientWidth` = 390
- **具体改法**
  1. `admin/payments/page.tsx:178` 的 Stack 改为
     `direction={{ xs: 'column', sm: 'row' }}` + `sx={{ mb: 2, flexWrap: { xs: 'nowrap', sm: 'wrap' } }}`
  2. 两个 Card 的 `minWidth: 180` 改为 `minWidth: { xs: 0, sm: 180 }`（或 `flex: { xs: '1 1 100%', sm: '0 0 180px' }`）。
  3. 全站排查同类：`today-signal/page.tsx:329-410` 的 6 张 Card 改为 `direction={{ xs:'column', sm:'row' }}` + `flexWrap:'wrap'`，`paper-signal/page.tsx:157` 同理。
  4. 建议加一个**开发期守卫**：`useEffect` 里比对 `document.documentElement.scrollWidth > clientWidth` 并在 dev console 报警，避免回归。
- **截图文件名**：`25-admin-payments-390.png`、`26-admin-payments-360.png`

---

#### [P1-7] 弱网下首屏是「空白 + 一个转圈」，没有骨架屏：只有 `app/loading.tsx` 一个根级 fallback，20+ 个路由全无 segment 级 loading

- **文件路径:行号**
  - `frontend/src/app/loading.tsx:1-27`（**全站唯一**的 loading.tsx；内容是一个纯 CSS spinner + 「加载中...」，`minHeight:60vh`）
  - `frontend/src/components/common/PageSkeleton.tsx:9-22`（**Skeleton 组件写好了但全站零引用** —— `grep -rn "PageSkeleton" src/ | grep -v common/PageSkeleton` → 空）
  - `frontend/src/components/common/LoadingSpinner.tsx:36-55`（`fullScreen` 模式 `position:fixed; inset:0; bgcolor:'background.paper'` —— **全屏纯白遮罩**，把已经渲染好的内容全部盖住）
  - 已实现骨架屏的仅 2 页：`frontend/src/app/dashboard/page.tsx:222-224`、`frontend/src/app/backtest/page.tsx:269-274`
- **现象描述**
  Next.js App Router 的 `loading.tsx` 是**按路由段**生效的。仓库里只有根级 `app/loading.tsx`，意味着：
  - 直接访问 `/signals`、`/wallet`、`/today-signal` 等 URL 时，首屏是 60vh 高的空白 + 一个孤零零的 spinner，用户不知道页面长什么样；
  - 客户端路由跳转时，**根级 loading.tsx 根本不会显示**（Next.js 只在当前 segment 及其祖先有 loading 时才显示），所以站内跳转是「旧页面 → 突然换新页面」，没有任何过渡提示；
  - `LoadingSpinner` 的 `fullScreen` 变体用 `position:fixed` + 纯白 `background.paper` **整屏盖住**已有内容，弱网下用户会以为应用卡死。
  另外 `loading.tsx:23` 把 `@keyframes` 定义在 `<style>` 标签里，每次渲染都重新插入，且是硬编码色值 `#1976d2`（与主题 token 重复定义）。
- **复现步骤**
  1. 视口 390×844，DevTools → Network → Slow 3G
  2. 直接打开 `http://localhost:3000/wallet` → 首屏 3~5s 只显示一个居中 spinner，下方什么都没有
  3. 站内从 dashboard 点进「钱包中心」→ 跳转瞬间没有任何 loading 提示
  4. 验证：`find src/app -name loading.tsx` → **只有一个** `src/app/loading.tsx`
- **具体改法**
  1. 给 5 个核心路由各加 segment 级 `loading.tsx`：`src/app/{dashboard,signals,today-signal,backtest,wallet}/loading.tsx`，内容用已有的 `PageSkeleton`（`src/components/common/PageSkeleton.tsx`），例如钱包页用 3 行 88px 矩形、信号页用 6 行卡片骨架。这样根级 `loading.tsx` 只作为真正的兜底。
  2. 把 `PageSkeleton` 组件从「零引用」接入至少上述 5 处。
  3. `LoadingSpinner` 的 `fullScreen` 变体（`LoadingSpinner.tsx:39-47`）改为**半透明遮罩** `bgcolor: 'rgba(255,255,255,0.7)'` + `backdropFilter: 'blur(2px)'`，避免整屏纯白造成的「应用崩溃」错觉。
  4. `loading.tsx:23` 的 `<style>` 内联 keyframes 挪到 `globals.css`。
  5. 配合 [P1-3] 的进度条，长任务不再只有 spinner。
- **截图文件名**：`02-dashboard-390.png`、`06-wallet-390.png`

---

### P2

---

#### [P2-1] 顶部导航两个 IconButton 完全没有可访问名（铃铛、头像），屏幕阅读器读作「按钮」

- **文件路径:行号**
  - `frontend/src/app` 无关；`frontend/src/components/layout/AppLayout.tsx:285-289`（铃铛 `<IconButton color="inherit" onClick={clearNotifications}>`——**无 `aria-label`**）
  - `frontend/src/components/layout/AppLayout.tsx:292-296`（头像 `<IconButton onClick={...} sx={{ p: 0 }}>`——**无 `aria-label`**）
  - `frontend/src/components/layout/AppLayout.tsx:351-353`（抽屉关闭按钮 `<IconButton onClick={...}>`——**无 `aria-label`**，仅靠 X 图标）
- **现象描述**
  审计脚本在**每一个**被测页面上都稳定检出 2 个「无可访问名的可聚焦元素」（`focusableNoName`）：`AppLayout.tsx:285` 和 `:292` 的两个按钮。它们只有 SVG 图标，`innerText` 为空、无 `aria-label`、无 `title`、无 `img[alt]`。
  屏幕阅读器用户会听到「按钮」「按钮」这种无意义提示，完全不知道是通知还是账号菜单——而这两个是全站仅有的两个全局入口。抽屉的关闭按钮同理（该按钮**有**可访问名，因为 MUI 的 Close 图标自带 `aria-label="Close"`，但语义是英文且不明确是「关闭菜单」）。
- **复现步骤**
  1. 视口 390×844，打开任意页（如 `http://localhost:3000/dashboard`）
  2. 验证：`[...document.querySelectorAll('button')].filter(b=>!(b.innerText.trim()||b.getAttribute('aria-label')||b.getAttribute('title')))` → 2 个（铃铛、头像）
  3. 屏幕阅读器（macOS 旁白 / VoiceOver）聚焦顶部右侧 → 播报「按钮」
- **具体改法**
  - `AppLayout.tsx:285` → `<IconButton aria-label="通知" onClick={clearNotifications}>`
  - `AppLayout.tsx:292` → `<IconButton aria-label="账号菜单" onClick={...}>`
  - `AppLayout.tsx:351` → `<IconButton aria-label="关闭导航菜单" onClick={...}>`
  - 铃铛的未读数 `<Badge badgeContent={...}>`（`:286`）建议补 `aria-label` 或在按钮上加 `aria-describedby` 指向计数文本，让 SR 播报「通知，3 条未读」。
- **截图文件名**：`15-drawer-390.png`、`02-dashboard-390.png`

---

#### [P2-2] 表格里可点击的行没有键盘可达性（`signals` 桌面表格行的 `onClick`）

- **文件路径:行号**
  - `frontend/src/app/signals/page.tsx:792-798`（`<TableRow sx={{ cursor:'pointer' }} onClick={() => openHistory(row.stock_code)}>`——**无 `tabIndex`、无 `role="button"`、无 `onKeyDown`**）
  - `frontend/src/app/signals/page.tsx:874-906`（历史明细 Dialog 里的表格）
- **现象描述**
  桌面端（`md` 以上）表格行靠 `onClick` 打开个股信号历史弹窗，但 `<tr>` 不可聚焦，键盘用户**完全无法打开这个功能**——这是核心的「查看多策略近 N 日信号事件历史与价格走势」入口。`cursor:pointer` 是唯一的视觉暗示。
  注意：移动端走的是 `MobileSignalCard`（`signals/page.tsx:735`），不受影响；受影响的是 ≥900px 的桌面/平板用户，以及外接键盘的移动用户。
- **复现步骤**
  1. 视口 1280×800（或 390px 模拟器但窗口宽），打开 `http://localhost:3000/signals`
  2. 生成信号后，用 Tab 键遍历：焦点**不会**落在任何表格行上
  3. 验证：`document.querySelectorAll('tr[onclick], tr[tabindex]')` → 0 个可聚焦
- **具体改法**
  1. 给可点击行加键盘支持：
     ```tsx
     <TableRow
       tabIndex={0}
       role="button"
       aria-label={`查看 ${row.stock_code} 的信号历史`}
       onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openHistory(row.stock_code); } }}
       onClick={() => openHistory(row.stock_code)}
       sx={{ cursor: 'pointer', '&:focus-visible': { outline: '2px solid', outlineColor: 'primary.main' } }}
     >
     ```
  2. 更好的做法：行内放一个真正的 `<Button>`（而不是让整行可点），天然具备焦点与可访问名。
  3. 全站排查同类 `onClick` 在非交互元素上的用法：`grep -rn "onClick" frontend/src/app/ | grep -i "tr\|TableRow\|Box\|Typography"`。
- **截图文件名**：`18-signals-form-390.png`

---

#### [P2-3] 表单 label 与控件的关联只做了 MUI 的一半，antd 的 `Form.Item` label 走的是另一套

- **文件路径:行号**
  - `frontend/src/app/signals/page.tsx:458-510`（4 个 `<FormControl>` + `<InputLabel id="...">` + `<Select labelId="...">` —— 关联**正确**：`aria-labelledby` 实测有值）
  - `frontend/src/app/today-signal/page.tsx:291-310`（`<Select size="small" value={active}>` **完全没有 `InputLabel`、没有 `label`、没有 `aria-label`** —— 旁边的「策略」只是普通 `<Typography>`）
  - `frontend/src/app/wallet/page.tsx:378`（antd `<Form.Item>` 内的 `Input`）
- **现象描述**
  实测审计结果不一致：
  - `signals` 页 4 个 Select 都有 `aria-labelledby="strategy-type-select-label"` 等，**合规**；
  - `today-signal` 的策略下拉（`:291`）**没有任何可访问名**——SR 只会读出当前值「22f」之类，用户不知道这是策略选择器；
  - `today-signal` 的「策略」文字（`:288-290`）是纯展示文本，不是 `<label>`；
  - `signals` 页审计还检出 4 个 `MuiSelect-nativeInput` 报「未标注」——这是 MUI 的**隐藏原生 input**（`<input type="hidden">` 形态），实际由外层 `role="combobox"` 承担语义，属误报，但说明 `aria-hidden` 没有加在原生 input 上，最好补 `aria-hidden="true"` + `tabIndex={-1}`。
- **复现步骤**
  1. 视口 390×844，打开 `http://localhost:3000/today-signal`
  2. 验证：`[...document.querySelectorAll('[role=combobox]')].map(c=>({lb:c.getAttribute('aria-labelledby'), al:c.getAttribute('aria-label')}))` → 唯一那个是 `null/null`
  3. 屏幕阅读器聚焦该下拉 → 播报「22f，折叠按钮」
- **具体改法**
  - `today-signal/page.tsx:291` 的 Select 补 `label="策略"` + `<InputLabel id="strategy-select-label">策略</InputLabel>` + `labelId="strategy-select-label"`，并删掉 `:288-290` 的裸 Typography。
  - 全站 Select 统一走 `labelId` + `InputLabel id` 模式（照抄 `signals/page.tsx:458` 的写法）。
  - MUI 的隐藏 native input 加 `inputProps={{ 'aria-hidden': true, tabIndex: -1 }}`。
- **截图文件名**：`03-today-signal-390.png`、`18-signals-form-390.png`

---

#### [P2-4] 「今天信号」在移动端也没有卡片视图，8 列表格在 `md` 以下直接挤压

- **文件路径:行号**
  - `frontend/src/app/today-signal/page.tsx:512-586`（买入明细 `<Table size="small">`，**8 列**：代码/名称/参考价/股数/金额/费用/分数/为什么买它）
  - `frontend/src/app/today-signal/page.tsx:591-629`（卖出明细表，7 列）
  - `frontend/src/app/today-signal/page.tsx:634-663`（持仓表）
  - 全文件 `grep "components/mobile"` → **0 命中**（`today-signal` 是核心移动端页面，却没有用任何 `Mobile*Card`）
- **现象描述**
  `signals` 页做对了双轨（`signals/page.tsx:703` 移动卡片 + `:752` 桌面表格），但 `today-signal`、`paper-signal`、`paper-portfolio` 三页**都只有裸表格**。
  `today-signal` 的表最宽：`为什么买它` 列是 chips 列表（`:555` `sx={{ maxWidth: 320 }}`）+ 8 列。叠加 [P0-2] 的 Card `overflow:hidden`，在 390px 下这张表几乎必然被裁。
  （注：当前该页因 [P0-1] 完全不渲染，故此问题被掩盖；修复 P0-1 后会立刻暴露。）
- **复现步骤**
  1. 修复 [P0-1] 后，视口 390×844 打开 `http://localhost:3000/today-signal`
  2. 观察买入明细表：8 列挤在 358px 里，「为什么买它」的 chips 被裁掉
  3. 代码级验证：`grep -n "components/mobile" src/app/today-signal/page.tsx` → 0 命中
- **具体改法**
  1. 新建 `frontend/src/components/mobile/MobileTradeCard.tsx`：标题行（代码 + 名称 + 买入/卖出 Chip），指标区用 2 列 key-value 网格（参考价/股数/金额/费用/分数），因子 chips 放底部并带贡献度。
  2. `today-signal/page.tsx:513/591/634` 三处各包一层 `<Box sx={{ display:{xs:'block',md:'none'} }}>` 放卡片，原 `<Table>` 包 `<Box sx={{ display:{xs:'none',md:'block'}, overflowX:'auto' }}>`。
  3. `paper-signal`、`paper-portfolio` 同理（见 [P0-2]）。
- **截图文件名**：`03-today-signal-390.png`

---

#### [P2-5] 移动端表格列宽分配失衡：「名称」列 30px，汉字逐字折行（表头 4 行、数据行 2 行）

- **文件路径:行号**
  - `frontend/src/app/paper-portfolio/page.tsx:220-257`（7 列 `<Table size="small">`）
  - `frontend/src/app/paper-signal/page.tsx:64-95`
  - `frontend/src/theme/mobileOverrides.ts:121-130`（`MuiTableCell` 移动端 `padding: '12px 8px'` + `fontSize: '0.875rem'` —— 字号下调但**没有对列做最小宽度约束**）
- **现象描述**
  实测（390px，paper-portfolio）：表头各列宽度 = 代码 92px / **名称 30px** / 价格 54px / 股数 60px / 金额 60px / 费用 54px / 分数 30px。
  - 「名称」列总宽 30px，扣掉 `padding: 8px×2` 后内容区仅 **14px** → 每个汉字占 14px，**3 个汉字的「萃华退」只能放 2 个字一行、折成 2 行**；
  - 表头「名称」两字被压成 **4 行**，表头行高 **73px**（正常应约 40px）；
  - 数据行行高 **85px**（正常应约 45px）——**每行只有 1 条记录却占了近 2 倍高度**，30 条持仓要滚 2500px 以上；
  - 同时 `fontSize: 0.875rem`（14px）在手机上偏小，配合逐字折行后极难辨认。
- **复现步骤**
  1. 视口 390×844，打开 `http://localhost:3000/paper-portfolio`
  2. 滚动到「卖出 (6)」表格，观察「名称」列：字一个一个往下掉
  3. 验证：`[...t.querySelectorAll('thead th')].map(th=>Math.round(th.getBoundingClientRect().width))` → `[92,30,54,60,60,54,30]`
- **具体改法**
  1. 给关键列加最小宽度并禁止折行：
     - 代码列：`sx={{ fontFamily:'monospace', whiteSpace:'nowrap', minWidth: 84 }}`
     - 名称列：`sx={{ minWidth: 72, whiteSpace:'nowrap' }}`
     - 数值列：`sx={{ whiteSpace:'nowrap' }}`
  2. 表格外层加 `overflowX:'auto'` + `sx={{ minWidth: 560 }}`（配合 [P0-2] 的兜底方案），宁可横向滚动也不要压成 30px。
  3. `mobileOverrides.ts:125` 的 `fontSize: '0.875rem'` 建议回调到 `'0.9rem'`（14.4px），或保持 14px 但在有横向滚动的表格里不缩小。
  4. 更优：走 [P0-2]/[P2-4] 的卡片方案，从根上不需要 7 列。
- **截图文件名**：`12-paper-portfolio-table-390.png`、`19-paper-portfolio-table-crop-390.png`

---

#### [P2-6] 断点体系只有 `xs/sm/md`，`md`(900px) 直接跳到宽屏，缺 `lg/xl` 的宽屏布局

- **文件路径:行号**
  - 全站统计：`grep -rhno "xs={[0-9]*}\|sm={[0-9]*}\|md={[0-9]*}\|lg={[0-9]*}\|xl={[0-9]*}" src/app/ | sed 's/.*://' | sort | uniq -c` → `xs={24}`×8、`sm={8}`×6、`md={14}`×1、`md={10}`×1，**`lg`/`xl` 各 0**
  - `frontend/src/components/layout/AppLayout.tsx:227`（汉堡菜单 `display:{xs:'block',lg:'none'}`）、`:248`（桌面菜单 `display:{xs:'none',lg:'flex'}`）—— **这是全站仅有的两处 `lg`**，只用来切换「抽屉 vs 顶栏」
  - `frontend/src/theme/mobileOverrides.ts:12/16/21/26/32/46/89/98/125/134/144/152/162/173/185/197`（全部 `down('sm')` 或 `down('md')`）
  - `frontend/src/app/dashboard/page.tsx:215`（`gridTemplateColumns: { xs:'1fr', sm:'repeat(2,1fr)', lg:'repeat(4,1fr)' }`）—— 少数用到 `lg` 的地方
- **现象描述**
  1. **900~1200px 区间是「平板死区」**：`md` 以上桌面表格/顶栏都显示了，但 `sm`(600) 以上的 `sm={8}` 三列栅格到 `md` 就变成两列，中间的 900~1200px 没有专门布局；
  2. **≥1200px 没有 `xl` 层级**：主内容 `maxWidth:'1400px'`（`AppLayout.tsx:396`）居中，两侧留白——1920px 视口下主区只有 1400px，**四周各 260px 空白**，信息密度低；
  3. 实测 1920px 下 dashboard 卡片宽度仅 **326px**（4 列），大屏上偏小，右侧大片空档；
  4. `mobileOverrides.ts` 里所有移动端规则都用 `down('sm')`/`down('md')`，**在 900~1200px 的中等屏幕上依然生效**——例如 `MuiTableCell` 的 `padding: '12px 8px'` + `fontSize 14px` 在 1000px 宽的桌面上也生效，把桌面表格压得又窄又小。
- **复现步骤**
  1. 视口 1920×1080 打开 `http://localhost:3000/dashboard`
  2. 观察：主区被限制在 1400px 居中，两侧大片空白；统计卡片 326px 宽
  3. 视口切到 1000px → 表格单元格仍是 14px 字号 + 8px 内边距
  4. 验证：`grep -rn "lg={\|xl={" src/app/` → 0 命中
- **具体改法**
  1. `mobileOverrides.ts` 全面改用 `down('md')`（900px）作为「移动」阈值，或更好——用 `theme.breakpoints.down('sm')` 只保留真正移动端专属规则（圆角/字号），把 `MuiTableCell`（`:121-130`）、`MuiTextField`（`:109-119`）等跟排版相关的规则从 `down('sm')` 调到 `down('md')`。
  2. 在 `AppLayout.tsx:396` 的 `maxWidth` 改响应式：`maxWidth: { xs: '100%', md: '1400px', xl: '1680px' }`，并配合 `dashboard/page.tsx:215` 的 `lg` 补 `xl: 'repeat(5, 1fr)'` 或加大卡片间距。
  3. 全站补 `lg`/`xl` 断点，重点是表格页（`signals`/`backtest`/`models`/`monitoring`）在宽屏下用双列或主从布局，而不是把桌面表格单列拉到 1400px。
- **截图文件名**：`20-dashboard-1920.png`

---

#### [P2-7] `signals` 页表单在 390px 下 8 个控件纵向堆叠，顶部要滚很久才能看到结果

- **文件路径:行号**
  - `frontend/src/app/signals/page.tsx:457`（`<Stack direction={{ xs:'column', md:'row' }} spacing={2}>`）
  - `frontend/src/app/signals/page.tsx:618-621`（筛选区同样的 `xs:'column'`）
  - `frontend/src/app/signals/page.tsx:512-531`（`最近N个交易日` `width:160`、`分页大小` `width:140` —— 固定宽度，在 360px 下 `width:180` 的日期框会溢出容器）
- **现象描述**
  实测 390px 下 `signals` 页可见的表单控件有 8 个（MUI FormControl/TextField），加上策略配置表单和筛选区，**结果表格在首屏之外**。移动端用户每次改一个筛选条件（策略/天数/分页/信号类型/起止日期）都要滚过整个表单才能看到结果。
  实测 `最近N个交易日` 宽 160px、`分页大小` 宽 140px、两个日期框各 180px —— 在 360px 视口下 180px 的日期框加 padding 会贴近容器右边界，虽未溢出但很挤。
  加上 [P1-3] 的「生成信号」是全市场长任务，这个页面的移动端体验是最差的一个。
- **复现步骤**
  1. 视口 390×844，打开 `http://localhost:3000/signals`
  2. 观察：页面从上到下依次是 标题/说明 → 策略类型/策略/股票池来源/天数/分页/全量生成/生成信号 → 策略参数表单 → 信号筛选/起止日期 → 结果区
  3. 滚动到底才能看到「最新信号（第 1 页…）」，首次使用不知道要滚多远
- **具体改法**
  1. 移动端把筛选区收进**可折叠的「筛选」面板**（`Accordion` / `Collapse`），默认收起，只显示已选条件的 summary chip。
  2. `最近N个交易日` / `分页大小` 这类高级参数放进「高级设置」折叠区，移动端默认隐藏。
  3. 生成后**自动滚动到结果区**（`resultsRef.current?.scrollIntoView({behavior:'smooth'})`）。
  4. 固定宽度的 `sx={{ width: 160 }}` 改为 `width: { xs: '100%', sm: 160 }`。
- **截图文件名**：`04-signals-390.png`、`09-signals-360.png`、`18-signals-form-390.png`

---

#### [P2-8] 错误提示只有纯文案，没有 `role="alert"` / `aria-live`，屏幕阅读器不播报

- **文件路径:行号**
  - `frontend/src/app/today-signal/page.tsx:252-256`（`<Alert severity="error" onClose={...}>`）
  - `frontend/src/app/signals/page.tsx:657-661`（`<Typography color="error">{error}</Typography>`—— **纯文本，连 Alert 都没有**）
  - `frontend/src/app/dashboard/page.tsx:198-205`、`frontend/src/app/backtest/page.tsx:248-255`（「加载失败」+「重试」）
  - `frontend/src/components/common/ErrorBoundary.tsx`、`frontend/src/app/error.tsx:28-30`
- **现象描述**
  - `signals/page.tsx:658` 的错误是裸 `<Typography color="error">`，**没有 `role="alert"`、没有 `aria-live`**。屏幕阅读器用户操作后不会收到任何通知——页面只是「看起来没变」。
  - MUI `Alert` 内部有 `role="alert"`（`MuiAlert-root` 默认），所以 `today-signal` / `dashboard` / `backtest` 尚可，但 `signals` / `wallet` / `admin/payments` 的纯文本错误没有。
  - `app/error.tsx:28` 的 `<h1>出错了</h1>` **没有 `role="alert"`**，错误边界触发时 SR 用户不会被立即打断通知。
  - 加载中的状态（`LoadingSpinner`）也没有 `aria-live` / `role="status"`，用户不知道「正在加载」还是「加载完了但没数据」。
- **复现步骤**
  1. 视口 390×844，打开 `http://localhost:3000/signals`，DevTools → Offline，点「生成信号」
  2. 验证：`document.querySelector('[role=alert]')` → `null`（错误以纯文本出现）
  3. 屏幕阅读器不会播报错误
- **具体改法**
  1. `signals/page.tsx:658` 改成 `<Alert severity="error" role="alert" onClose={...}>{error}</Alert>`（与 `today-signal` 一致）。
  2. `LoadingSpinner`（`src/components/common/LoadingSpinner.tsx:57-66`）外层加 `role="status" aria-live="polite"`，让 SR 播报「加载中」。
  3. `app/error.tsx:16` 的根 div 加 `role="alert"`，`<h1>` 前加 `tabIndex={-1}` 并在 `useEffect` 里 `ref.current?.focus()`（错误边界的标准做法）。
  4. 成功/失败的 toast 用 `GlobalSnackbar.tsx`（`src/components/common/`）统一加 `aria-live`。
- **截图文件名**：`04-signals-390.png`、`03-today-signal-390.png`

---

#### [P2-9] 全站没有底部导航/底部操作条，15 个功能入口全靠汉堡抽屉，移动端每次换页要点两次

- **文件路径:行号**
  - `frontend/src/components/layout/AppLayout.tsx:63-145`（`menuItems` 共 **16 项**）
  - `frontend/src/components/layout/AppLayout.tsx:335-388`（`<Drawer>` 抽屉，15~16 项 `ListItemButton`）
  - `frontend/src/components/layout/AppLayout.tsx:214`（`<Box sx={{ display:'flex', flexDirection:'column', minHeight:'100vh' }}>` —— **没有底部固定栏**）
  - `frontend/src/components/layout/AppLayout.tsx:404-420`（footer 是普通流式元素，非固定）
- **现象描述**
  好处是**没有底部栏遮挡内容**（这一点是达标的）。但代价是：
  - 移动端要在 15 个入口间切换，必须：点汉堡（44px）→ 在抽屉里找目标（抽屉实测宽 306px，15 项垂直排列，**总高 818px > 视口 800px，需要在抽屉内二次滚动**）→ 点选。**每次换页 2~3 次点击**；
  - 抽屉实测在 360×800 下 `scrollHeight=818 > clientHeight=800`，即**最后一项「充值订单」默认被裁掉**，需要滚动才能看到；
  - 抽屉内的 `ListItemButton` 高度 47px（≥44px，达标），但**没有搜索/过滤**，16 个入口在抽屉里只能靠滚动找；
  - 抽屉里的当前项高亮用 `bgcolor:'primary.main'` + `color:'primary.contrastText'`（`:369-374`），选中/未选中的区分**只靠颜色**（虽然有 `selected` 属性的语义，但视觉上没有非颜色冗余）。
- **复现步骤**
  1. 视口 360×800，打开 `http://localhost:3000/dashboard`
  2. 点左上汉堡 → 抽屉展开，测量 `document.querySelector('.MuiDrawer-paper').scrollHeight` = **818** vs `clientHeight` = **800** → 可滚动
  3. 观察：底部「充值订单」被裁一半
  4. 验证：抽屉内 15 个 `ListItemButton`，高度均 47px
- **具体改法**
  1. 加**底部 Tab 栏**（4~5 个高频入口：仪表板 / 今日信号 / 策略信号 / 任务 / 更多），放在 `AppLayout.tsx` 主区之后，`position:fixed; bottom:0`，配合 [P0-3] 的 `paddingBottom: 'calc(64px + var(--safe-bottom))'` 避免遮挡内容；主内容区加对应的 `pb`。
  2. 若不加底部栏，至少给抽屉加**搜索框**（`TextField` + `InputAdornment search`）过滤 16 个入口，并把 `ListItemIcon` 的 `minWidth: 40`（`:378`）改小以省出文字空间。
  3. 抽屉项的选中态补非颜色冗余：左侧加 2px 指示条或 ✓ 图标。
  4. 抽屉容器加 `aria-label="主导航"`，`List` 加 `role="navigation"`。
- **截图文件名**：`21-drawer-360.png`、`15-drawer-390.png`

---

#### [P2-10] 登录页有 2 个 21px 高的文字链接（低于 44px 触摸标准），且卡片固定 `width: 420` 靠父级 flex 压扁

- **文件路径:行号**
  - `frontend/src/app/login/page.tsx:152-159`（`忘记密码？` / `注册新账号` 两个 `<Link>`，实测高 **21px**、字号 14px）
  - `frontend/src/app/login/page.tsx:110`（`<Card sx={{ width: 420, p: 4, ... }}>` —— 固定 420px，无 `maxWidth`）
  - `frontend/src/app/login/page.tsx:153`/`:156`（`style={{ color: '#1976d2', fontSize: 14 }}` —— 硬编码色值，绕过主题）
- **现象描述**
  - 实测「忘记密码？」与「注册新账号」两个链接的 `getBoundingClientRect().height` = **21px**。iOS HIG 建议 ≥44px。21px 高的文字链接在手机上极易误触/漏触。
  - `width: 420` 的 Card 在 390px 视口下被父级 flex 挤成 **366px**（`left=12, right=378`）——**没有溢���，靠 flex 压缩侥幸没破**，但 `p: 4`（32px 上下左右内边距）在 390px 下占掉 64px，内容区仅 302px，属于「碰巧能用」而非「设计正确」。
  - 链接颜色 `#1976d2` 硬编码，在白底上对比度约 **4.6:1**（14px 文字需 ≥4.5:1，勉强达标），但如果将来做深色模式会失效。
- **复现步骤**
  1. 视口 390×844，打开 `http://localhost:3000/login`
  2. 观察：底部两个链接行高很矮
  3. 验证：`[...document.querySelectorAll('a')].map(a=>Math.round(a.getBoundingClientRect().height))` → `[21, 21]`
- **具体改法**
  1. `login/page.tsx:152-159` 的链接行改为 `sx={{ display:'flex', gap: 2, '& a': { minHeight: 44, display:'inline-flex', alignItems:'center' } }}`，并给每个链接加 `padding: '10px 0'`。
  2. `login/page.tsx:110` 的 Card 改为 `width: '100%', maxWidth: 420, p: { xs: 2.5, sm: 4 }`（移动端减小内边距，让内容区更宽）。
  3. 硬编码色值换成主题 token：`color: 'primary.main'`。
  4. 登录表单两个输入框实测 `font-size: 16px`（`mobileOverrides.ts:112-116` 生效），**已达标**——iOS 不会自动缩放，这点保持住。
- **截图文件名**：`01-login-390.png`

---

### P3

---

#### [P3-1] 抽屉在 360×800 下内容高 818px > 视口 800px，最后一项默认被裁

- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:335-388`（`<Drawer>` 内部 `<Box sx={{ width: 256, p: 2 }}>`，16 项 × 47px + 标题区）
- **现象描述**：实测 `MuiDrawer-paper` 的 `scrollHeight` = **818**、`clientHeight` = **800**，`overflowY: auto`（可滚，但最后一项「充值订单」初始被裁掉一半）。同时 `mobileOverrides.ts:194-203` 的 `MuiDrawer` 覆盖只在 `down('sm')` 生效，宽度取 `85%` / `maxWidth: 320`（360 视口下实测 306px）。
- **复现步骤**：视口 360×800 → 打开任意页 → 点汉堡 → 观察抽屉底部「充值订单」被裁 → 需要在抽屉内滚动。
- **具体改法**：抽屉 List 容器加 `overflowY:'auto'` + `flex:1` + `minHeight:0`（当前是整页一起滚），使列表区域独立滚动、标题固定；或压缩 `ListItemButton` 的 `minHeight` 到 44px 并减小 `List` 间距。
- **截图文件名**：`21-drawer-360.png`

---

#### [P3-2] 多个页面用「请求失败 / Network Error」等英文或技术性文案直接呈现给用户

- **文件路径:行号**
  - `frontend/src/app/dashboard/page.tsx:202`（「重试」按钮上方显示后端原文）
  - `frontend/src/services/api.ts:283-284`（网络错误时 `apiLogger.error('网络连接失败，请检查网络设置')`）
  - `frontend/src/app/today-signal/page.tsx:220`（`setError(e.message || '加载信号失败')` —— 把后端 message 直接透出）
- **现象描述**：实测断网时 dashboard 显示 **「Network Error」**（axios 原始英文错误），today-signal 显示 **「请求失败」**（`api.ts:197` 的通用兜底，丢掉了后端的中文 detail）。普通用户看不懂 "Network Error"；而 `请求失败` 又过于笼统，无法区分「网络断了」「服务器 500」「被限流」「参数错」。
- **复现步骤**：视口 390×844 → dashboard → DevTools Offline → 点「重试」→ 观察红色 Alert 显示「Network Error」。
- **具体改法**：`api.ts` 增加 `const FRIENDLY: Record<number,string> = { 400:'请求参数有误', 401:'登录已过期，请重新登录', 403:'当前订阅等级无权访问该功能', 404:'资源不存在', 429:'操作太频繁，请稍后再试', 500:'服务器开小差了，请稍后重试' }`，网络错误统一映射为「网络连接失败，请检查网络设置后重试」。`today-signal/page.tsx:220` 等处优先用 `e.message`。
- **截图文件名**：`02-dashboard-390.png`

---

#### [P3-3] `TaskDetailContent.tsx` 等处内联重复实现了一份 `EmptyState`，与 `src/components/common/EmptyState.tsx` 重复

- **文件路径:行号**：`frontend/src/app/tasks/[id]/TaskDetailContent.tsx:852`（内联 `function EmptyState({...})`）vs `frontend/src/components/common/EmptyState.tsx:12`
- **现象描述**：`TaskDetailContent.tsx` 里内联了一份 `EmptyState` 组件（852 行起），而 `common/EmptyState.tsx` 另有一份官方实现。重复实现容易导致样式漂移（`common/EmptyState.tsx:21` 用 `variant="h3"` 显示 emoji 图标，在 390px 下是 24px 大字 emoji，`mobileOverrides.ts:11-15` 只缩了 h4/h5/h6，**h3 不受影响**）。
- **复现步骤**：代码级：`grep -n "function EmptyState" src/app/tasks/\[id\]/TaskDetailContent.tsx` → 852。
- **具体改法**：删除内联实现，统一 import `common/EmptyState`；同时给 `EmptyState` 的 `h3` 加上移动端字号覆盖（`mobileOverrides.ts` 补 `h3: { fontSize: { xs: '2rem' } }`）。
- **截图文件名**：`17-tasks-390.png`

---

#### [P3-4] `Monitoring` / `Admin` 等页的连接状态 Chip 在移动端被隐藏，用户无法得知 WS 是否在线

- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:276-282`（`<Chip label={wsConnected ? '已连接' : '未连接'} sx={{ display: { xs: 'none', sm: 'flex' } }} />`）
- **现象描述**：390px 下 `display:none`，顶栏只剩汉堡 + 图标 + 铃铛 + 头像。WebSocket 断线（`wsService.on('system:alert')` 的告警计数）移动端**完全不可见**——用户不知道后台推送是否还活着，而 `AppLayout.tsx:167` 每 5 秒轮询一次 `wsService.isConnected()` 却在移动端把结果藏了。
- **复现步骤**：视口 390×844 → 任意页 → 观察顶栏无连接状态指示。
- **具体改法**：移动端用一个小圆点（8px，`success.main` / `error.main`）代替文字 Chip，放进铃铛按钮左侧，并加 `aria-label="连接状态：已连接/未连接"`（顺带解决 SR 不可感知的问题）。
- **截图文件名**：`02-dashboard-390.png`

---

#### [P3-5] 抽屉里 16 个导航项在移动端没有分组/搜索，且「今日信号 / 纸面信号 / 多策略纸面盘」三个 Signal 图标完全相同

- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:109-128`（三个菜单项都用 `icon: Signal`）
- **现象描述**：`/signals`、`/today-signal`、`/paper-signal`、`/paper-portfolio` 四个入口中前三个用**完全相同的 `Signal` 图标**（lucide），用户在抽屉里只能靠文字区分。同时 16 个平铺的项目没有按「交易 / 模型 / 系统」分组，16 行扁平排列。
- **复现步骤**：视口 390×844 → 打开抽屉 → 观察三个 Signal 图标一模一样。
- **具体改法**：给三个入口配不同图标（`TrendingUp` / `CalendarToday` / `ClipboardList`），并用 `Divider` 或 `ListSubheader` 分组（交易类 / 模型类 / 系统类）。
- **截图文件名**：`15-drawer-390.png`、`21-drawer-360.png`

---

#### [P3-6] `globals.css` 的自定义滚动条样式对移动端无效（iOS 会覆盖），但 `overflowX:auto` 表格在 iOS 上又缺少惯性滚动

- **文件路径:行号**：`frontend/src/app/globals.css:25-40`（`::-webkit-scrollbar { width:6px; height:6px }` 等 4 条规则）
- **现象描述**：`::-webkit-scrollbar` 在 iOS Safari 上不生效（iOS 会用系统覆盖式滚动条），桌面 Chrome 上又只有 6px 宽，几乎不可见。对比度也偏低：`hsl(var(--foreground) / 0.2)` 在浅色背景下约 1.4:1，几乎看不见——**用户不知道表格可以横向滚动**，这会放大 [P0-2] 的伤害。
- **复现步骤**：视口 390×844 → 打开 `http://localhost:3000/paper-portfolio` → 尝试横向拖动表格（发现根本拖不动，因为没有 `overflowX:auto`，见 [P0-2]）；在桌面 Chrome 打开 `/data` 页的宽表格 → 滚动条细到几乎看不见。
- **具体改法**：
  1. 滚动条加 `scrollbar-width: thin` + `scrollbar-color:`（Firefox），桌面端把 `width/height` 提到 10px、透明度提到 0.35；
  2. 所有横向滚动容器加 `WebkitOverflowScrolling: 'touch'`（iOS 惯性）+ `scrollbar-gutter: stable`；
  3. 更重要的是**加视觉提示**：滚动容器右侧加渐隐遮罩（`mask-image: linear-gradient(...)`）或底部加「← 左右滑动查看更多 →」文案，让用户知道还有内容。
- **截图文件名**：`12-paper-portfolio-table-390.png`

---

## 二、先修清单（P0/P1，按优先级）

| # | 级别 | 问题 | 位置 | 一句话改法 |
|---|------|------|------|-----------|
| 1 | **P0** | `/trading/*` 响应缺 `{success,data}` 信封 → 今日信号页 100% 不可用 | `backend/app/api/v1/trading.py:75` ↔ `frontend/src/services/api.ts:194,306` | 后端 3 个 handler 套 `StandardResponse`；前端 `apiRequest.get` 兼容裸 payload |
| 2 | **P0** | 7~8 列表被 Card `overflow:hidden` 静默裁掉（分数列永久不可见） | `paper-signal/page.tsx:64`、`paper-portfolio/page.tsx:220`、`today-signal/page.tsx:524,602,640` | 外层加 `overflowX:'auto'` + `minWidth`；同时补移动端卡片视图 |
| 3 | **P0** | 全站零 `safe-area-inset`，刘海屏/灵动岛遮挡顶栏与 footer | `frontend/src/app/globals.css`、`AppLayout.tsx:218,410`、`layout.tsx:13` | 加 `--safe-*` CSS 变量 + `viewportFit:'cover'` |
| 4 | **P1** | 触摸目标：antd 按钮 32px、若干 40/43px，低于 44px | `wallet/page.tsx`（antd）、`today-signal` Chip 30px | antd `ConfigProvider` 设 `controlHeight:44`；MUI `sizeSmall` 提到 44 |
| 5 | **P1** | antd Input `font-size:14px` → iOS 聚焦自动放大、键盘顶飞弹窗 | `wallet/page.tsx:378` | `ConfigProvider` 全局 `fontSize:16`；或该 Input 加 `style={{fontSize:16}}` |
| 6 | **P1** | 全站无可见焦点框（outline 恒 0px），IconButton 焦点完全不可见 | `theme/muiTheme.tsx:89`（无 focus 覆盖） | 加 `MuiButtonBase` `focusVisible` outline 2px + antd 补 `:focus-visible` |
| 7 | **P1** | 顶栏铃铛/头像两个 IconButton 无可访问名 | `AppLayout.tsx:285,292,351` | 补 `aria-label="通知"/"账号菜单"/"关闭导航菜单"` |
| 8 | **P1** | `/admin/payments` 在 390/360 下产生页面级横向滚动（419/412px） | `admin/payments/page.tsx:178-194` | Stack 改 `direction={{xs:'column',sm:'row'}}` + `flexWrap` + `minWidth:{xs:0,sm:180}` |
| 9 | **P1** | 弱网无骨架屏：只有根级 `loading.tsx`，`PageSkeleton` 零引用 | `app/loading.tsx`、`common/PageSkeleton.tsx` | 5 个核心路由加 segment `loading.tsx` + 接入 PageSkeleton；`LoadingSpinner fullScreen` 改半透明 |
| 10 | **P1** | 长任务无进度/无取消：`/signals` 全市场生成是串行 while 循环 | `signals/page.tsx:318-347,704-710` | 累加页数显示百分比 + LinearProgress + 停止按钮 + 单页超时 |
| 11 | **P1** | 错误态无重试入口 + 429 无退避提示 | `today-signal/page.tsx:252`、`signals/page.tsx:657`、`api.ts:274` | 抽 `<ErrorState>` 组件；api 层加 429 指数退避与倒计时 |
| 12 | **P1** | 图表全 canvas，无 aria/数据表替代；涨跌只靠红绿 | `charts/TradingViewChart.tsx:481`、`EquityCurveChart.tsx:144` | 容器加 `role="img"`+动态 `aria-label`；补「查看数据表」折叠区；信号加图标冗余 |

---

## 三、总体评价

**移动端适配「有形无神」**：骨架有了（`src/components/mobile/*` 7 个卡片组件、`MobileSignalCard` 的双轨范式）但只覆盖了 6 个页面，核心的 today-signal / paper-signal / paper-portfolio 三页仍是裸表格；`@media (pointer: coarse)` 的 44px 规则只对 MUI 生效，antd 组件完全绕开；`safe-area` 一个字都没写。全站唯一真正的 P0 不是样式问题而是**接口契约不一致**（`/trading/*` 缺 `success/data` 信封）——这让「今日信号」在手机上直接白屏无内容，比任何布局瑕疵都致命。

**可访问性基本处于「能看不能用」的水平**：没有任何可见焦点框、两个顶栏图标无可访问名、图表是纯 canvas 无数据替代、涨跌只用红绿编码、错误提示不带 `role="alert"`。这不是细节打磨，而是键盘/屏幕阅读器/色盲用户完全不可用的状态。

**最值得肯定的三点**：MUI 主题层有明确的 `mobileOverrides.ts` 集中治理（方向正确）；输入框 `font-size:16px` 的 iOS 缩放问题在 MUI 侧已处理；dashboard/backtest 有骨架屏、dashboard/backtest 有「重试」按钮（说明团队已在往这个方向走，只是没推广到其他页面）。

**修复顺序建议**：先修 #1（接口契约，一行 wrapper 就能让今日信号复活），再修 #2/#3/#8（数据不可见 + 硬件适配，改动集中且收益立竿见影），然后集中做 #4~#7（a11y 与触摸目标，收益高但需要新增 `ConfigProvider` 与全局 focus 规则），最后 #9~#12（弱网与图表可访问性，是持续投入项）。补完 #1~#8 后，移动端可用性大致能从当前的「核心功能不可用」提升到「可正常使用」。

**审计覆盖与局限**：本轮为只读审计，未修改任何源码。实测覆盖 dashboard / today-signal / signals / backtest / wallet / paper-signal / paper-portfolio / tasks / models / data / monitoring / admin-payments / pricing / optimization / login，390×844 与 360×800 两档视口，均开启 `pointer:coarse` 触摸模拟（否则 MUI 的 44px 规则不生效，测量结果会失真）。因后端限流（60 次/分），部分页面的带数据态是通过真实数据页（paper-signal / paper-portfolio）与接口直连验证的；`today-signal` 的表格因 [P0-1] 无法渲染，其 8 列挤压结论来自同构的 paper-signal 实测与源码分析，修复 P0-1 后需复测。视觉服务的 `vision_analyze` 在本次会话中不可用，所有结论均基于 DOM 几何测量（`getBoundingClientRect` / `getComputedStyle`）与源码交叉验证得出，未依赖截图目测。
