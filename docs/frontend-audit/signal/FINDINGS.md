# 前端核心赚钱决策页审计报告（只读）

- 审计对象：`dashboard` / `today-signal` / `signals` / `paper-signal` 四个页面 + `components/charts/*` + `utils/featureConfigs.ts`
- 审计时间：2026-10-27 08:00–09:40 CST
- 环境：前端 `http://localhost:3000`（dev server PID 72823）、后端 `http://127.0.0.1:18082`（健康检查 200）
- 账号：`ronghui@example.com` / `XMPzFEZDFpvV`（`subscription_tier=enterprise`、`is_admin=1`）
- 方式：browser_exec 真实渲染 + 截图（目录 `docs/frontend-audit/signal/`），配合 curl 核对后端真实响应、源码行号定位
- 截图清单：
  - `01-dashboard-desktop.png` — 仪表盘桌面端
  - `02-today-signal-BROKEN.png` — 今日信号页生产环境实况（**完全不可用**）
  - `02-today-signal-WORKING.png` — 今日信号页（用真实后端数据 mock 信封后渲染出的「设计意图」画面）
  - `02b-today-signal-buylist.png` — 今日信号页买入/卖出/持仓三表
  - `03-paper-signal.png` — 纸面信号桌面端（信号日期 2026-10-16，对比多策略纸面盘的 2026-10-26）
  - `04-signals.png` — 策略信号页初始态（策略下拉只有「布林带策略（bollinger）」）
  - `04-signals-generated.png` / `04-signals-filter-BUY.png` — 生成 5594 行后 / 筛选 BUY 后（3633 行）
  - `05-signals-mobile.png` — 策略信号页移动端（文档高 423,040px）
  - `06-today-signal-mobile.png` — 今日信号页移动端（**右 4 列被裁**）
  - `07-dashboard-mobile.png` — 仪表盘移动端
  - `08-paper-signal-mobile.png` — 纸面信号移动端（**右 2 列被裁**）

---

## 一、问题清单

### P0-1 · 今日信号页在生产环境**完全不可用** —— 付费用户打开就是「请求失败」
- **文件**：`frontend/src/app/today-signal/page.tsx:186`（`/trading/strategies`）、`:208`（`/trading/signal`）、`:213`（`/trading/signal-quality`）
- **根因（源码 + 实测双证）**：
  - 后端 `backend/app/api/v1/trading.py:74-82`、`:130`、`:158` 直接 `return {...}` 裸 dict，**没有 `{success, data}` 外层信封**。实测 `curl` 返回 200 且顶层无 `success`/`data` 键。
  - 前端 `frontend/src/services/api.ts:306` 的 `apiRequest.get` 硬性要求 `res.data.success` 为真，否则 `throw new Error(res.data?.message || '请求失败')`；`:310-314` 再要求 `res.data.data` 存在。
  - 对比：`/api/v1/tasks`、`/api/v1/signals/latest`、`/api/v1/backtest/strategies` **都有** `{"success":true,"data":...}` 信封（实测确认），只有 `/trading/*` 三个端点漏了。
- **现象**：付费 enterprise 用户打开「今日信号」，页面顶部红条 **「请求失败」**，策略下拉框**完全空白**（0 个选项），连信号日期都读不到。对比 `/api/v1/tasks` 正常返回 → 证明不是鉴权、不是网络、不是限流。
- **复现步骤**：
  1. 登录 `ronghui`（enterprise 档）
  2. 顶部导航点「今日信号」
  3. 观察：红色错误条「请求失败」+ 策略下拉为空
  4. 佐证：DevTools 里 `/api/v1/trading/strategies` 状态 **200**，但响应体顶层无 `success` 字段
- **影响**：这是付费用户「真正买的东西」（页头原话：「含股数 / 金额 / 成本，可直接照着下单」），**100% 打不开**。所有靠信号变现的付费逻辑全部落空，直接影响平台收入。
- **具体改法**（二选一，推荐 A）：
  - **A（后端统一信封，最小改动）**：在 `backend/app/api/v1/trading.py` 三个端点返回前包一层 `{"success": True, "data": <payload>, "timestamp": ...}`；或引入项目已有的 `StandardResponse` 响应模型（`/api/v1/backtest.py:158` 就是这么写的），把 `response_model=StandardResponse` + 返回 `StandardResponse(data=payload)`。
  - **B（前端容错）**：在 `api.ts` 的 `apiRequest.get` 里加兜底 —— 若响应体无 `success` 键但 HTTP 200，则把整个 body 当作 data 返回：`if (res.data?.success === undefined && res.status === 200) return res.data as T;`。**B 会掩盖后续同类 bug，建议 A 为主、B 为兜底。**
  - 修完必须回归：至少 3 个端点各 1 次真实请求（返回 200 + 顶层有 `success`），并让前端在「今日信号」页能真正渲染出买入表。
- **截图**：`02-today-signal-BROKEN.png`

### P0-2 · 付费用户被误判为「体验版」并被推去付费墙 —— 付费墙逻辑反向伤客
- **文件**：`frontend/src/app/today-signal/page.tsx:260`（tier Chip）、`:269-278`（升级按钮）、`:188-190`（setTier/setTierLabel）
- **现象**：因 P0-1 导致 `/trading/strategies` 抛错，`catch` 分支只 `setError(...)`，**`tier` / `tierLabel` 保持 useState 初始值**（`:172-173` = `'free'` / `'体验版'`）。于是页面上真实档位 `enterprise` 的用户看到的是：
  - 档位标签 **「体验版」**（灰色 Chip）
  - 蓝底按钮 **「升级解锁全部信号线」** —— 付费用户被引导去重复付费
- **复现步骤**：用 `ronghui`（DB 里 `subscription_tier=enterprise`）登录 → 打开 `/today-signal` → 截图可见「体验版」标签 + 「升级解锁全部信号线」按钮。
- **具体改法**：
  1. `useState` 初始值改为 `tier: null`、`tierLabel: null`（**未加载完不渲染档位 Chip 和升级按钮**），加载成功后再赋值；
  2. `catch` 分支里不要伪造「体验版」—— 加载失败时档位应显示「—」或「加载中」，**绝不显示升级入口**（否则任何后端抖动都会把付费用户推向重复付费）；
  3. 升级按钮的显示条件收紧为 `tier === 'free' && tierLabel !== null`。
- **截图**：`02-today-signal-BROKEN.png`

### P0-3 · 「纸面信号」页展示的是**两周前的旧快照**，与「今日信号」页口径打架
- **文件**：`frontend/src/app/paper-signal/page.tsx:99`（`fetch('/paper-signal.json')`）
- **现象**：
  - 纸面信号页：`meta.date = 2026-10-16`、`generated_at = 2026-10-19T22:22`，买 30 / 卖 0 / 持 30，组合总值 ¥998,796。
  - 而多策略纸面盘 `data/paper/portfolio.json` 的 22f 线：`signal_date=2026-10-23`、`trade_date=2026-10-26`、**买 3 / 卖 3**、值 ¥990,864。
  - 两个页面同时挂在导航栏（`AppLayout.tsx:120` 纸面信号 / `:125` 多策略纸面盘），**信号内容互相矛盾**。用户按纸面信号页的 30 笔买入单下单，实际那些票 10 天前就买过了。
  - 根因：`paper-signal.json` 由**旧单策略脚本** `backend/scripts/weekly_signal.py:313` 生成，而生产口径已迁到 `backend/scripts/paper_portfolio.py:1072`（写 `paper-portfolio.json`）。`paper-signal.json` 已在 10-19 后无人刷新。
- **复现步骤**：
  1. 打开 `/paper-signal` → 观察「信号日期 2026-10-16」「生成于 2026-10-19T22:22:07」
  2. 打开 `/paper-portfolio` → 观察 22f 线「信号日期 2026-10-23 / 成交日 2026-10-26」
  3. `ls -la frontend/public/paper-signal.json` → mtime 10-19；`paper-portfolio.json` → mtime 10-27
- **具体改法**：
  1. **立即**：`paper-signal` 页改为读 `paper-portfolio.json`（取 22f 线），或在页头加醒目「数据已过期 N 天」横幅（`generated_at` 距今天 > 2 个交易日即警告）；
  2. 删除或重定向 `/paper-signal` 到 `/paper-portfolio`（旧页信息已被完全覆盖）；
  3. 补一条数据新鲜度守卫：静态 JSON 超过 48h 未更新 → 页面显示「数据已过期，等待平台生成新信号」而不是照常渲染旧买单。
- **截图**：`03-paper-signal.png`

### P1-4 · 「纸面信号」页的持仓/金额 JSON **无鉴权即可下载**（付费内容泄露）
- **文件**：`frontend/public/paper-signal.json`（静态产物）、`frontend/src/app/paper-signal/page.tsx:99`
- **现象**：`GET http://localhost:3000/paper-signal.json` → **200**，未登录、无 token，直接拿到 30 只持仓的代码/名称/股数/价格/金额/成本 + 组合总值。虽然是纸面数据，但它是**付费产品的核心样例**（展示平台策略选股能力），且「多策略纸面盘」的 `paper-portfolio.json` 同理无鉴权。
- **复现步骤**：`curl -s http://localhost:3000/paper-signal.json | head -c 200` → 直接返回 JSON。
- **具体改法**：
  1. 敏感产物（持仓/金额/逐笔）移出 `public/`，改由后端受保护端点（如 `/api/v1/trading/paper-portfolio`）下发，套 `require_current_user`；
  2. 若必须静态，至少把 `paper-portfolio.json` 移到 `public/` 之外的受控路径并加 `Content-Disposition` / 签名短链；
  3. 同步核对 `frontend/public/paper-signal.json` 是否还需保留（与 P0-3 一起处理）。
- **截图**：`03-paper-signal.png`

### P1-5 · 买入表的「分数」和「为什么买它」两列**永远是 `—`** —— 用户要的「为什么买」根本没交付
- **文件**：`frontend/src/app/today-signal/page.tsx:551-553`（分数）、`:555-582`（为什么买它）
- **现象**（mock 信封还原后实测）：买入表 10 行，`分数` 列 10/10 为 `—`，`为什么买它` 列 10/10 为 `—`。页面给了两列却没有任何值。
- **根因**：`backend/scripts/paper_portfolio.py:822-832` 的 `rebuild_line_from_files()` 从 ledger.csv 重建 buys 时，只搬 `code/name/price/shares/amount/fee/commission/stamp_tax/slippage`，**没搬 `score`**，更没有 `factors`（ledger.csv 表头就没有 factors 列，13 列：`date,action,code,name,price,shares,amount,commission,stamp_tax,slippage,score,cash_after,portfolio_value`）。而 `paper_portfolio.py:645/699` 生成的实时 buys/sells **是有 `score` 和 `factors` 的** —— 但被 `rebuild_line_from_files` 覆盖掉了。
  - 佐证：`data/paper/portfolio.json` 的 honest_42f buys 实际键是 `['code','name','price','shares','quantity','amount','fee','commission','stamp_tax','slippage']` —— 无 score、无 factors。
  - 而 ledger.csv 里 **是有 score 的**（如 `001213.SZ,0.916`）。
- **复现步骤**：打开今日信号页 → 买入表 → 「分数」「为什么买它」两列全 `—`。
- **具体改法**：
  1. `rebuild_line_from_files()` 的 item 字典补上 `"score": _tofloat(r.get("score"))`；
  2. `factors` 需要**落盘**：在 `paper_portfolio.py` 写 ledger 时同步写 `data/paper/<strategy>_factors.json`（code → [{factor, contribution, direction}]），或在 `portfolio.json` 的 line 里保留 factors；
  3. 前端在无数据时把 `—` 换成明确文案「口径未提供」而非破折号，避免用户误以为「没有理由」= 「不该买」。
- **截图**：`02b-today-signal-buylist.png`、`02-today-signal-WORKING.png`

### P1-6 · 「最大回撤 / 年化夏普」两张风险卡**永远显示 `—`**，且卡内文案漏出 `undefined`
- **文件**：`frontend/src/app/today-signal/page.tsx:349-384`（回撤卡）、`:376-383`（夏普卡）
- **现象**：mock 信封还原后实测，回撤卡显示 `—` + 副标题 **「纸面盘 undefined 点」**，夏普卡显示 `—` + 「逐笔胜率待积累」。
- **根因**：
  1. `backend/app/api/v1/trading.py:126-129` 的 `/trading/signal` 只塞了 `tier/tier_label/delay_days/upgrade_hint`，**没有** `sharpe_ratio` / `max_drawdown_pct` / `nav_points` / `nav_basis` / `nav_insufficient` / `trade_win_rate_pct`。实测 `/trading/signal` 返回体里这 7 个字段**全部 absent**。
  2. 这些字段是在 `signal_service.performance()` 里算的（commit 3e1eb771 新增），**`/trading/performance` 端点有**（实测 `sharpe_ratio:null, max_drawdown_pct:null, nav_points:3, nav_insufficient:true`），但**前端 today-signal 页压根没调 `/trading/performance`** —— 只调了 `/trading/signal` + `/trading/signal-quality`。
  3. 于是 `:364` 的 `` `${signal.nav_basis || '纸面盘'} ${signal.nav_points} 点` `` → `nav_basis` 缺失走 fallback「纸面盘」，`nav_points` 是 `undefined` → 直接渲染出字面量 **「undefined 点」**。
- **复现步骤**：`/today-signal` → 「最大回撤」卡 → 副标题出现「undefined 点」。
- **具体改法**：
  1. 后端 `/trading/signal` 补上绩效字段（直接复用 `performance()` 的计算）；
  2. 前端 today-signal 页**并行调用** `/trading/performance?strategy=X` 并合并进 `signal`；
  3. `:364` 改为 `${signal.nav_basis || '纸面盘'}${signal.nav_points != null ? ` ${signal.nav_points} 点` : ''}`，消除 `undefined` 字面量；
  4. 样本不足时（`nav_insufficient=true`，实测当前 `nav_points=3 < 20`）应显示 **「样本不足（3/20 期），暂无法计算」** 而不是 `—`，避免用户误读为「回撤为 0 / 无风险」。
- **截图**：`02-today-signal-WORKING.png`

### P1-7 · 「策略信号」页一次性渲染 5594 行 → 文档高度 **186,030px**，移动端 423,040px
- **文件**：`frontend/src/app/signals/page.tsx:318-347`（`generateAll` 循环拉全量）、`:792-822`（`filteredRows.map` 全量渲染）
- **现象**：`generateAll` 默认 `true`（`:72`），点「生成信号」后前端循环翻页把**全市场 5594 只**全部拉进 `rows` state，然后**一次性**渲染成 5594 个 `<TableRow>`。实测：
  - 桌面 `document.documentElement.scrollHeight = 186,030`
  - 移动端 `scrollHeight = 423,040`
  - 表头 `最新信号（第 1 页，5594 / 5594 / 5594，…）` —— 「第 1 页」和「5594」并列，页码语义已失效
  - 切到 BUY 筛选后仍渲染 3633 行
- **复现步骤**：打开 `/signals` → 点「生成信号」→ 等 60s → 看表格行数与页面高度。
- **具体改法**：
  1. 关掉 `generateAll` 默认值（改 `false`），或改成「只渲染前 200 行 + 虚拟滚动」；
  2. 引入虚拟列表（`@tanstack/react-virtual` 或 MUI `List` 虚拟化），或服务端分页 + 真实页码控件（现在「上一页/下一页」按钮被 `disabled={generateAll || ...}` 直接锁死，等于没有分页）；
  3. 顶部计数文案改成「共 5594 只，当前显示 1–200（按信号日期倒序）」，并补一个**按信号日期倒序**的默认排序（当前按股票代码升序，最新的信号被埋在中间）。
- **截图**：`04-signals-generated.png`、`04-signals-filter-BUY.png`

### P1-8 · 移动端表格右列被**硬裁**（无横向滚动），「金额/费用/分数/为什么买它」在手机上不可见
- **文件**：`frontend/src/app/today-signal/page.tsx:524`、`:602`、`:640`（三张 `<Table>` 均无 `overflowX` 包裹）；`frontend/src/app/paper-signal/page.tsx:64`
- **现象**（390px 视口实测）：
  - 今日信号买入表：table 宽 436px，外层 CardContent 仅 318px，MuiPaper 的 `overflow: hidden` 把超出部分**直接裁掉**。列右边界 vs 裁切线 354px：`金额`(374) `费用`(427) `分数`(458) `为什么买它`(488) **四列全部 clipped**。
  - 纸面信号买入表：table 宽 432px vs 容器 316px，`成本`(415) `分数`(469) **两列被裁**，且 `overflowX` 是 `visible`（既不可见也不可滚）。
  - 对比：`signals/page.tsx:752` 和 `:874` **有** `overflowX: 'auto'`，说明同项目内已有正确做法，这三处漏了。
- **复现步骤**：Chrome DevTools 切 iPhone 尺寸（390×844）→ 打开 `/today-signal` → 滚到买入表 → 右侧列被截断，且页面无横向滚动条。
- **具体改法**：
  1. 三张表统一包一层 `<Box sx={{ overflowX: 'auto' }}>`（照抄 `signals/page.tsx:752` 的写法）；
  2. 更推荐移动端改卡片式展示（复用现成的 `components/mobile/MobileSignalCard.tsx`），把「代码/名称/价格/股数/金额/费用」以 label:value 形式堆叠；
  3. 窄屏下把「为什么买它」这一列整列隐藏（理由用 tooltip / 展开行），避免 8 列硬挤。
- **截图**：`06-today-signal-mobile.png`、`08-paper-signal-mobile.png`

### P1-9 · 「策略信号」页把**3 个月前的陈旧信号**和今天的信号混在一张表里，无任何时效标识
- **文件**：`frontend/src/app/signals/page.tsx:805-818`（单元格渲染）、`:706-709`（`noWrap` + `title`）
- **现象**：5594 行的信号日期实测分布 `2026-06-09 ~ 2026-10-27`（68 个不同日期）：
  - **1359 行**信号日期距今 > 30 天
  - **203 行** > 60 天
  - 最老的一行 **140 天前**（`HOLD @ 2026-06-09`）
  - 中位年龄 11 天，p90 = 46 天
  - 只有 **309 行**是近 1 日内的
  - 表格里这些行**长得完全一样**：同样的黑字、同样的 `BUY @ 2026-08-20` 格式，唯一的区分是 `title` 悬浮提示（移动端根本看不到 tooltip）
  - 表头写「最新信号」，但语义是「窗口内最后一次事件」—— 用户极易把 3 个月前的 `BUY` 当成今天的建议
- **复现步骤**：`/signals` → 生成信号 → 用日期筛选 `开始日期=2026-10-20` → 对比总行数与「当前筛选条件下」行数；或直接看表头下第一列 `000021.SZ BUY @ 2026-10-16`、`000166.SZ BUY @ 2026-08-20`。
- **具体改法**：
  1. 表头文案从「最新信号」改为「窗口内最后一次信号（默认 60 交易日）」，加副标题说明这是**回看**不是**今日建议**；
  2. 给超过 N 个交易日（如 5 天）的行加视觉降级（灰字 + 「⚠ 陈旧」Chip），并提供「只看近 5 日」快捷筛选；
  3. 移动端 `MobileSignalCard`（`components/mobile/MobileSignalCard.tsx:130-148`）目前只显示月/日不显示年，且**丢了「原因」** —— 移动端用户完全无从判断信号新鲜度。
- **截图**：`04-signals-generated.png`、`04-signals-filter-BUY.png`

### P2-10 · 「策略信号」页价格列**不做小数位格式化**，浮点尾数直接暴露
- **文件**：`frontend/src/app/signals/page.tsx:806-808`（`${val.price}` 裸插值）
- **现象**：5594 行价格的小数位分布实测 `0 位: 108 行 / 1 位: 679 行 / 2 位: 4805 行 / **15 位: 2 行**`。那 2 行直接渲染成 `14.799999999999999`、`14.169999999999998`（原始 JS 浮点尾数），界面上肉眼可见。
- **复现步骤**：`/signals` 生成信号 → 搜索 `14.799999999999999`。
- **具体改法**：`:807` 改为 `val.price != null ? val.price.toFixed(2) : ''`，并在 `MobileSignalCard.tsx:114` 保持同样的 `toFixed(2)`（该处已正确，仅桌面端漏了）。

### P2-11 · 「策略信号」页 **BUY / SELL 无红绿配色**，与 A 股习惯和同项目图表不一致
- **文件**：`frontend/src/app/signals/page.tsx:810-818`（桌面表格无颜色区分）
- **现象**：实测 BUY 与 SELL 单元格的 `color` 均为 `rgba(0, 0, 0, 0.87)`（纯黑），只有悬浮 tooltip 里能靠文字分辨。
  - 而 `components/charts/TradingViewChart.tsx:524-525` 用的是 **A 股口径**（`upColor: '#ef5350'` 红涨 / `downColor: '#26a69a'` 绿跌）；
  - `components/mobile/MobileSignalCard.tsx:24-29` 用的是 **欧美口径**（BUY→`success` 绿 / SELL→`error` 红）；
  - `today-signal/page.tsx:462` 信号质量卡用 `success.main`（绿）表示正超额、`:569` 因子贡献正数用绿 —— 也是欧美口径。
  - **同一产品里 A 股口径和欧美口径混用**，移动端卡片甚至与自家 K 线图相反（用户看图习惯红涨，到卡片里买入却是绿色）。
- **复现步骤**：`/signals` 生成 → 看表格全黑；对比打开任意 K 线图（红涨绿跌）→ 再看移动端信号卡（买=绿）。
- **具体改法**：统一为 A 股口径并抽成常量（`src/utils/colors.ts`）：`upColor='#ef5350'` / `downColor='#26a69a'`，在 `MobileSignalCard` 的 `getSignalColor` 里把 BUY→红、SELL→绿；`today-signal` 的正超额改红、负超额改绿（或至少保持信号与涨跌两套语义分离并加图例说明）。

### P2-12 · 仪表盘「系统状态」全是**硬编码假数据**，且与真实后端健康状态脱钩
- **文件**：`frontend/src/app/dashboard/page.tsx:443-493`
- **现象**：
  - 「API服务 **正常**」「数据服务 **正常**」「模型服务 **负载高**」「系统负载 **65%**」全部是写死的 JSX 字符串。
  - `:83-93` 真实调了 `/monitoring/health` 并算出 `systemHealth`（`'good'|'warning'|'error'`），但**这个值在 JSX 里完全没被使用** —— 只有 `:60/:93/:189` 三处赋值，渲染时无一处引用（grep 确认）。**后端挂了页面照样显示「正常」。**
  - `:188` 错误重试分支里还硬写了 `dataFiles: 156`。
- **复现步骤**：打开 `/dashboard` → 「系统状态」三行状态恒定不变；停掉后端再刷新，页面仍显示「API服务 正常」。
- **具体改法**：
  1. 把 `systemHealth` 真正接到 JSX：用它驱动三个服务 Chip 的颜色与文案；
  2. 删除写死的 65% 和「负载高」，改为读 `/monitoring/health` 返回的真实指标；无数据时显示「未知」而非假值；
  3. 删掉 `:188` 的 `dataFiles: 156`。

### P2-13 · 仪表盘与「今天该买什么」**零关联**，付费用户登录后第一眼看不到任何信号/收益
- **文件**：`frontend/src/app/dashboard/page.tsx:234-543`（整页）
- **现象**：登录后的默认落地页是 `/dashboard`（`src/app/page.tsx` → dashboard）。该页只有 4 个任务计数卡（实测全 0）、最近任务、硬编码系统状态、快速操作。**没有任何一张卡展示今日信号、组合总值、收益或持仓。** 付费用户「3 秒内知道今天该买什么」在这个入口上完全拿不到。
- **复现步骤**：登录 → 落地 `/dashboard` → 页面无一个与交易/信号/收益相关的数字。
- **具体改法**：在仪表板顶部加「今日信号摘要」条（复用 `/trading/signal` 的数据）：组合总值 / 今日买 X 卖 Y / 信号日期 / 距上次调仓天数 / 今日操作金额，点击跳 `/today-signal`。这是本次审计里对**信息层级**提升最大的一条。

### P2-14 · 「策略信号」页策略下拉**只有研究型策略**，与纸面盘/今日信号的 22f / 42f / GBM 线完全脱节
- **文件**：`frontend/src/app/signals/page.tsx:195`（`DataService.getAvailableStrategies()` → `/backtest/strategies`）
- **现象**：实测下拉默认是「**布林带策略（bollinger）**」，可选的是 bollinger / stochastic / cci 等技术类策略。而真正在跑纸面盘、对外收费的 6 条线（`honest_42f` / `honest_42f_neu` / `22f` / `factor_ranking` / `gbm_voltgt` / `rsi_71421`）**一条都不在这个下拉里**（它们在 `/trading/strategies`，不在 `/backtest/strategies`）。
  - 结果：用户进「策略信号」页研究的是 bollinger，而实盘要跟的是 42 因子复合 —— **两页结论互相矛盾，用户不知道该信哪个**。
- **复现步骤**：`/signals` → 策略下拉 → 观察列表内容。
- **具体改法**：
  1. `/signals` 的策略下拉**合并** `/trading/strategies` 的 6 条付费线（用分组：`付费信号线` / `研究用技术策略`），付费线置顶；
  2. 选中付费线时，表头明确标注口径（`42 因子等权 · 持仓上限 60 · 周频调仓`）；
  3. 页面顶部加一句「本页信号 ≠ 今日信号执行清单，后者见 /today-signal」的导航提示。

### P2-15 · 「纸面信号」页回测口径数字**硬编码在 JSX 里**，与实际纸面盘数据脱节且无来源标注
- **文件**：`frontend/src/app/paper-signal/page.tsx:170-173`
- **现象**：黄条里写死「回测口径：22f 周频 / 实盘成本；2021-2026 OOS **9.38% 年化 / -27.1% 回撤 / 0.685 夏普**」。这些数字：
  - 与同页显示的实际纸面盘（组合总值 ¥998,796 vs 初始 100 万 → 约 **-0.12%**）不是同一口径也没有说明关系；
  - 无来源链接/生成时间，用户无法核对；
  - 数字写在 JSX 里，后端回测结果更新后不会跟着变。
- **复现步骤**：`/paper-signal` → 读黄条 → 与上方「组合总值 ¥998,796 / 信号日期 2026-10-16」并列看，两者无法互相印证。
- **具体改法**：
  1. 把回测数字移进 `paper-signal.json`（由生成脚本写入，带 `generated_at` 与口径说明），前端只渲染不写死；
  2. 明确写「以下是 2021-2026 滚动 OOS 回测（研究口径），下方是 2026-10 起的纸面实跑（验证口径），两者不可直接相减」；
  3. 若 22f 仅为对照基线，标注「对照线，非推荐主线」。

### P3-16 · 「今日信号」页 **费用列口径在买/卖之间不一致**，且与「预计成本」卡对不上
- **文件**：`frontend/src/app/today-signal/page.tsx:547`（买：`commission + slippage`）、`:623`（卖：`commission + stamp_tax + slippage`）、`:233-236`（`totalCost` 只累加**买入**的 commission + slippage）
- **现象**：
  - 买入表「费用」= 佣金+滑点（正确，买无印花税）；卖出表「费用」= 佣金+印花税+滑点（正确）—— 但两列**同名「费用」口径不同**，用户横向对比时会误读。
  - 「预计成本 ¥202.52」只统计了**买入**侧（10 笔买入约 ¥202），**完全不含 10 笔卖出的费用**（实测卖出侧约 ¥275）。用户拿这个数字当「今天交易的总成本」会低估一半。
- **复现步骤**：`/today-signal` → 「本次操作 买 10 · 卖 10」→ 下方「预计成本 ¥202.52 含佣金与滑点」→ 手动把买入表费用列 + 卖出表费用列相加 = ¥202.52 + ¥275 ≈ ¥477，是卡上数字的两倍多。
- **具体改法**：
  1. `totalCost` 改为买卖两侧合计：`buys.reduce(...commission+slippage) + sells.reduce(...commission+stamp_tax+slippage)`；
  2. 买入/卖出两表的列名区分：买入「费用(佣金+滑点)」、卖出「费用(佣金+印花税+滑点)」，或统一拆成 3 个子列。

### P3-17 · 「今日信号」页把**研究脚本的持仓**呈现为可直接下单清单，缺「这是模拟盘」的显著声明
- **文件**：`frontend/src/app/today-signal/page.tsx:244`（副标题「收盘后自动生成，含股数 / 金额 / 成本，可直接照着下单」）
- **现象**：页面顶部只说「可直接照着下单」，但：
  - 这批「买入/卖出」实际来自**纸面盘账本**（`data/paper/portfolio.json`），即**模拟成交**，不是实盘委托；
  - 页面上**唯一**的免责提示是「有信号没成交（实盘同样买不进/卖不出）」，那是**针对被拒的信号**说的，正常成交的清单没有任何「这是模拟盘记录」的字样；
  - 参考价是**信号日收盘价**（`fill_assumption: '信号在收盘后生成，按下一交易日开盘/收盘价成交'`），不是用户次日实际能拿到的成交价；页面顶部无此提示（提示在第 429 行的长 Alert 里，位置偏下）。
- **复现步骤**：`/today-signal` → 读副标题「可直接照着下单」→ 通读全页，页面**未声明**这些是纸面盘模拟成交。
- **具体改法**：
  1. 副标题改为「**纸面盘模拟成交清单**（参考价=信号日收盘价，实际成交以次日开盘/收盘为准）」；
  2. 在买入表上方加一条 `Alert severity="warning"`：「本清单由平台纸面盘生成，**尚未接入券商自动下单**；实盘请自行核对价格与可成交性」；
  3. 明确披露成交假设（当前 `fill_assumption` 只写在底部 Alert，建议上移）。

### P3-18 · 「策略信号」页 15 个数值控件在窄屏挤成一行，且**没有任何校验/默认值提示**
- **文件**：`frontend/src/app/signals/page.tsx:457-552`（控制条）
- **现象**：控制条含 策略类型 / 策略 / 股票池来源 / 最近N个交易日 / 分页大小 / 全量生成开关 / 生成信号按钮 + 下方策略参数表单 + 信号筛选 + 起止日期，实测桌面 1280px 下已换行，390px 下极挤（截图 `05-signals-mobile.png`）。「最近N个交易日」「分页大小」是 `type="number"` 的 TextField，输入非法值时 `Number(e.target.value || 60)` 静默回退，无任何提示。
- **复现步骤**：`/signals` → 移动端 → 看控制条；再把「分页大小」清空 → 值静默变 200。
- **具体改法**：控制条在 `xs` 断点改 `direction="column"`（`:457` 已有 `direction={{xs:'column',md:'row'}}`，但子元素仍靠 `minWidth` 撑开，需逐项给 `width: '100%'`）；数值输入加 `onBlur` 校验 + 错误提示。

### P3-19 · 「今日信号」页**没有基准/超额对比**，用户无法判断绝对收益好坏
- **文件**：`frontend/src/app/today-signal/page.tsx:328-421`（6 张 KPI 卡）
- **现象**：6 张卡是 组合总值/最大回撤/年化夏普/持仓/本次操作/预计成本。**全页 grep「基准」/「benchmark」命中 0 次**。当前 honest_42f 组合总值 ¥983,932.94（-1.61%），用户无从判断这是跑赢还是跑输沪深300 —— 而同期 A 股 10 月大概率是上涨的，-1.61% 可能是**大幅跑输**。
- **复现步骤**：`/today-signal` → 看 6 张卡，无任何基准对比项。
- **具体改法**：在「组合总值」卡内加一行「同期沪深300 **+X.XX%**（超额 **-Y.YY%**）」，基准取同窗口等权或沪深300指数；这同时解决了「信号质量」卡里「相对全市场等权」有基准、但**净值曲线本身没基准**的断层。

### P3-20 · 「今日信号」页持有 60 只股票，**表格一次性铺开 60 行**，无搜索/筛选/折叠
- **文件**：`frontend/src/app/today-signal/page.tsx:640-663`
- **现象**：「当前持仓（60）」表无排序、无搜索、无分页、无折叠，60 行全渲染；「买入日」列是 `align="right"` 的日期串（与其他数字列右对齐混排，视觉上像数值）。
- **复现步骤**：`/today-signal` → 滚到「当前持仓（60）」→ 60 行直排。
- **具体改法**：加「按买入日/成本价排序」+ 输入框搜索代码或名称；日期列改 `align="left"` 或用 `nowrap` + 单独列样式；60 行以上改折叠（默认收起，显示 Top10 + 「展开全部」）。

---

## 二、先修清单（按 P0/P1 排序）

| # | 严重度 | 问题 | 位置 | 一句话改法 | 截图 |
|---|---|---|---|---|---|
| 1 | **P0** | 今日信号页生产环境完全打不开 | `today-signal/page.tsx:186,208,213` | 后端 `/trading/*` 三个端点补 `{success,data}` 信封（对齐 `StandardResponse`） | `02-today-signal-BROKEN.png` |
| 2 | **P0** | 付费用户被误显示「体验版」+ 升级按钮 | `today-signal/page.tsx:172-173,260,269` | tier 初始值改 null；加载失败时不渲染档位与付费墙 | `02-today-signal-BROKEN.png` |
| 3 | **P0** | 纸面信号页是 10 天前的旧快照，与今日信号页矛盾 | `paper-signal/page.tsx:99` | 改读 `paper-portfolio.json`；加数据过期守卫 | `03-paper-signal.png` |
| 4 | **P1** | 买入表「分数」「为什么买它」恒为 `—` | `today-signal/page.tsx:551,555` | `rebuild_line_from_files()` 补 `score`；factors 落盘 | `02b-today-signal-buylist.png` |
| 5 | **P1** | 回撤/夏普卡恒为 `—`，并漏出「undefined 点」 | `today-signal/page.tsx:349,364,376` | 合并 `/trading/performance`；`nav_points` 加 null 判断 | `02-today-signal-WORKING.png` |
| 6 | **P1** | 移动端表格右 2–4 列被硬裁且不可滚 | `today-signal:524,602,640`；`paper-signal:64` | 加 `overflowX:'auto'` 包裹；或改移动端卡片 | `06-…-mobile.png`、`08-…-mobile.png` |
| 7 | **P1** | 策略信号页一次渲染 5594 行（文档 186,030px） | `signals/page.tsx:318-347,792` | 虚拟滚动 / 服务端分页；修正页码文案 | `04-signals-generated.png` |
| 8 | **P1** | 140 天前的陈旧信号与今日信号同表无标识 | `signals/page.tsx:805-818` | 表头改口径；陈旧行降级 + 「只看近5日」筛选 | `04-signals-filter-BUY.png` |
| 9 | **P1** | 纸面持仓 JSON 无鉴权可下载 | `frontend/public/paper-signal.json` | 移出 `public/`，改受保护端点下发 | `03-paper-signal.png` |
| 10 | **P1** | 仪表盘与「今天买什么」零关联（付费墙体验断点） | `dashboard/page.tsx:234-543` | 顶部加「今日信号摘要」条 + 跳转 | `01-dashboard-desktop.png` |
| 11 | **P1** | 策略信号页策略下拉只有 bollinger，与付费 6 线脱节 | `signals/page.tsx:195` | 合并 `/trading/strategies`，付费线置顶分组 | `04-signals.png` |
| 12 | **P2** | 仪表盘系统状态全是硬编码，真实健康度未接线 | `dashboard/page.tsx:443-493` | 把 `systemHealth` 接到 JSX；删 65% | `01-dashboard-desktop.png` |

---

## 三、总体评价

**这组页面目前是「能看不能用」的状态：设计意图（股数/金额/成本/分数/原因/夏普/回撤/信号质量）相当完整，纸面盘与信号服务的后端数据也基本齐备，但中间那一层 API 信封约定没对齐，导致最核心的「今日信号」页在生产环境 100% 报错、付费用户还被反向引导去重复付费 —— 这是必须今天就修的 P0。**

修完 P0-1 之后，页面能亮起来，但**「3 秒内知道今天该买什么、为什么买、买多少、风险多大」这个目标仍然只达成一半**：风险指标恒为 `—`、「为什么买」恒为 `—`、「预计成本」少算一半、没有基准对比、纸面模拟成交被写成「可直接照着下单」—— 这些是 P1 那一列要解决的。移动端表格硬裁（金额和费用都看不到）在「用手机照单下单」的场景下是硬伤，必须在 P0 修完后立即处理。

策略信号页（5594 行全渲染 + 陈旧信号混排）目前更像**研究工具**，但它和付费信号页挂在同一导航层、策略集合还不一样，用户会误以为两边结论等价 —— 在它和「今日信号」的线对齐之前，建议在页面上明确降级为「研究/回看工具」定位，避免稀释付费信号的可信度。
