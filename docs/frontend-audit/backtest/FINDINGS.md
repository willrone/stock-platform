# 研究工作台前端只读审计报告

**审计范围**：`frontend/src/app/{backtest,paper-portfolio,tasks/*,optimization,models,templates}` + `components/{backtest,optimization,tasks,charts}/*`
**审计方式**：真实浏览器访问 + 实际渲染截图 + 后端 API/DB 交叉验证（未修改任何源码）
**环境**：前端 `http://localhost:3000`（Next.js 14 dev），后端 `http://127.0.0.1:18082`
**账号**：`ronghui@example.com`（DB 中 email 为 `ronghui@example.com`，用户名 `ronghui`）
**日期**：2026-10-27

---

## 0. 审计方法与两个必须先说清楚的前提

### 0.1 登录口径
`ronghui` 是 **username** 而非 email，直接填 `ronghui` 登录会报「邮箱或密码错误」。正确 email 是 `ronghui@example.com`（`backend/data/app.db` 的 `users` 表确认）。

### 0.2 限流对本审计的影响（重要）
后端 `backend/app/middleware/rate_limiting.py` 的限流 **只按 IP 计数，不按用户**（`_get_client_id()` 只看 `X-Forwarded-For` / `client.host`）：

| 维度 | 配置 | 位置 |
|---|---|---|
| 每分钟 | 60 | `backend/app/main.py:234` |
| 每小时 | 1000 | `backend/app/main.py:235` |
| 突发桶 | 10 | `backend/app/main.py:236` |

单页加载的 API 请求扇出就已经接近突发桶上限（回测页 1 次列表 + 布局健康检查 5s 轮询 + WS）。
实测在本审计期间，**同一个 IP 在 1 小时内打满 1000 次后，整个站点的所有页面开始返回 429**：
- `/backtest` → 「加载失败 · 请求过于频繁」
- `/tasks` → 表格空白但**不报错**（见 P1-06）
- 甚至 `/api/v1/auth/refresh` 被 429 → 前端清空 token 跳登录页（见 P1-05）

这直接导致 P0-01 ~ P0-04 的「页面空白」现象有部分是被限流放大的。**下文每条问题都标注了是在限流未触发（已验证）还是限流下才出现的。**

---

## 1. 问题清单

### P0-01　交易策略下拉框完全空白，但提交时静默使用 `moving_average`
- **文件**：`frontend/src/app/tasks/create/page.tsx:126`（策略列表请求）、`:91`（默认策略）
- **现象**：创建回测任务页的「交易策略 \*」是一个 **MUI Select，输入框有值 `moving_average` 但显示为空、点开无任何选项（0 个 option）**。用户看不到任何策略可选；填完日期提交后，任务实际以 `moving_average` 运行，页面上却从未告知过用户。
- **复现**：
  1. 登录 → 导航「策略回测」→「创建回测任务」
  2. 观察「交易策略 \*」下拉框：显示空白（`input.value === 'moving_average'`，`li[role=option]` 数量 = 0）
  3. 同一页面的「超参优化 → 创建任务」表单里，同一个 `StockSelector` 却能正常列出 5594 只股票
- **根因**：`create/page.tsx:126` 用的是**裸 `fetch`**，没有带 `Authorization` 头：
  ```ts
  const response = await fetch('/api/v1/backtest/strategies');   // ← 无 Authorization
  ```
  后端 `/api/v1/backtest/strategies` 需要 Bearer（实测无 token 返回 `401 缺少 Authorization 请求头`）。请求失败后 `setAvailableStrategies` 从未被调用，Select 没有任何 `<MenuItem>`，MUI 于是渲染成空白。
  **对比**：优化表单用的是 `apiRequest`（`services/api.ts:161-175` 有请求拦截器注入 token），所以那里策略列表正常。
- **改法**：
  1. 把 `create/page.tsx:126` 与 `:171` 的裸 `fetch` 换成 `apiRequest.get('/backtest/strategies')` 与 `apiRequest.get(\`/tasks/${id}\`)`（`apiRequest` 已带 baseURL `/api/v1` + token + 401 刷新重试）。
  2. 加载失败时不要静默 —— `catch` 里 `setStrategyLoadError(true)` 并在 Select 下方渲染 `<Alert severity="error">策略列表加载失败，请刷新重试</Alert>`，同时 **禁用提交按钮**。
  3. 加载完成前把 `strategy_name` 置空（而非默认 `moving_average`），并在校验里加 `if (!formData.strategy_name) newErrors.strategy_name = '请选择交易策略'`。当前 `validateForm()`（`:322-355`）**根本不校验 strategy_name**，所以空白也能提交。
  4. 顺带修 `fetch('/api/v1/backtest/strategies')` 之外同页 `:171` 的 `fetch('/api/v1/tasks/' + rebuildTaskId)`（同样无 token，「重建任务」必然失败）。
- **截图**：`10-strategy-dropdown-empty.png`、`09-backtest-create-form.png`

---

### P0-02　参数调优台的基准任务下拉框永远为空（URL 双前缀 `/api/v1/api/v1/...`）
- **文件**：`frontend/src/app/tasks/tuning/page.tsx:170`、`:267`、`:305`
- **现象**：「参数调优台 🎛️ ① 选择基准任务」的下拉框没有任何任务可选，页面永远停在第一步，无法进入调参流程。页面**不显示任何错误提示**。
- **复现**：
  1. 登录 → 导航「参数调优」→ 打开 DevTools Network
  2. 观察到请求 URL 为 `/api/v1/api/v1/tasks?limit=200&status=completed`（**双前缀**）
  3. 页面文本恒为「① 选择基准任务 / 选择已完成回测任务」，后续 ②③④ 步骤永不出现
- **根因**：`apiRequest` 的 axios 实例 `baseURL: '/api/v1'`（`services/api.ts:135`），axios 的 `combineURLs` 会把 `baseURL` 与 `url` 直接拼接。传入 `'/api/v1/tasks?...'` → 最终 `/api/v1/api/v1/tasks?...` → 后端 404。同一文件 `:267`/`:305` 的 `preview-config` / `submit` 同样带 `/api/v1` 前缀，会一起 404。
- **改法**：把 `tuning/page.tsx` 中 3 处 `apiRequest.get/post('/api/v1/...')` 改为 `apiRequest.get/post('/tasks/...')`（去掉前缀）。同时在 `catch` 里 `setError('加载任务列表失败')` 并在页面顶部渲染错误条（当前 `catch` 只 `logger.error`，见 `tuning/page.tsx:169-179`）。
  > 顺带建议：给 `apiRequest` 加一个开发期断言，若 `url.startsWith('/api/v1')` 就 `console.warn`，避免同类问题复发。
- **截图**：`07-tasks-tuning.png`、`17-tuning-double-prefix.png`

---

### P0-03　回测任务列表的「总收益 / 夏普」列**永远**是 "-"，已完成的回测看起来像没跑
- **文件**：`frontend/src/app/backtest/page.tsx:210-222`（读取逻辑）、`:432-441`（渲染）、`frontend/src/types/task.ts:13-14`
- **现象**：跑完一个回测（实测 `audit_smoke_rsi`：总收益 +10.78%、夏普 2.478），回到「策略回测」列表页，该行的「进度」列显示 **"-"** —— 既不显示收益也不显示夏普，也不显示 100% 进度。同一任务在详情页却有完整指标。用户看到列表会以为回测没产出结果。
- **复现**：
  1. 创建并跑完一个回测 → 详情页显示「总收益率 +10.78% / 夏普比率 2.478」
  2. 返回 `/backtest` 列表 → 该行「进度」列 = `-`
  3. DevTools 观察 `/api/v1/tasks` 响应：含 `result_summary: {total_return: 0.1078, sharpe_ratio: 2.478, ...}`，但 **不含 `result` / `results` 字段**
- **根因**：**前后端字段名不一致**。后端 `backend/app/api/v1/schemas.py:378-422` 的 `extract_result_summary()` 把指标放在 `result_summary` 里（且列表接口 `exclude_result=True` 不加载 `result` 大字段），但前端 `getBacktestSummary()` 读的是 `task.results?.backtest_results || task.backtest_results || task.result` —— **这三个字段在列表响应里都不存在**，`getBacktestSummary` 恒返回 `null`，渲染走 `else` 分支输出 `-`。
- **改法**：
  1. `backtest/page.tsx:211` 改为优先读 `task.result_summary`：
     ```ts
     const backtestData = task.result_summary ?? task.results?.backtest_results ?? task.result;
     ```
  2. `types/task.ts` 的 `Task` 接口补 `result_summary?: { total_return?: number; sharpe_ratio?: number; max_drawdown?: number; win_rate?: number; total_trades?: number; annualized_return?: number; volatility?: number; }`。
  3. 顺手把「进度」列拆成两列：**进度**（`LinearProgress` + 百分比，任何状态都显示）和 **指标**（总收益 / 夏普 / 回撤 / 换手）。现在这一列把「进度」和「收益」两个语义混在一起，列名叫「进度」但已完成的行显示收益，是设计缺陷。
  4. 同一个字段不一致也影响 `backtest/page.tsx:64-89` 的「是否回测任务」过滤 —— 该过滤依赖 `task.result`（列表里没有），所以已完成的回测任务**可能连行都渲染不出来**。实测因为 `task.config?.strategy_name` 命中才侥幸显示出来；一旦有回测任务的 config 里没有 `strategy_name`（如 `model_signal` 模式），该行会消失。
- **截图**：`20-backtest-list-with-task.png`（对比 `11-task-detail.png` 同一任务的指标）

---

### P0-04　「策略赛马」视图缺失核心对比维度：只有收益率，没有夏普 / 回撤 / 换手率 / 净值曲线
- **文件**：`frontend/src/app/paper-portfolio/page.tsx:130-190`（卡片渲染）、`:192-264`（明细）
- **现象**：页面标题写着「多策略纸面盘（策略赛马）」，6 条策略线横向排列，但每张卡只有 **市值 / 累计收益率 / 持仓只数 / 一句话描述** 4 个信息。用户最关心的赛马维度**一个都没有**：
  - 没有夏普比率、**没有最大回撤**、没有年化收益
  - **没有换手率 / 年换手次数**（用户明确反感高频，这是最该暴露的指标）
  - **没有净值曲线 / 对比图**（只有静态 JSON 快照，没有图表）
  - 6 条线之间**没有相对排名 / 排序**，也没有「谁跑赢基准」的结论
  - 累计收益率是**单次调仓区间的收益**（2026-10-23 信号日 → 2026-10-26 成交日，3 个交易日），却和「100 万初始资金」并列显示 → `+0.83%` 这种量级会被误读成「策略长期表现」
- **复现**：
  1. 登录 → 导航「多策略纸面盘」
  2. 观察 6 张策略卡：gbm_voltgt `¥999,477.4 / -0.05%`、22f `-0.91%`、honest_42f `-1.61%`、rsi_71421 `+0.83%`、factor_ranking `-1.28%`、honest_42f_neu `-0.11%`
  3. 页面无任何图表，无夏普/回撤/换手列
- **根因**：数据源 `frontend/public/paper-portfolio.json` 的 `lines[*]` 只有 `{strategy, desc, date, cash, value, n_positions, sells, buys, holds}`，**没有 nav 历史、没有 sharpe、没有 drawdown、没有 turnover**。而生成脚本 `backend/scripts/paper_portfolio.py:709-720` 其实**已经在写 `nav_history`**（`{date, value, return_pct}`，保留最近 800 点），但 `paper_portfolio.py:1020-1050` 组装最终 JSON 时**没有把它写进 `lines[*]`**（实测 JSON 里 6 条线全部没有 `nav_history` 键）。
- **改法**：
  1. **后端**：`paper_portfolio.py:1031-1033` 组装 `lines` 时把 `nav_history` 一并输出（`rebuild_line_from_files()` 已经能从持仓/ledger 侧取数，数据源在 `paper_portfolio.py:709-720` 的 `nav_history` 累积逻辑，只需把它写进最终 `lines[*]`）；同时由 `nav_history` 现算每条线的 `sharpe`（年化，按周频 52 期折算）、`max_drawdown`、`annualized_return`、`turnover`（年换手 = 换手股票数×2 / 持仓数 / 年数）。
  2. **前端**：`paper-portfolio/page.tsx` 的卡片改为一张**赛马汇总表**：策略 / 累计收益 / 年化 / 夏普 / 最大回撤 / 年换手 / 当前持仓 / 最近调仓日，**按夏普降序排列**；并把 `nav_history` 用 `EquityCurveChart`（echarts，已存在）画成 **6 条线叠在一张图**上的对比净值曲线，共享坐标轴。
  3. **必须在 UI 上区分「本期收益」与「累计/年化收益」**，当前把 3 个交易日的区间收益和「100 万本金」并排，误导性最强。
  4. **必须显式暴露换手率**。这直接回应用户「反感高频」的诉求：gbm_voltgt 月度滚动重训 + 高换手策略应当被自动标注为「高频」并给出警告，而不是和周频策略并排比收益。
- **补充**：DB 里已经存在一份**现成的 16 策略全市场 3 年基线**（task `baseline_baseline_full_market_3y`，`result.strategies` 含 moving_average/rsi/macd/bollinger/... 共 16 条，每条都有 `total_return / sharpe_ratio / max_drawdown / total_trades / annualized_return / win_rate`），`result.benchmark` 还有基准线。但**前端没有任何视图读它** —— 这正是用户要的「多策略赛马」，数据现成，缺一个页。示例数据（可直接用作赛马视图的验收基准）：

  | 策略 | 总收益 | 年化 | 夏普 | 最大回撤 | 交易次数 | 胜率 |
  |---|---|---|---|---|---|---|
  | bollinger | +80.15% | 21.74% | 0.611 | -27.27% | 2490 | 68.0% |
  | stochastic | +112.43% | 28.64% | 0.899 | -26.65% | 1929 | 69.8% |
  | obv | +77.23% | 21.08% | 0.549 | -46.71% | **28351** | 23.7% |
  | macd | +49.88% | 14.48% | 0.349 | -48.20% | **13448** | 29.1% |
  | factor_ranking | +38.53% | 11.51% | 0.924 | -10.59% | 3326 | 54.7% |
  | low_volatility | +40.14% | 11.94% | 0.246 | -47.01% | **8** | 0.0% |
  | moving_average | -38.10% | -14.81% | -0.445 | -54.68% | 2533 | 38.9% |

  > 注意 `obv`（28351 笔）夏普 0.549、`macd`（13448 笔）夏普 0.349 —— **换手越高的策略，夏普越低**。这正是用户想要的证据，但当前 UI 完全呈现不出来，用户只能靠"收益高=策略好"来判断。
- **截图**：`02-paper-portfolio-race.png`、`24-paper-portfolio-cards.png`

---

### P1-01　纸面盘参数行渲染成 `TopK= / 每次换 只 / 单仓目标权重 NaN%`
- **文件**：`frontend/src/app/paper-portfolio/page.tsx:197-201`
- **现象**：策略明细标题下方一行显示 `TopK= / 每次换 只 / 单仓目标权重 NaN% / 现金 ¥824,667.4` —— 三个数字全空，目标权重是字面量 `NaN%`。
- **复现**：登录 →「多策略纸面盘」→ 看任一策略的「调仓明细」标题行（默认选中 gbm_voltgt）
- **根因**：`paper-portfolio.json` 的 6 条线**全部**缺 `topk` / `ndrop` / `target_weight`（实测 6/6 missing）。生成脚本里 `run_strategy()` 有这三个字段（`paper_portfolio.py:761`），但**回退路径 `rebuild_line_from_files()`（`:774` 起）没有补**，而本次运行 `meta.lines_updated` 为 `[]`（全部走了 rebuild 路径）。前端 `(line.target_weight * 100).toFixed(2)` → `null * 100 = 0`… 实际 `null` 参与算术得 `NaN`。
- **改法**：
  1. **后端**（根治）：`rebuild_line_from_files()` 里从 `STRATEGIES[strategy]` 取 `topk`/`ndrop`，`target_weight = (1.0 - CASH_RESERVE) / topk` 补齐，与 `run_strategy()` 保持一致。
  2. **前端**（兜底）：`paper-portfolio/page.tsx:198-199` 对空值做保护，`line.topk ?? '—'`，并对 `target_weight` 做 `Number.isFinite` 判断，避免 NaN 直接漏到 UI。
  3. 建议在这行后面补上**换手信息**（TopK=60 / 每次换 5 只 → 隐含年换手），呼应 P0-04。
- **截图**：`02-paper-portfolio-race.png`、`24-paper-portfolio-cards.png`（文字部分见 innerText 抽取）

---

### P1-02　「回测预览」指标是**硬编码的假数据**，会直接误导对策略真实表现的判断
- **文件**：`frontend/src/app/templates/page.tsx:53-61`、`:87-95`、`:121-129`、`:284-380`（渲染）
- **现象**：「策略模板」页每个模板卡片都展示「回测预览 (2020-2024)：总收益 +32.5% / 夏普 1.45 / 最大回撤 -18.2% / 年化 12.8% / 胜率 48.3% / 交易次数 126」，看起来像真实回测结果。**但这些是源码里的字符串常量，与任何回测任务无关，从不调用后端。**
- **复现**：登录 →「策略模板」→ 观察三张卡片的「回测预览 (2020-2024)」区块；`grep totalReturn src/app/templates/page.tsx` 可见均为字面量
- **根因**：`strategyTemplates` 常量数组里硬编码 `backtestPreview` 对象（`page.tsx:53`、`:87`、`:121`），渲染时直接取 `preview.totalReturn` 等（`:303`、`:317`、`:331`、`:345`、`:359`、`:373`）。
- **改法**（**优先级高，因为这是"假收益"**）：
  1. **最低成本**：删掉 `backtestPreview` 整块，改为「该模板尚未在本平台跑过回测」+「用此模板创建回测」按钮。**不能让用户把 1.45 的夏普当成真实预期。**
  2. **更好**：让模板页调用后端跑一次真实基线（离线预跑 + 缓存结果 + 标注「结果生成于 YYYY-MM-DD，策略 vX，全市场 N 只，YYYY-YYYY 区间」）。渲染时必须带**口径脚注**（股票池大小、区间、是否含成本）。
  3. 顺带：这三个模板的 `strategy_name` 里有 **`ma_cross_strategy` 和 `momentum_strategy` 在后端 29 个可用策略中并不存在**（实测 `ma_cross_strategy: False`、`momentum_strategy: False`）→ 一键使用后必然回测失败（见 P1-03）。
- **截图**：`04-templates.png`

---

### P1-03　模板「一键使用」跳转后，**6 个预填参数全部丢失**，表单是空的
- **文件**：`frontend/src/app/templates/page.tsx:150-161`（生成参数）、`frontend/src/app/tasks/create/page.tsx:147-162`（读取参数）
- **现象**：模板页点「一键使用」→ 跳到 `/tasks/create?type=backtest&templateId=ma_cross&strategy=ma_cross_strategy&startDate=2020-01-01&endDate=2024-12-31&initialCash=1000000&config={...}`，URL 参数齐全，但**创建页一个都没读**：
  - 回测开始日期 = 空、结束日期 = 空（模板本是 2020-01-01 ~ 2024-12-31）
  - 初始资金 = **100000**（模板是 1000000，差 10 倍）
  - 交易策略 = 空白（实际会静默用 `moving_average`）
  - 股票池 = 空（模板指定 `['000300.SH','000905.SH','399006.SZ']` 三个指数）
  - 策略参数（`fast_ma: 5, slow_ma: 20, ma_type: 'SMA'`）= 未填
  - 任务名称 = 空
- **复现**：
  1. 登录 →「策略模板」→ 点「简单均线策略」的「一键使用」
  2. 观察 URL 参数齐全，但表单全空（字段值实测：`回测开始日期=[]`、`回测结束日期=[]`、`初始资金=[100000]`）
- **根因**：`templates/page.tsx:153-158` 写入 `templateId / strategy / startDate / endDate / initialCash / config` 共 6 个参数；`create/page.tsx:148-164` 的 `searchParams.get()` **只读 `type`、`mode`、`model_id`、`rebuild`** 4 个，完全不认这 6 个。
- **改法**：
  1. `create/page.tsx` 增加读取：`const tplId = searchParams.get('templateId')`、`strategy`、`startDate`、`endDate`、`initialCash`、`config`（JSON.parse 后进 `strategyConfig`），并把 `templateId` 对应的 `stock_codes` 一并预填。
  2. 读取失败（策略不存在于后端列表）时，**明确提示**「模板策略 `ma_cross_strategy` 在当前后端不可用，已回退到 moving_average」，而不是静默。
  3. 修 `templates/page.tsx` 里的策略名：`ma_cross_strategy` → 后端实际支持的 `moving_average`（或在后端注册别名）；`momentum_strategy` → `momentum_factor`；`multi_factor` 存在，可用。
- **截图**：`28-templates-oneclick.png`

---

### P1-04　任务详情页把**任何加载失败**都显示成「任务不存在或已被删除」
- **文件**：`frontend/src/app/tasks/[id]/useTaskDetailPage.ts:105-107`（吞异常）、`frontend/src/app/tasks/[id]/TaskDetailPageView.tsx:123-143`（渲染假文案）
- **现象**：一个**确实存在、已成功完成**的回测（`a000c692-…`，DB 状态 `completed`，API 返回 200），打开详情页却显示「任务不存在或已被删除 / 返回任务列表」。用户会以为回测被删了，浪费时间重建。
- **复现**（token 过期后复现率高）：
  1. 登录后静置 > 30 分钟（`JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30`，`backend/app/core/config.py:87`）
  2. 打开任意已完成任务的详情页 → 显示「任务不存在或已被删除」
  3. 同期 DevTools 显示 `/api/v1/tasks/<id>` 返回 **401**（或被限流时 429）
  4. 刷新一次（token 刷新成功后）→ 页面正常
- **根因**：`useTaskDetailPage.ts:105-107` 的 `catch (_error) { // noop }` 把所有错误吞掉，`currentTask` 保持 `null`；`TaskDetailPageView.tsx:123` 只判断 `!currentTask` 就渲染「任务不存在或已被删除」，**不区分 404 与 401/429/网络错误**。
- **改法**：
  1. `useTaskDetailPage.ts` 增加 `const [loadError, setLoadError] = useState<string|null>(null)`，在 `catch` 里 `setLoadError(...)`；区分 `error.status === 404`（真不存在）与其他（加载失败）。
  2. `TaskDetailPageView.tsx:123` 分三种渲染：404 → 「任务不存在或已被删除」；401/429 → 「会话已过期 / 请求受限，请重试」+ 重试按钮；网络错误 → 「加载失败」+ 重试按钮。
  3. 顺手修 `loadBacktestDetailedData` 的 `catch`（`:74-78`）——它把 detailed 数据失败也静默成 null，图表页会显示「暂无回测数据」而不是「图表数据加载失败」。
- **截图**：`26-task-detail-false-notfound.png`（对比正常的 `11-task-detail.png`）

---

### P1-05　限流 429 打在 token 刷新上 → **用户被强制登出**
- **文件**：`frontend/src/services/api.ts:227-256`（401 刷新流程）、`backend/app/middleware/rate_limiting.py:112`（限流先于路由执行）
- **现象**：access token 过期（30 分钟）后，只要 refresh 请求命中 429，前端就 `clearStoredAuth()` + `redirectToLogin()`，**把用户踢回登录页**，且不告诉用户发生了什么。
- **复现**：
  1. 登录，静置 30 分钟让 access token 过期
  2. 同时让该 IP 的小时配额打满（1000 次）
  3. 打开任意页面 → 跳回 `/login`，`localStorage` 中 `access_token` / `refresh_token` / `user_info` 全被清空
- **根因**：`api.ts:227` 起，401 触发 `refreshAccessToken()`；该函数用裸 `axios.post('/api/v1/auth/refresh')`（`:92`），若返回 429，`:249-253` 的 `catch` 执行 `clearStoredAuth(); redirectToLogin();` —— **把「暂时限流」当成「认证失效」处理**。限流中间件在路由之前执行（`main.py:238`），连 `/auth/refresh` 都会被限。
- **改法**：
  1. `api.ts` 的 refresh catch 里区分状态码：`if (error.response?.status === 429)` → **不清 token、不跳登录**，改为「请求过于频繁，请稍后重试」提示 + 指数退避重试（3 次，2s/4s/8s）。
  2. 401 且 refresh 成功时，原请求已带新 token 重试（现有逻辑 OK）。
  3. 401 且 refresh 401（真过期）才清 token 跳登录。
  4. **后端配套**：`main.py:233-237` 建议把限流改为**按用户 ID 计数**（JWT sub）而非按 IP；或至少对 `/api/v1/auth/*` 路径豁免/放宽，否则 NAT 后的多用户会互相拖累。
- **截图**：`26-task-detail-false-notfound.png`（伴随现象）、`16-backtest-list-empty.png`

---

### P1-06　`/tasks` 列表加载失败**不显示任何错误**，静默呈现「0 个任务」
- **文件**：`frontend/src/app/tasks/page.tsx:94-105`
- **现象**：API 失败（429 或其他）时，`/tasks` 页面只 `logger.error`，页面照常渲染，顶部统计卡显示「总任务数 0 / 运行中 0 / 已完成 0 / 失败 0」，表格区标题行下面一片空白。**用户会以为自己的任务全没了。**
- **复现**：让该 IP 触发限流 → 打开「任务管理」→ 四个统计卡全是 0，表格空，无错误提示（对比 `/backtest` 页会显示红色「加载失败」，两页行为不一致）
- **根因**：`tasks/page.tsx:100-101` 的 `catch (error) { logger.error('加载任务失败:', error); }` —— 只写 console，不更新任何 UI 状态。对比 `backtest/page.tsx:99-104` 有 `setLoadError(errMsg)` + snackbar + 重试按钮。
- **改法**：
  1. `tasks/page.tsx` 增加 `loadError` state，catch 里 `setLoadError(msg)`；在列表上方渲染 `<Alert severity="error" action={<Button onClick={reload}>重试</Button>}>`。
  2. 加载失败时**不要把统计卡渲染成 0**（应显示 `-` 或 `?`），避免"任务被清空"的错觉。
  3. 抽一个共用的 `<LoadErrorState error onRetry>` 组件给 tasks / backtest / models / optimization / tuning 复用，顺带统一四页的错误呈现。
- **截图**：`15-tasks-list-empty.png`（对比 `20-backtest-list-with-task.png`）

---

### P1-07　任务列表**不显示失败原因**，用户必须逐个点进详情页
- **文件**：`frontend/src/app/tasks/page.tsx:484-582`（表格）、`frontend/src/app/backtest/page.tsx:390-479`（回测列表）
- **现象**：失败任务在列表里只有一个红色的「失败」chip，**没有错误文本**。实测 `audit_fail_probe` 的后端 `error_message` 是完整可读的：`TaskError: 回测执行失败: 未知的策略类型: __nonexistent_strategy__，可用策略: [...]`，但列表页完全看不到。用户要点进详情页才知道原因。
- **复现**：
  1. 创建一个会失败的任务（策略名填 `__nonexistent_strategy__`）
  2. 打开「任务管理」→ 失败行的操作列只有一个「重新运行」图标，**没有原因**
  3. 点进详情页才看到完整错误（`23-task-detail-failed.png`）
- **根因**：`error_message` 字段在 API 里是有的（`schemas.py:479`），但两张列表表都没渲染它。`tasks/page.tsx` 只在 WebSocket `task:failed` 事件里把 error 写进 store（`:169`），**刷新页面后 store 里的 error 丢失**，列表就只剩 chip。
- **改法**：
  1. `tasks/page.tsx` 的状态列：状态为 `failed` 时，在 chip 下方用 `<Tooltip title={task.error_message}>` 包一行 `<Typography variant="caption" color="error" noWrap>` 显示错误首行（截断 40 字），或直接加一个「原因」列（可排序、可筛选）。
  2. `backtest/page.tsx` 同样加。
  3. 列表页加一个「只看失败」快捷筛选（状态筛选已有 `failed` 选项，但失败原因不可见导致筛出来还是没法快速定位）。
- **截图**：`21-tasks-list-failed.png`（对比 `23-task-detail-failed.png`）

---

### P1-08　回测参数表单**没有任何口径 / 危险参数警示**，用户无从判断结果可信度
- **文件**：`frontend/src/app/tasks/create/page.tsx:438-436`（整块表单）、`frontend/src/components/backtest/StrategyConfigForm.tsx`（策略参数表单）
- **现象**：创建回测任务页把「股票选择 / 策略 / 日期 / 资金 / 费率 / 滑点」平铺给用户，**没有任何一处提示**：
  - **股票池子集偏差**：随手「随机选择 10 只」就得到一个回测，UI 不提示「这 10 只不是全市场，结果会严重偏离」；也不显示所选股票数占总池的比例、不提示「800 只子集 vs 全市场 5594 只」的口径差异。
  - **幸存者偏差 / 未来函数**：不提示「本地股票列表只包含当前仍存续的股票，退市股不在样本内」——这正是首页 `page.tsx:80` 已经在对用户讲的口径，但在真正做回测的表单里一句都没有。
  - **样本外 / 多次试验**：不提示「这是样本内回测」。
  - **默认参数诱导高频**：初始资金默认 100000（不是 100 万），手续费率 0.0003、滑点率 0.0001 都偏低（A股双边实际约 佣金 2.5‱ + 印花税 5‱(卖出) + 滑点 1‱），**低费率会让高频策略的收益虚高**。
  - **日期默认空**：`:92-93` 的 `start_date: ''` / `end_date: ''` 是空的，页面也没有"最近 3 年"这类一键区间。
  - **无「换手率 / 年化」预告**：跑完才知道交易了多少次。
- **复现**：登录 →「策略回测」→「创建回测任务」→ 通读整页，**找不到任何一句关于口径、偏差、频率的提示**（`grep -rn "未来函数\|幸存者\|偏差\|口径" src/app/backtest src/app/tasks src/components/backtest` 仅命中 `BacktestOverview.tsx` 的「旧口径/可执行口径执行率」两处，无警示语）
- **改法**：
  1. 表单顶部加 `<Alert severity="info">` 口径说明卡：**「本地股票池为当前存续股票（不含退市），回测结果存在幸存者偏差；随机选取的 N 只股票不是全市场子集，N 越小结果越不可信。」**
  2. 股票选择区实时显示：`已选 N 只 / 本地共 5594 只（占 x%）`，并在 N < 100 时给 warning 级提示 `所选股票数过少，结果不具统计意义，建议 ≥ 200 只或改用「组合/截面」策略`。
  3. 费率默认值改成贴近 A 股实盘的口径（佣金 0.00025、印花税 0.0005 卖出、**滑点 0.0005**），并在费率字段旁注明「滑点越低，高频策略收益越虚高」。
  4. 日期给默认值（最近 3 年）+ 快捷按钮（1 年 / 3 年 / 5 年），并提示「建议 ≥ 1 年，过短区间易被噪声主导」。
  5. 在提交前展示一个「回测口径摘要」条：策略 / 股票数 / 区间 / 初始资金 / 总费率 / 预估最大换手，提交后写进任务 config（便于事后复现）。
- **截图**：`09-backtest-create-form.png`、`10-strategy-dropdown-empty.png`

---

### P1-09　优化表单**没有任何过拟合 / 样本外警示**，目标指标默认「稳定赚钱」难以解释
- **文件**：`frontend/src/components/optimization/CreateOptimizationTaskForm.tsx:51-78`（默认值）、`:620-650`（目标指标）
- **现象**：超参优化表单把「试验次数 50 / 优化方法 TPE / 目标指标 稳定赚钱(stability) / 优化方向 最大化」摆出来，但：
  - **没有任何一处提示「这是样本内网格搜索，极易过拟合，必须做样本外验证」**（`grep -n "过拟合\|样本内\|OOS\|多重比较\|数据窥探" CreateOptimizationTaskForm.tsx` 无命中）
  - 默认 `objective_metric: 'stability'`（`:61`）是个自定义复合指标，**页面上没有解释它怎么算的**，用户选了也不知道在优化什么
  - 目标指标下拉里**没有「换手率 / 年换手次数」**（实测 12 项：夏普/卡玛/稳定赚钱/IC/IC_IR/总收益/年化/胜率/盈亏比/最大回撤/成本占比/自定义）—— 对反感高频的用户，这是最该加的一项
  - 「自定义组合」权重里有 `cost_ratio: 0.1`，但没有 `turnover` 权重
- **复现**：登录 →「超参优化」→「创建任务」→ 查看「优化目标」区块
- **改法**：
  1. 表单顶部加 `<Alert severity="warning">`：**「超参优化是在给定区间内搜索最优参数，天然存在样本内过拟合。结果仅在所选区间内有效；建议：① 用 ≥2 年区间；② 选出的参数必须到未参与优化的区间做样本外验证；③ 试验次数越多，过拟合风险越高。」**
  2. 目标指标下拉**加「年换手次数 / 换手率 (Turnover)」**，并在 `objectiveWeights`（`:68-75`）加 `turnover: 0.1` 权重项 —— 直接对应用户「低频少换仓」的诉求。
  3. `stability` 加 tooltip 解释公式（建议展示「年化收益 / 年化波动」或「最大回撤的倒数」的实际定义）。
  4. 试验次数默认 50 → 提示文案已写「参数空间较大时建议 300–500+」，但**没说明试验次数越多越容易过拟合**，与 P1-08 的警示合并处理。
  5. 优化完成后（`OptimizationTaskDetail.tsx:165-178`）应强制展示「样本外验证」区块，否则优化结果页也会给用户「这就是最优参数」的错觉。
- **截图**：`06-optimization-create.png`、`27-optimization-objective.png`

---

### P1-10　「配置对比」页只对比**配置参数**，不对比**回测结果**（与赛马诉求完全错位）
- **文件**：`frontend/src/app/tasks/compare/page.tsx:1-9`（文件头注释）、`:192-210`（调 compare-configs）
- **现象**：页面标题是「配置对比 / 选择 2–5 个已完成回测任务，**对比配置参数差异**」。它 diff 的是 `backtest_config.start_date` / `commission_rate` 这类**入参**，
  **完全不对比输出**：收益、夏普、回撤、换手率。
  用户想做「赛马」→ 进这个页 → 只能看到「这两个任务的开始日期不一样」。
- **复现**：登录 →「任务管理」→「配置对比」→ 选 2 个已完成任务 → 表格里只有参数名 / 值 / 差异高亮
- **改法**：
  1. **在同一个页面上加第二个 Tab：「结果对比」** —— 复用 `GET /api/v1/tasks/{id}/detailed`（已可用，实测返回 `total_return / sharpe_ratio / max_drawdown / total_trades / win_rate / portfolio_history`），把 2~5 个任务的**收益 / 年化 / 夏普 / 最大回撤 / 交易次数 / 胜率 / 换手率**排成一张并排对比表，**按夏普排序**。
  2. 再加一张**多策略净值曲线叠加图**（`portfolio_history` → `EquityCurveChart`，echarts 已就位，6 处 `import * as echarts`）。
  3. 「配置对比」与「结果对比」并列成两个 Tab，各自解决「为什么不一样」和「谁更好」。
  4. 这条与 P0-04 一起构成「多策略赛马」诉求的完整落地路径。
- **截图**：`08-tasks-compare.png`、`18-compare-empty.png`

---

### P2-01　三套图表库并存，其中 recharts 的唯一消费者是死代码
- **文件**：`frontend/src/components/backtest/PortfolioStrategyResults.tsx:14-26`（recharts 唯一 import）
- **现象 / 数据**：

  | 库 | node_modules 体积 | 实际使用文件数 | 状态 |
  |---|---|---|---|
  | echarts 5.4.3 | **43 MB** | 9（全部 `import * as echarts`） | 主力，在用 |
  | recharts 3.6 | 7.7 MB | **1** | `PortfolioStrategyResults.tsx` — **该组件只被 `components/backtest/index.ts` re-export，全项目无任何页面 import 它 → 死代码** |
  | lightweight-charts 4.1 | 3.1 MB | 1 | `TradingViewChart.tsx`，在用（价格走势） |

  另有 **antd 6.5.2（61 MB）** 只在 5 个旧页面用（`register` / `reset-password` / `wallet` / `settings/subscription` / `hooks/useCommerce`），与本审计范围无交集但同属全站包体。
- **改法**：
  1. **删掉 `components/backtest/PortfolioStrategyResults.tsx` 及 `components/backtest/index.ts:13,16` 的 re-export** → 从 `package.json` 移除 `recharts`（-7.7 MB 直接依赖）。**或者**反过来：把它接到 P0-04 的赛马页（用 Pie 展示权重/贡献度），但**前提是统一到 echarts**，不要引入第 4 套视觉语言。
  2. echarts 全部改成**按需引入**（`echarts/core` + `use([LineChart, ...])`），当前 9 处 `import * as echarts` 会把整个 echarts 打进每个 chunk（**43 MB 源码**），是包体膨胀主因。
  3. 明确保留 2 套：echarts（回测/优化统计图）+ lightweight-charts（金融 K 线，交互模型与 echarts 差异大，保留合理）。**目标：3 套 → 2 套。**
  4. 统一视觉规范：echarts 与 lightweight-charts 目前各自定义颜色/字体/网格线，建议抽 `chartTheme.ts` 统一轴色/字体族/网格。
- **截图**：`12-task-detail-charts.png`、`14-equity-curve-canvas.png`

---

### P2-02　`import * as echarts` 全量引入 + 图表容器无高度约束，易出现空白图表
- **文件**：`frontend/src/components/charts/EquityCurveChart.tsx:9`、`:60-140` 等 9 处；`frontend/src/components/charts/InteractiveChartsContainer.tsx:35-39`
- **现象**：9 个文件全部 `import * as echarts from 'echarts'`。另外 echarts 容器 div 未设 `min-height` / 显式高度，初始化时若父容器宽高为 0（常见于 flex/grid + `display:none` 的 tab），echarts 会画出 0×0 的空白 canvas 且**不报错**。
  实测过程中我两次用截图工具都抓到了「空白图表区域」，最终用 `canvas.toDataURL()` 直接导出才确认**权益曲线实际是正常渲染的**（绿色折线 + 面积填充 + 双 Y 轴 ¥万元刻度 + 2025-01-02~2025-06-25 日期轴）—— 也就是说**空白截图 ≠ 空白图表**，但这类"渲染成功与否"在 UI 上无法自证。
- **改法**：
  1. 改按需引入（见 P2-01）。
  2. 给每个 echarts 容器加 `style={{ width: '100%', height: 400 }}` 或 `minHeight`，并在 `useEffect` 里用 `ResizeObserver` 监听容器尺寸变化后 `chart.resize()`（当前 `EquityCurveChart` 只在 `data`/`loading` 变化时重建，**窗口 resize 或 tab 切换后不会 resize**，图表可能被压扁或留白）。
  3. 加"无数据"显式分支：数据为空时渲染 `<Alert severity="info">该区间无净值数据</Alert>`，而不是让用户面对一张空白画布猜测是加载中还是没数据。
- **截图**：`13-equity-curve-blank.png`（误导性空白截图）、`14-equity-curve-canvas.png`（真实渲染结果）

---

### P2-03　任务详情页指标自相矛盾：「成本影响 +23.21%」「无成本年化 0.00%」
- **文件**：`frontend/src/app/tasks/[id]/TaskDetailContent.tsx`（成本分析区）、`frontend/src/services/backtestDataAdapter.ts`
- **现象**：已完成回测的概览页同时显示：
  - 含成本年化收益 **23.21%**、无成本年化收益 **0.00%**、「成本影响 **+23.21%**」
  - 但「含成本指标 / 无成本指标」对照表里又写着「平均收益 0.0000% / 标准差 0.0000% / 信息比率 0.000」
  - 「胜率 0.0%」「盈亏比 0.00」「平均单笔盈亏 +0.00%」
  - 而**总收益率 +10.78%、夏普 2.478**
  
  → 同一页里「无成本收益 0%」和「成本影响 +23.21%」直接矛盾（成本不可能创造收益），且一组「2.478 夏普 / 0% 胜率 / 0 盈亏比」的指标同时出现，**用户无法判断哪组是真数据**。
- **复现**：登录 → 跑一个回测 → 详情页「概览」Tab → 逐段读「风险评估总结 / 交易成本明细」
- **根因**：`portfolio_history[*].portfolio_value_without_cost` 在本例中与 `portfolio_value` 相等（都是 100000 起步、期间无持仓变化），adapter 用 `Math.random()` 或等值分支算出了 0.00%；而 `annualized_return` 走的是另一条路径 → 两套口径不一致。
- **改法**：
  1. `backtestDataAdapter.ts` 的成本对比：只有在**确实存在无成本序列**时才展示「无成本指标」列；否则整块降级为一句「本次回测无成本序列，未做含/无成本对照」。
  2. 「成本影响」符号逻辑修正：成本影响应为**负值**（成本拖累收益），现在显示 `+23.21%` 是把 `含成本 - 无成本` 的减法写反了。
  3. 指标为空时显示「—」而不是 `0.00`（0 会被读成"真实测得为 0"）。
- **截图**：`11-task-detail.png`

---

### P2-04　「策略赛马」页的持仓数与策略规格对不上，无任何说明
- **文件**：`frontend/src/app/paper-portfolio/page.tsx:149-190`（卡片）、`backend/scripts/paper_portfolio.py` 的 `STRATEGIES` 定义
- **现象**：卡片上标着「gbm_voltgt **12 只**」「factor_ranking **49 只**」，但策略规格是 gbm_voltgt `topk=60, ndrop=5`、factor_ranking `topk=50, ndrop=5`。
  - factor_ranking 49/50 → 差 1 只是可接受的（停牌/涨跌停）
  - **gbm_voltgt 12/60 → 差 48 只**，是数量级级别的偏差，UI 完全不解释
  用户看到「12 只持仓」会以为策略只选 12 只股票（低频、集中），实际上是被流动性/可成交性拦掉了 48 只 —— **这直接影响用户对策略风险敞口的判断**，且这与"高频"担忧正好相反地误导人。
- **复现**：登录 →「多策略纸面盘」→ 对比卡片「N 只」与策略描述
- **改法**：
  1. 卡片持仓数改为 `12 / 60` 形式（实际/目标），并在 tooltip 里说明「因涨停/停牌/流动性不足未能建仓 48 只」。
  2. JSON 补充 `blocked_buys` 计数（脚本 `paper_portfolio.py:766` 已有 `blocked_buys` 列表，只是页面没用）。
  3. 补充**单票权重分布**（现在 TopK 缺失导致权重也缺失，用户无法评估集中度风险）。
- **截图**：`24-paper-portfolio-cards.png`

---

### P2-05　「模型管理」页显示 0 个模型，全站 35 个模型对当前用户不可见
- **文件**：`frontend/src/app/models/page.tsx`（列表渲染）
- **现象**：模型管理页显示「模型列表 (0 个) / 暂无模型 / 创建您的第一个预测模型」。而用同一账号访问 `/models` API（带 token）时能看到 **35 个模型**（`CI smoke XGBoost model`、多个 `official-alpha158/360-smoke` 等，含就绪/失败/训练中各状态）。
  → **页面和 API 对不上**。根因与 P0-03 同源：列表接口按 `user_id` 过滤，而模型是全局/其他用户创建的。
- **复现**：登录 →「模型管理」→ 显示 0 个；同页「创建模型」旁的 API 响应含 35 条
- **改法**：
  1. 后端 `/api/v1/models` 列表接口支持 `owner=me|all`，前端提供切换（默认 `all` 对研究平台更合理）。
  2. 若坚持按用户隔离，则页面应显示「您还没有模型」并提供「查看全局模型」入口，而不是让用户以为系统里没有模型。
  3. 顺带：模型卡片上「准确率 0.0%」的失败模型（`official-alpha158-smoke` 等一批 `失败` 状态）**没有失败原因**，与 P1-07 同类问题。
- **截图**：`03-models-list.png`

---

### P2-06　回测任务列表的「策略」列对模型类回测显示「未知策略」
- **文件**：`frontend/src/app/backtest/page.tsx:383-386`
- **现象**：`const strategyName = task.config?.strategy_name || task.config?.strategy_config?.strategy_name || '未知策略';`
  实际 config 里的策略名在 `backtest_config.strategy_name`（嵌套一层），当前代码查的是 `task.config?.strategy_name`（顶层）和 `task.config?.strategy_config?.strategy_name`（少一层）。
  实测在 `model_signal` / `portfolio` 模式下会退化成「未知策略」。
- **改法**：`task.config?.backtest_config?.strategy_name || task.config?.backtest_config?.strategy_config?.strategy_name || task.config?.strategy_name || '未知策略'`。
  另：模型类回测应显示**模型名 + 信号周期**（如 `model_signal · 1d`），而不是「未知策略」——这直接关系到用户判断「gbm_voltgt 标签5天→1天」这类改动是否生效。
- **截图**：`20-backtest-list-with-task.png`

---

### P2-07　`/tasks` 页的「筛选状态」标签重复渲染，且存在不可见的零宽占位符
- **文件**：`frontend/src/app/tasks/page.tsx:421-435`
- **现象**：页面文本里出现 **两遍「筛选状态」**，且表格区域有若干不可见的零宽字符 ``（U+200B）单独成段。这些是 MUI `InputLabel` + `Select` 未正确收缩时的残留（`label="筛选状态"` 与 `InputLabel` 同时存在 → 渲染两份）。
- **复现**：登录 →「任务管理」→ 查看筛选区，DOM 里 `.MuiInputLabel-root` 出现两次
- **改法**：删掉 `Select` 上的 `label="筛选状态"`（`InputLabel` 已足够），或删掉 `InputLabel` 改用 `aria-label`；同时清理 `backtest/page.tsx` 等处的零宽字符。
- **截图**：`15-tasks-list-with-task.png`

---

### P3-01　「参数调优」与「超参优化」是两个功能重叠但互不感知的入口
- **文件**：`frontend/src/app/tasks/tuning/page.tsx`、`frontend/src/app/optimization/page.tsx`
- **现象**：导航里「参数调优」= 基于**已有任务快照**改参数再提交；「超参优化」= Optuna 搜索。两者都叫"调优"，但优化页完全不知道调优台产出的任务，调优台也不能引用优化结果。用户来回试时需要手工搬运 task_id。
- **改法**：调优台选中任务后，提供「用同一区间继续超参优化」按钮，把该任务的 `backtest_config` / `stock_codes` / 日期区间直接灌进优化表单；反之优化完成后提供「把最优参数存为模板 / 用该参数重建回测」。
- **截图**：`07-tasks-tuning.png`、`19-optimization-list.png`

---

### P3-02　「多策略纸面盘」是静态 JSON，与平台后端完全脱节
- **文件**：`frontend/src/app/paper-portfolio/page.tsx:82`（`fetch('/paper-portfolio.json')`）
- **现象**：页面读的是 `frontend/public/paper-portfolio.json`（构建时静态文件），不是后端 API。数据生成时间 `2026-10-27T04:49:17` 写在页脚，但**页面不显示"数据滞后"提示**。若脚本没跑，页面会红字报错让用户去跑一个后端脚本 —— 这把运维操作暴露给了终端用户。
- **改法**：把纸面盘暴露为后端 API（`/api/v1/paper-portfolio`），前端改调 API；或至少在页头加「数据快照于 YYYY-MM-DD HH:MM，滞后 N 天请重跑」并把「如何生成」的报错文案改成「数据暂不可用，请联系管理员」。
- **截图**：`02-paper-portfolio-race.png`

---

### P3-03　任务详情页有 3 处 `catch { // noop }`，导出结果失败也无任何提示
- **文件**：`frontend/src/app/tasks/[id]/useTaskDetailPage.ts:200-202`、`:241-243`、`:274-276`
- **现象**：`handleExport`（导出 CSV）失败时完全静默，用户点了没反应也不知道为什么。`handleTaskCompleted` 的 catch 同样静默 → WebSocket 收到完成事件后若详情拉取失败，页面会一直停在旧状态。
- **改法**：三处都改为 `showSnackbar.showError(...)`（项目已有 `useSnackbarStore`）；导出成功给 success 提示。
- **截图**：`11-task-detail.png`

---

## 2. 先修清单（按 P0/P1 排序，共 12 条）

| # | 严重度 | 问题 | 一句话改法 | 位置 |
|---|---|---|---|---|
| 1 | **P0** | 交易策略下拉框空白，提交时静默用 `moving_average` | 裸 `fetch` → `apiRequest`；失败时禁用提交并报错；校验 `strategy_name` 非空 | `tasks/create/page.tsx:126,91,322-355` |
| 2 | **P0** | 参数调优台基准任务下拉框永远为空（`/api/v1/api/v1/…`） | 去掉 `apiRequest` URL 里的 `/api/v1` 前缀（3 处）+ 补错误提示 | `tasks/tuning/page.tsx:170,267,305` |
| 3 | **P0** | 回测列表的收益/夏普列永远是 `-`（读 `task.result`，API 给的是 `result_summary`） | `getBacktestSummary` 改读 `result_summary`；拆「进度」与「指标」两列 | `backtest/page.tsx:210-222,432-441` |
| 4 | **P0** | 策略赛马页只有收益率，缺夏普/回撤/**换手率**/净值曲线 | 后端输出 `nav_history` 并算 sharpe/maxDD/turnover；前端加赛马汇总表 + 6 线净值叠加图；区分「本期 vs 累计」 | `paper-portfolio/page.tsx:130-190`；`backend/scripts/paper_portfolio.py:1020-1050` |
| 5 | **P1** | 回测表单零口径警示（子集/幸存者/未来函数/样本外） | 加口径 Alert + 「已选 N/5594 只」占比提示 + 费率默认值贴近实盘 | `tasks/create/page.tsx:438+` |
| 6 | **P1** | 模板「回测预览」是硬编码假数据（+32.5%/夏普1.45） | 删除假数据，或改为真实基线缓存结果并带口径脚注 | `templates/page.tsx:53,87,121,284-380` |
| 7 | **P1** | 模板「一键使用」6 个预填参数全部丢失 | `create/page.tsx` 增加读取 `templateId/strategy/startDate/endDate/initialCash/config` | `templates/page.tsx:150-161`；`tasks/create/page.tsx:147-162` |
| 8 | **P1** | 任务详情把任何加载失败都显示成「任务不存在或已被删除」 | 区分 404 / 401 / 429 / 网络错误，四种分别给文案 + 重试 | `useTaskDetailPage.ts:105`；`TaskDetailPageView.tsx:123` |
| 9 | **P1** | 限流 429 打在 token 刷新上 → 用户被强制登出 | refresh catch 区分 429：不清 token、不跳登录，改退避重试；后端改按 user_id 限流 | `services/api.ts:227-256`；`backend/app/main.py:233-238` |
| 10 | **P1** | `/tasks` 加载失败静默显示「0 个任务」 | 加 `loadError` + 错误条 + 重试；失败时统计卡不显示 0 | `tasks/page.tsx:94-105` |
| 11 | **P1** | 失败原因只在详情页可见，列表页完全看不到 | 列表状态列加错误摘要 Tooltip / 「原因」列 | `tasks/page.tsx:484-582`；`backtest/page.tsx:390-479` |
| 12 | **P1** | 优化表单无过拟合/样本外警示，目标指标无换手率 | 加 warning Alert；目标指标加「年换手次数」+ `turnover` 权重；`stability` 加公式说明 | `CreateOptimizationTaskForm.tsx:51-78,620-650` |

> 建议执行顺序：先做 1→2→3（三个 P0 都是「页面空白/数据不显示」的硬故障，互不依赖，可并行），再 4（赛马，独立但工作量最大，需要后端配合出 `nav_history`），然后 5→6→7（口径与可信度，直接影响用户对结果的判断），最后 8→9→10→11→12（错误呈现与过拟合防护）。

---

## 3. 总体评价

**这组「研究工作台」目前不具备研究可用性**：三个核心流程（选策略建回测、参数调优、结果对比）在真实浏览器里**全部走不通** —— 策略下拉框空白、调优台任务列表空、赛马页缺全部关键指标；而已经跑出且后端有完整指标的回测（+10.78% / 夏普 2.478）在列表页却显示成 `-`，用户第一眼看到的是"什么都没跑出来"。更值得警惕的是**可信度问题**：模板页用硬编码的 `+32.5% / 夏普 1.45` 冒充"回测预览"，回测表单和优化表单对子集偏差、幸存者偏差、过拟合**只字不提**，赛马页把 3 个交易日的区间收益和 100 万本金并排显示却拿不出夏普、回撤和换手率 —— 这些恰好是用户最需要用来判断"收益是不是真的"的信息。**好消息是数据侧其实很扎实**：后端已存有 16 策略全市场 3 年基线（含基准线）、`paper_portfolio.py` 已在写 `nav_history`、优化目标指标已有卡玛/IC_IR/成本占比，缺的几乎全是**最后一公里的呈现**；把 P0 四条 + P1 的 5/6/7/12 处理完，这个工作台就能支撑用户真正想要的低频多策略赛马，而不是现在这样"页面能开、数据看不见"。

---

## 附录 A：截图索引

| 文件 | 内容 |
|---|---|
| `01-tasks-create-form.png` | 创建任务页首屏（首次访问，策略列表加载失败态） |
| `02-paper-portfolio-race.png` | 多策略纸面盘（策略赛马）—— 6 张卡 + 调仓明细，含 `TopK= / NaN%` 缺陷 |
| `03-models-list.png` | 模型管理页 —— 显示「0 个模型」（API 实有 35 个） |
| `04-templates.png` | 策略模板页 —— 硬编码「回测预览」假指标 |
| `05-optimization.png` | 超参优化任务列表（空） |
| `06-optimization-create.png` | 超参优化创建表单（目标指标/试验次数区） |
| `07-tasks-tuning.png` | 参数调优台 —— 停在第 ① 步，下拉框空 |
| `08-tasks-compare.png` | 配置对比页 —— 只 diff 入参，无结果对比 |
| `09-backtest-create-form.png` | 回测配置区（无任何口径提示） |
| `10-strategy-dropdown-empty.png` | **P0-01 证据**：交易策略下拉框空白 |
| `11-task-detail.png` | 任务详情概览（正常态）—— 含「成本影响 +23.21% / 无成本 0.00%」矛盾 |
| `12-task-detail-charts.png` | 交互式图表 Tab |
| `13-equity-curve-blank.png` | 截图工具抓到的"空白"帧（**误导性**，非真实渲染） |
| `14-equity-curve-canvas.png` | 权益曲线**真实**渲染（canvas.toDataURL 导出） |
| `15-tasks-list-empty.png` | /tasks 空列表（加载失败静默） |
| `15-tasks-list-with-task.png` | /tasks 有任务（失败任务无原因） |
| `16-backtest-list-empty.png` | /backtest 限流 429 错误态 |
| `17-tuning-double-prefix.png` | **P0-02 证据**：调优台停在第 ① 步 |
| `18-compare-empty.png` | 配置对比页（加载失败） |
| `19-optimization-list.png` | 优化任务列表（空） |
| `20-backtest-list-with-task.png` | **P0-03 证据**：已完成任务的收益/夏普列 = `-` |
| `21-tasks-list-failed.png` | 失败任务在列表里只有「失败」chip，无原因 |
| `22-backtest-list-failed.png` | 回测列表（限流态） |
| `23-task-detail-failed.png` | 失败任务详情（错误信息完整） |
| `24-paper-portfolio-cards.png` | 赛马卡片特写 —— 只有收益，无夏普/回撤/换手 |
| `25-position-analysis.png` | 持仓分析 Tab（限流下未渲染） |
| `26-task-detail-false-notfound.png` | **P1-04 证据**：有效任务被显示为「任务不存在或已被删除」 |
| `27-optimization-objective.png` | 优化目标下拉 12 项 —— 无换手率指标 |
| `28-templates-oneclick.png` | **P1-03 证据**：一键使用后表单全空 |

## 附录 B：关键数据交叉验证记录

| 验证项 | 方法 | 结果 |
|---|---|---|
| 策略下拉空白根因 | 对比 `fetch`（无 token）与 `apiRequest`（带 token） | 无 token → `401 缺少 Authorization 请求头`；带 token → 29 个策略 |
| 调优台空列表根因 | 读 `performance.getEntriesByType('resource')` | `/api/v1/api/v1/tasks?limit=200&status=completed`（双前缀） |
| 列表指标字段不匹配 | 对比 API 响应键与前端读取路径 | API: `result_summary{...}`；前端读 `task.result`/`task.results`（列表响应中均不存在） |
| 16 策略基线数据存在 | `sqlite3 backend/data/app.db` 取 `baseline_baseline_full_market_3y` | `result.strategies` 16 条 + `result.benchmark`，字段齐全，前端无任何消费方 |
| `nav_history` 存在但未输出 | 读 `paper_portfolio.py:709-720` 与实际 JSON | 脚本写 `nav_history`；JSON 的 6 条线**全无**该键 |
| 纸面盘 TopK 全缺 | 逐字段检查 JSON | 6/6 缺 `topk`/`ndrop`/`target_weight`；`meta.lines_updated = []`（全走 rebuild 路径） |
| 模板策略名不存在 | 对比后端 29 个策略 key | `ma_cross_strategy` ❌ / `momentum_strategy` ❌ / `multi_factor` ✅ |
| recharts 是死代码 | `grep -rn PortfolioStrategyResults src/` | 仅 `components/backtest/index.ts` re-export，无页面 import |
| 限流触发登出 | 观察 429 后 `localStorage` 与跳转 | token 被清空并跳 `/login` |
| 回测详情误报"不存在" | 有效任务 + 过期 token | 页面显示「任务不存在或已被删除」，刷新后正常 |

## 附录 C：未覆盖 / 建议后续补做

- **真·多策略赛马页**（P0-04 的实现）需要后端配合输出 `nav_history` 与派生指标，属于**改代码**范围，本次只读审计未动源码。
- `mcp__stock_platform__run_backtest` 跑 3 年全市场 6000+ 笔的长回测（用于验证"高频策略在 UI 上如何呈现"）本次未执行 —— 现有 `audit_smoke_rsi`（11 只 / 7 笔 / 2 秒）已足够暴露列表与图表的缺陷，但**"高频策略的收益是否被 UI 美化"这一条（P1-04 关联）需要在赛马页上线后，用 `rsi_71421`（876 笔）与 `gbm_voltgt`（标签 5天→1天）实跑复核**。
- 移动端（<768px）布局只验证了首屏未做逐页断点检查。
- 未审计：`src/app/predictions`、`src/app/signals`、`src/app/paper-signal`、`src/app/data`、`src/app/monitoring`（不在本次范围）。
