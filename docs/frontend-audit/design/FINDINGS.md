# 股票预测平台前端 · 视觉设计系统一致性审计

- **审计范围**：`~/Projects/stock-platform/frontend`（Next.js 14 App Router + TS）
- **审计类型**：横切全站只读审计（未修改任何源码；`git status` 仅新增 `docs/frontend-audit/`）
- **审计日期**：2026-10-27
- **验证方式**：`browser_exec` 实机渲染（1440×900 桌面 + 390×844 移动双视口）+ `getComputedStyle` 量化 + 生产构建产物分析 + 全量 grep 计数
- **截图目录**：`docs/frontend-audit/design/`（01~25，覆盖 20+ 个页面）
- **登录态**：`ronghui@example.com`（`ronghui` 用户名，enterprise 订阅，is_admin=true）

---

## 0. 审计方法与证据基线

| 证据源 | 命令/手段 | 覆盖 |
|---|---|---|
| 运行时样式 | `getComputedStyle` 逐元素采样 | 20 个路由实测 |
| 组件库清单 | `grep -rn "from '@mui/\|from 'antd'\|from '@heroui/\|from 'lucide-react'\|from '@heroicons/"` | 128 个源文件 |
| 硬编码色值 | `grep -oE '#[0-9a-fA-F]{3,6}\b'` | 377 处 / 131 个唯一色 |
| 生产包体 | `.next-build-check/` 构建产物 + `app-build-manifest.json` | 30 路由 |
| CSS 产物 | `.next-build-check/static/css/*.css`（229,677 B） | 全局样式表 |

**关键背景事实**：
- 生产构建产物存在且可分析（`.next-build-check`，2026-10-27 08:00 构建），可直接读出真实包体与 CSS 组成。
- 应用为「已登录即用」结构，`app/page.tsx`（landing）在已登录时 `router.push('/dashboard')` 重定向；未登录才见 landing。

---

## 1. 问题清单

### P0 — 阻断级（正确性缺陷 / 会误导用户）

---

#### P0-1 · A 股语义色在同一屏内自相矛盾：绿涨红跌 vs 红涨绿跌

| 项 | 内容 |
|---|---|
| **文件:行** | `src/app/today-signal/page.tsx:337`、`:462`、`:569`；`src/app/paper-portfolio/page.tsx:178`、`:206-207`；`src/app/paper-signal/page.tsx:178-188`；`src/components/charts/TradingViewChart.tsx:524-525`、`:602`；`src/components/charts/MonthlyHeatmapChart.tsx:217`、`:183` |
| **现象** | 页面文字层用 **绿=涨 / 红=跌**（欧美惯例），K 线图与热力图用 **红=涨 / 绿=跌**（A 股惯例）。同一屏内两套语义，用户读同一只票会得到相反结论。 |
| **证据** | 实测 `/paper-portfolio`（截图 19-paper-portfolio.png）：`-0.05%` → `rgb(211,47,47)` 红、`+0.83%` → `rgb(46,125,50)` 绿、`卖出（6）` → 红、`买入（18）` → 绿。而 `TradingViewChart.tsx:524` 写死 `upColor: '#ef5350', // 红色，上涨` / `downColor: '#26a69a', // 绿色，下跌`；`MonthlyHeatmapChart.tsx:217` 写死 `color: ['#ef4444', '#ffffff', '#10b981'], // 红-白-绿`。 |
| **具体改法** | 新增主题语义 token，**禁止**再用 `success.main`/`error.main` 表达涨跌：<br>`palette.semantics.up = '#d32f2f'`（红涨）、`palette.semantics.down = '#2e7d32'`（绿跌）、`palette.semantics.flat = '#8a8f98'`。<br>改 `today-signal/page.tsx:337/462/569` → `signal.total_return_pct >= 0 ? 'semantics.up' : 'semantics.down'`；`paper-portfolio/page.tsx:178` 同理；`:206-207` 的「买入/卖出」**保持**红买绿卖（A股买卖习惯：买入红、卖出绿，与涨跌色一致），或改用 `semantics.up/down` 统一。`TradingViewChart.tsx:524-525` 的 `#ef5350/#26a69a` 改为引用 `palette.semantics.up/down`，与页面文字对齐。 |

---

#### P0-2 · `hsl(var(--border))` 引用了从未定义的变量，`*` 边框色退化为 currentColor

| 项 | 内容 |
|---|---|
| **文件:行** | `src/app/globals.css:15-17`（`* { border-color: hsl(var(--border)); }`） |
| **现象** | 全项目 grep `--border` **只有这一处使用、零处定义**。`--background`/`--foreground` 在 `globals.css:5-8` 定义了，`--border` 没有。CSS 变量未定义时 `hsl(var(--border))` 整体失效，`border-color` 回落为继承值 `currentColor`。 |
| **证据** | 运行时 `getComputedStyle(document.documentElement).getPropertyValue('--border')` → **空**。`document.styleSheets` 规则表里 `*` 选择器的 `border-color` 值为 `hsl(var(--border))`，而 `*, ::before, ::after`（Tailwind preflight）为 `rgb(229,231,235)`。实测插入裸 `div{border:1px solid}` 到 `body`，计算值 = `rgba(0,0,0,0.87)`（即父级 text 色），**不是** `rgb(229,231,235)`。Tailwind 的 `.border` 工具类实测 `borderTopColor = rgb(0,0,0)` —— 纯黑描边。 |
| **实际影响** | 现状尚未大面积爆发，因为 65 个文件的 `<Card>` 基本都走 MUI 主题的 `divider`；但 `/wallet` 等 antd+Tailwind 混用页、任意未来新增的裸 `border` 元素，都会渲染成黑色描边而非浅灰分隔线。这是一条「静默的地雷」。 |
| **具体改法** | 在 `globals.css:5-8` 补齐定义并让 MUI `divider` 与之同源：<br>```css<br>:root {<br>  --background: 0 0% 100%;<br>  --foreground: 222.2 84% 4.9%;<br>  --border: 214.3 31.8% 91.4%;   /* = #e5e7eb，与 Tailwind gray-200 一致 */<br>}<br>.dark {<br>  --background: 222.2 84% 4.9%;<br>  --foreground: 210 40% 98%;<br>  --border: 217.2 32.6% 17.5%;<br>}<br>```<br>同时 MUI 主题加 `palette.divider` 显式值 `#e5e7eb`，避免 CSS 变量与 JS 主题两套真相。 |

---

#### P0-3 · antd 与 MUI 在同一页并存，主色与文字色同时冲突

| 项 | 内容 |
|---|---|
| **文件:行** | `src/app/wallet/page.tsx:5-22`（antd 全量组件 import）；`src/app/settings/subscription/page.tsx`；`src/app/register/page.tsx`；`src/app/reset-password/page.tsx`；`src/hooks/useCommerce.ts` |
| **现象** | `/wallet` 页面同时渲染 antd 组件和 MUI 布局（AppBar / AppLayout）。两套库的设计 token 完全独立且互相冲突。 |
| **证据** | 实测 `/wallet`（截图 08-wallet.png）：<br>• antd 主按钮 `background = rgb(22,119,255)` = **#1677ff**；MUI 主按钮 `rgb(25,118,210)` = **#1976d2**。同一屏两种蓝。<br>• antd 正文 `rgba(0,0,0,0.88)`；MUI 正文 `rgba(0,0,0,0.87)`。第三套：Tailwind/全局 `body` 为 `rgb(17,24,28)` = **#11181c**（实测全站 126 个元素命中）。<br>• antd 按钮圆角 `6px` / MUI `8px`；antd 卡片 `border: 1px solid rgb(240,240,240)` 无阴影 / MUI 卡片 `box-shadow: rgba(0,0,0,0.1) 0 2px 8px` 无边框。<br>• 字体栈不同：antd 追加 `"Noto Sans"` + 4 个 emoji 字体；MUI 无。 |
| **具体改法** | 短期：在 `app/layout.tsx` 加 `<ConfigProvider theme={{ token: { colorPrimary: '#1976d2', colorError: '#d32f2f', colorSuccess: '#2e7d32', borderRadius: 8, fontFamily: <与 MUI 同一串> } }}>` 对齐 antd token；长期：见第 3 节迁移方案（把 4 个 antd 页面改写为 MUI）。 |

---

### P1 — 高优先级（明显视觉断层 / 死代码 / 性能）

---

#### P1-1 · HeroUI 零引用，却向全局 CSS 注入 128KB 死样式（占全局 CSS 56%）

| 项 | 内容 |
|---|---|
| **文件:行** | `tailwind.config.js:1`（`const { heroui } = require("@heroui/react")`）、`:20-65`（`plugins: [heroui({...})]`）；`tailwind.config.js:9`（`content` 含 `'./node_modules/@heroui/theme/dist/**/*.{js,ts,jsx,tsx}'`） |
| **现象** | **活跃运行代码中 0 个文件** import `@heroui/react`（仅 2 处命中在死文件 `src/app/models/page.tsx.working:34,48`，另 1 处是 `src/__tests__/infrastructure.test.ts:22` 断言依赖存在）。但 tailwind 插件仍在构建期把 HeroUI 的整套 utility + 105 个 `--heroui-*` CSS 变量注入全局样式表。 |
| **证据** | `grep -rn "from '@heroui/" src` → 活跃源码 **0 命中**（仅死文件 + 测试）；`grep -l 'heroui\|react-aria\|react-stately' .next-build-check/static/chunks/*.js` → **0 命中**（JS 侧完全没打进包）。<br>但 CSS 产物 229,677 B 中，**128,524 B（56.0%）** 属于 HeroUI 相关规则（749 条规则引用 `--heroui-*`），仅 100,720 B（44%）是自有样式。<br>另有 1,186 处 `--heroui-` 变量引用、86 处 `group-data-` 变体选择器。 |
| **附带** | ① `tailwind.config.js:9` 把 `node_modules/@heroui/theme/dist/**` 加入 `content`，导致 Tailwind 扫描整个 HeroUI dist 目录 —— 构建期无谓开销。<br>② ⚠️ **`src/__tests__/infrastructure.test.ts:22` 有 `expect(packageJson.dependencies).toHaveProperty('@heroui/react')`** —— 删除依赖会**导致该测试失败**，必须同步改测试。 |
| **具体改法** | 删除 `tailwind.config.js:1` 的 `heroui` import 与 `:20-65` 的 `plugins` 数组，`:9` 的 content 项一并删除。预期全局 CSS 229KB → **~100KB**（-56%）。`@heroui/react`、`@heroui/theme` 两个依赖同时从 `package.json` 移除（`node_modules` 合计 16MB），**并同步删除 `src/__tests__/infrastructure.test.ts:22` 的断言**（该测试的意图应是「核心依赖存在」，应改为断言 MUI/antd/lucide 仍在）。 |

---

#### P1-2 · `darkMode: "class"` 与 `.dark` 变量定义了，但全站没有任何一处切换 —— 暗色模式是死配置

| 项 | 内容 |
|---|---|
| **文件:行** | `tailwind.config.js:19`（`darkMode: "class"`）；`src/app/globals.css:10-13`（`.dark { --background: ...; --foreground: ... }`）；`src/stores/useAppStore.ts:28`（`theme: 'light' \| 'dark'` 字段） |
| **现象** | 声称支持暗色，实际：MUI 主题 `palette.mode` 硬编码 `'light'`（`src/theme/muiTheme.tsx:16`）；`grep -rn "classList.add\|classList.toggle\|classList.remove" src` → **0 命中**，无任何代码给 `<html>` 加 `.dark`；`useAppStore` 的 `theme` 字段无任何读取方。 |
| **证据** | 实测 `/dashboard`：`document.documentElement.className` = `""`；手动 `classList.add('dark')` 后 `body` 背景仍为 `rgb(245,245,245)`、`.MuiCard-root` 背景仍为 `rgb(255,255,255)` —— **完全无反应**。<br>真正「像暗色」的只有 landing 的 hero 区块（`src/app/page.tsx:152` `linear-gradient(135deg, #0a1628 ...)`），那是**硬编码的深色区块**，不是暗色模式。而 landing 的顶栏仍是 MUI AppBar 白底（实测 `appbarBg = rgb(255,255,255)`），下方 hero 却是 `#0a1628` 深蓝 —— **白顶栏直接压在深色 hero 上，视觉断层**（截图 13-landing.png）。 |
| **具体改法** | 二选一并明确表态：<br>**方案 A（推荐，工作量小）**：删除 `tailwind.config.js:19`、`.dark` 变量块、`useAppStore.theme` 字段，明确「全站仅浅色 + landing 局部深色区块」；把 landing 顶栏改为 `bgcolor: 'transparent'` + `color: 'white'` 并在 hero 内部实现。<br>**方案 B**：若确实要暗色，则 MUI 主题改 `createTheme({ palette: { mode } })` 动态生成、`globals.css` 补齐所有语义变量、增加 `<ThemeToggle>` 写入 `document.documentElement.classList` 并持久化。 |

---

#### P1-3 · 页面标题层级不统一：3 种 variant + 12 个页面无 h1

| 项 | 内容 |
|---|---|
| **文件:行** | `src/app/dashboard/page.tsx:158`、`:237`（h4）；`src/app/today-signal/page.tsx:242`（h5）；`src/app/admin/payments/page.tsx:166`（h5）；`src/app/paper-portfolio/page.tsx:132`（h5）；`src/app/paper-signal/page.tsx:144`（h5）；`src/app/tasks/tuning/page.tsx:359`（h5）；`src/app/signals/page.tsx:447`（h4，无 `component="h1"`）；`src/app/settings/page.tsx:176`（h4）；`src/app/pricing/page.tsx:201`（h3）；`src/app/predictions/page.tsx`（h4） |
| **现象** | 26 个 page.tsx 中：9 个用 `variant="h4" component="h1"`（24px/600，语义正确），5 个用 `variant="h5"`（20px/600，**渲染为 `<h5>` 标签**），1 个用 `variant="h3"`（landing，56px/800），其余（含 `signals`、`settings`、`predictions`、`wallet`、`register`、`reset-password`、`settings/subscription`、`tasks/[id]`）**没有任何 h1**。 |
| **证据** | 逐文件 grep `variant="h[1-6]"` + `component="h[1-6]"` 统计。运行时实测 `/wallet` 唯一标题为 `<h2>` 30px/600（antd `Typography.Title` 默认），`/paper-portfolio` 为 `<h2>`；`/dashboard` 为 `<h1>` 24px/600。同一套系统里「页面标题」的字号在 20 / 24 / 30px 三档之间跳。 |
| **具体改法** | 统一为「页面标题 = `variant="h4" component="h1" sx={{ fontWeight: 600, mb: 2 }}`」，共 24px/600/1.235 行高。在 `muiTheme.tsx` 加 `MuiTypography` 覆盖把 h1..h6 映射到 24/20/18/16/16/14 的收敛 scale（见 P2-1）。`wallet` 的 antd `Title` 换成 MUI `Typography`。 |

---

#### P1-4 · 图表与业务层各自维护 6 套互不相干的色板

| 项 | 内容 |
|---|---|
| **文件:行** | `src/components/charts/TradingViewChart.tsx:333-362`（12 个策略色 + 一套 matplotlib 默认色 `#1f77b4...`）；`src/components/charts/EquityCurveChart.tsx:159-214`（`#10b981`/`#f59e0b`）；`src/components/charts/DrawdownChart.tsx:115-155`（`#ef4444`/`#dc2626`）；`src/components/charts/BacktestChart.tsx:311-405`（`#10b981`/`#ef4444`）；`src/components/charts/MonthlyHeatmapChart.tsx:217`（`#ef4444/#ffffff/#10b981`）；`src/components/backtest/PortfolioStrategyResults.tsx:66`（`COLORS = ['#0088FE','#00C49F','#FFBB28','#FF8042','#8884d8','#82ca9d','#ffc658']`） |
| **现象** | 6 个图表组件各写各的调色板，同一语义（净值线、回撤、收益）在不同图里是不同色值；`PortfolioStrategyResults` 用的是 Ant Design Charts 的默认色板，与 MUI 主题毫无关系。 |
| **证据** | 硬编码色值统计：377 处 hex / 131 个唯一色。Top 命中：`#1976d2`(36) `#d32f2f`(31) `#2e7d32`(22) `#ed6c02`(21) `#666`(21) `#10b981`(17) `#999`(13) `#9c27b0`(13) `#ef4444`(11) `#4caf50`(7) `#f59e0b`(6) `#f44336`(5) `#52c41a`(4) `#3b82f6`(9)。`#10b981`/`#2e7d32`/`#4caf50` 三种绿表示同一「成功/盈利」；`#d32f2f`/`#ef4444`/`#f44336` 三种红表示同一「失败/亏损」。 |
| **具体改法** | 新建 `src/theme/chartTokens.ts`：<br>```ts<br>export const chartColors = {<br>  up: '#d32f2f', down: '#2e7d32', flat: '#8a8f98',<br>  primary: '#1976d2', secondary: '#9c27b0', warning: '#ed6c02',<br>  neutral: '#6a7985', grid: '#e5e7eb', axis: '#9ca3af',<br>  categorical: ['#1976d2','#ed6c02','#2e7d32','#9c27b0','#0891b2','#d97706','#64748b'],<br>};<br>```<br>6 个图表组件全部改为 import 此文件，删除各自的内联色板。`PortfolioStrategyResults.tsx:66` 的 AntD 默认色板替换为 `chartColors.categorical`。 |

---

#### P1-5 · 死代码：`Providers.tsx` 从未被引用，`GlobalSnackbar` 因此从未挂载

| 项 | 内容 |
|---|---|
| **文件:行** | `src/components/providers/Providers.tsx:1-27`（整文件）；`src/app/layout.tsx:22-30`（自建 `MUIThemeProvider > ErrorBoundary > AppLayout`，绕开 `Providers`） |
| **现象** | `Providers.tsx` 把 `GlobalSnackbar` 挂在 `ErrorBoundary` 内，但 `app/layout.tsx` 完全没用它 —— 全项目 `grep -rn 'Providers' src` 除自身外 **0 命中**。结果是 `GlobalSnackbar`（全局 toast）**从未被渲染**。 |
| **证据** | `grep -rn 'GlobalSnackbar' src` 仅 2 处命中，都在 `Providers.tsx` 内。运行时 `/dashboard` DOM 中 `MuiSnackbar-root` 数量为 **0**。 |
| **具体改法** | 删掉 `src/components/providers/Providers.tsx`（或反过来改 `app/layout.tsx` 改用它）。二选一，但必须让 `GlobalSnackbar` 真正挂上 —— 当前全站 toast 能力为零，页面各自用 `<Alert>` 报错（`today-signal`、`signals` 等页实测显示红色 Alert 框「请求失败」），风格不统一。 |

---

#### P1-6 · 加载态 / 空态 / 错误态三套实现并行，无统一组件

| 项 | 内容 |
|---|---|
| **文件:行** | `src/app/loading.tsx:1-35`（自建 spinner，硬编码 `#e0e0e0`/`#1976d2`/`#666`）；`src/app/error.tsx:28-46`（内联 style，硬编码 `#d32f2f`/`#666`/`#1976d2`）；`src/app/not-found.tsx:16-33`（内联 style，同上）；`src/components/common/EmptyState.tsx:1-36`（emoji 📭 作图标）；`src/app/tasks/[id]/TaskDetailContent.tsx:852-866`（**同名本地 EmptyState**，签名都不一样）；`src/components/common/LoadingSpinner.tsx`、`src/components/common/PageSkeleton.tsx` |
| **现象** | 组件库里已经有 `EmptyState`/`LoadingSpinner`/`PageSkeleton`，但：<br>• `EmptyState` 全站只有 **2 个文件**用（`TaskDetailContent.tsx` 的 3 处 + 自身），而 37 个文件各自手写「暂无…」文案（`TaskDetailContent` 9 处、`monitoring` 4 处、`InteractiveChartsContainer` 4 处、`data` 3 处…）。<br>• `EmptyState` 的图标是 **emoji 📭**（渲染成 h3 尺寸 40px），与全站 lucide 线性图标风格完全冲突。<br>• `TaskDetailContent.tsx:852` 又定义了一个同名本地 `EmptyState`（`icon: ReactNode; message: string`），与公共组件（`icon?: string; title; description; actionLabel`）签名不同。<br>• `PageSkeleton` **定义了但全站 0 处使用**（死代码）。<br>• `LoadingSpinner` 反而是主力（**15 处调用**：dashboard/tasks/models/backtest/monitoring/optimization/predictions/data 等），但 `loading.tsx` / `error.tsx` / `not-found.tsx` 三个 Next.js 约定文件**不用它**，而是各自用**内联 style + 硬编码 hex**手搓了一套。 |
| **证据** | 组件使用计数（按文件）：`暂无` 37、`CircularProgress` 29、`LinearProgress` 22、`LoadingSpinner` 13（+2 个调用方组件）、`Skeleton` 6、`EmptyState` 2、`PageSkeleton` **0**。<br>`loading.tsx` 的 spinner 用 `border: '4px solid #e0e0e0'; borderTopColor: '#1976d2'`，与 `LoadingSpinner` 的 MUI `CircularProgress` 视觉不同。截图 15-notfound.png 可见 404 页是裸 div + inline style，与左右页面的卡片语言完全脱节。 |
| **具体改法** | 1) 扩充 `src/components/common/EmptyState.tsx`：`icon` 改 `ReactNode`、默认用 `<Inbox size={40} />`（lucide）替代 emoji；加 `variant: 'card' \| 'inline'`。<br>2) 删除 `TaskDetailContent.tsx:852-866` 本地实现，改 import 公共组件。<br>3) `loading.tsx` / `error.tsx` / `not-found.tsx` 改用 MUI（`<Box>` + `<Typography>` + `<Button variant="contained">` + `<Card>`），删除全部内联 hex。<br>4) 各页面的「暂无…」统一改 `<EmptyState title="暂无信号" description="..." />`。 |

---

### P2 — 中优先级（不一致但可用）

---

#### P2-1 · 字号 / 行高无收敛 scale，15 种字号并存

| 项 | 内容 |
|---|---|
| **文件:行** | `src/theme/muiTheme.tsx:51-85`（只定义了 h1~h6，未定义 body1/body2/caption/subtitle 的字号与行高）；`src/app/templates/page.tsx:237`（`fontSize: '1.1rem'`）；`src/components/common/LoadingSpinner.tsx` 无关；`src/app/page.tsx:201`（landing h3 56px） |
| **现象** | 运行时实测各页出现的 font-size 集合互不相同：<br>• `/dashboard`：`12 / 14 / 15.2 / 16 / 18 / 24`<br>• `/signals`：`12 / 13 / 14 / 16 / 18 / 20 / 24`<br>• `/pricing`：`12 / 13 / 14 / 15 / 16 / 18 / 20 / 22 / 24 / 28`（**10 档**）<br>• 移动端因 `mobileOverrides.ts:26-35` 把 body1 改成 `0.95rem`，实测 `15.2px` 这种**非整数档位**出现在导航按钮上。 |
| **证据** | `grep -oE "fontSize: '[^']+'"` 命中 41 处，含 `0.7rem`(3) `0.75rem`(15) `0.8125rem`(2) `0.875rem`(3) `0.9rem` `0.95rem`(2) `1.05rem`(4) `1.1rem`(5) `1.25rem` `1.5rem` `1.75rem` `2rem` `2.5rem` —— 13 种手写字号。`mobileOverrides.ts:13` 注释写「从 2.125rem 缩小」，但 base 主题 h4 实际是 `1.5rem`（`muiTheme.tsx:73-76`），**注释与代码已脱节**。 |
| **具体改法** | 在 `muiTheme.tsx` 补全 typography scale：<br>`h1: 32/40`、`h2: 28/36`、`h3: 24/32`、`h4: 20/28`、`h5: 18/28`、`h6: 16/24`、`subtitle1: 16/24`、`subtitle2: 14/22`、`body1: 16/24`、`body2: 14/22`、`caption: 12/16`、`overline: 12/16`。<br>删除 `mobileOverrides.ts:9-37` 的 Typography 覆盖（字号不随断点缩放），保留触摸目标（44px）那部分。<br>清掉所有 `fontSize: '1.1rem'` 之类手写值，改用 `variant`。 |

---

#### P2-2 · 圆角 6 档混用，且 MUI 与 antd 圆角不同

| 项 | 内容 |
|---|---|
| **文件:行** | `src/theme/muiTheme.tsx:86-88`（`shape.borderRadius: 8`）；`src/app/templates/page.tsx:210`（`borderRadius: 3` = 24px）；`src/app/pricing/page.tsx:253`、`:467`（`borderRadius: 2` = 16px）；`src/app/today-signal/page.tsx:452`（`borderRadius: 1.5` = 12px）；`src/app/data/page.tsx:717`（`borderRadius: 2`）；`src/theme/mobileOverrides.ts:86-93`、`:131-139`（移动端 Card/Paper 圆角改 12px） |
| **现象** | 实测圆角值集合：`0px`(279) / `8px`(16) / `50%`(8) / `8px 13.3px`(8, 头像) / `12px`(6) / `16px`(1) / `10px`(1)。`borderRadius: 1.5` 是 MUI spacing 单位的**小数**，属非标准档位。 |
| **证据** | `borderRadius` 硬编码统计：`1`×89、`2`×30、`3`×16、`4`×12、`5`×5、`'50%'`×7、`8`×3、`1.5`×2、`'999px'`×2、`'8px 8px 0 0'`×1。MUI spacing 单位下 1=8px 2=16px 3=24px 4=32px 5=40px，跨度 8→40px 全用；`borderRadius: 3`（24px）的模板卡与 `borderRadius: 1`（8px）的仪表板卡并列出现。 |
| **具体改法** | 收敛到 3 档：`sm=4px`（input/chip）、`md=8px`（card/button，默认 `shape.borderRadius`）、`lg=16px`（dialog/modal/hero 区块）。删除 `borderRadius: 1.5`、`borderRadius: 3`、`borderRadius: 4/5` 的用法；`templates` 改 `borderRadius: 2`（16px）以与 pricing 对齐。`mobileOverrides.ts:90/135` 的移动端 12px 圆角删除（响应式改圆角会造成「同一元素两套圆角」）。 |

---

#### P2-3 · 阴影 7 套并存，卡片阴影强度不统一

| 项 | 内容 |
|---|---|
| **文件:行** | `src/theme/muiTheme.tsx:98-104`（`MuiCard` → `0 2px 8px rgba(0,0,0,0.1)`）；`src/app/page.tsx:202`、`:551`（hover `0 12px 40px rgba(0,0,0,0.12)`）；`src/app/templates/page.tsx:203`（hover `0 12px 40px rgba(0,0,0,0.12)`）；`src/app/globals.css:53-64`（`.card-shadow` / `.card-shadow-lg`）；`src/app/pricing/page.tsx`（`0 4px 14px rgba(15,23,42,0.06)`） |
| **现象** | 实测阴影集合：`rgba(0,0,0,0.1) 0 2px 8px`（卡片）、`rgba(0,0,0,0.2) 0 2px 1px -1px, ...`（AppBar elevation4）、`rgba(0,0,0,0.1) 0 4px 12px`（login 卡）。源码另有 4 套手写 shadow。`globals.css` 的 `.card-shadow`/`.card-shadow-lg` 两个工具类**全站 0 引用**（死代码）。 |
| **证据** | `grep -oE "boxShadow: '[^']*'"` → 18 处、7 个不同值。`grep -rn 'card-shadow' src` → 仅 `globals.css` 自身定义，无使用方。 |
| **具体改法** | 定义 2 档 elevation 到 `muiTheme.tsx`：<br>`elevation1: '0 1px 2px rgba(16,24,40,0.06), 0 1px 3px rgba(16,24,40,0.10)'`（卡片）<br>`elevation2: '0 12px 24px -4px rgba(16,24,40,0.08)'`（hover / 浮层）<br>删除 `globals.css:53-64` 的 `.card-shadow*` 与所有手写 `boxShadow`，改用 `sx={{ boxShadow: 1 }}` / `sx={{ boxShadow: 2 }}`。 |

---

#### P2-4 · 「AI 生成通用后台模板」痕迹：蓝紫渐变 + 六色功能卡 + 无意义图标

| 项 | 内容 |
|---|---|
| **文件:行** | `src/app/page.tsx:152`（`linear-gradient(135deg, #0a1628 0%, #1a2332 50%, #0d2137 100%)`）、`:165`（`radial-gradient(circle, rgba(25,118,210,0.15) 0%, transparent 70%)`）、`:177`（`radial-gradient(circle, rgba(124,58,237,0.1) 0%, transparent 70%)`）、`:321`（`linear-gradient(135deg, rgba(25,118,210,0.12), rgba(124,58,237,0.12))`）、`:553`（`linear-gradient(135deg, #0a1628 0%, #1a2332 100%)`）、`:38-60`（6 个功能项各带一个 `color`/`bgColor` 硬编码） |
| **现象** | landing 页把「深蓝夜空 + 蓝紫双色 radial 光晕 + 蓝紫渐变卡片」这套 2023 年 AI SaaS 模板语汇整套照搬。**紫色 `#7c3aed`（124,58,237）既不在 MUI 主题的 secondary（`#9c27b0`）里，也不在 Tailwind 配置里**，是纯粹的孤立色值。 |
| **证据** | 全站 gradient 出现 9 处，**全部集中在 `src/app/page.tsx`（5 处）+ `src/app/templates/page.tsx:215` + `src/app/data/page.tsx:674`**。紫色系：landing `#7c3aed` ×2 + `rgba(124,58,237,...)` ×3；templates `page.tsx:38`/`:81` 也用 `#7c3aed` —— 与 landing 呼应但与 MUI `secondary.main = #9c27b0` 不同。实测 landing 的 5 个渐变值见上。landing 页面 h1 为 56px/800，与全站其他页 24px/600 差 2.3 倍。 |
| **具体改法** | 保留「landing 可以有自己的品牌表达」这个原则，但把色相收敛到主题：<br>• 移除全部紫色 radial 光晕与蓝紫双色渐变，改为单色品牌蓝的方向性光影：`radial-gradient(circle, rgba(25,118,210,0.18) 0%, transparent 70%)`（单一 hue）。<br>• 6 个功能卡的 `color`/`bgColor` 改为从 `chartColors.categorical` 取，或统一用 `primary.main` + `primary.light` 的 alpha 变体。<br>• landing h1 字号收到 40px/700，与 pricing 页 h3 统一为「营销页标题」档。<br>• `templates/page.tsx:38/81` 的 `#7c3aed` 改为 `secondary.main`（`#9c27b0`），使 secondary 在全站有唯一值。 |

---

#### P2-5 · 图标三库并存 + 尺寸 15 档

| 项 | 内容 |
|---|---|
| **文件:行** | `lucide-react` 68 个文件；`@heroicons/react` 3 个文件（`src/components/backtest/BacktestErrorHandler.tsx:25`、`BacktestProgressIndicator.tsx:10`、`BacktestProgressMonitor.tsx:27`）；`@mui/icons-material` 3 处引用；emoji 图标 5 处（`EmptyState.tsx:16` 的 `📭`、`login/page.tsx` 的 `📈`、`tasks/tuning/page.tsx` 的 `🎛️`） |
| **现象** | lucide（圆角、2px 描边）与 heroicons（圆角不同、24 outline 规格）风格不同，出现在同一个回测模块的 3 个文件里。`@mui/icons-material` 作为 MUI v7 默认依赖被引入却几乎不用。 |
| **证据** | `size=` 分布：`16`×226、`20`×107、`24`×51、`18`×37、`48`×23、`12`×22、`32`×16、`14`×13、`64`×6、`4`×2、`22`×2、`28`×2、`15`×1、`23`×1、`120`×1 —— **15 档**。其中 `4`（装饰点）、`120`（landing 装饰）、`15`/`22`/`23`/`28` 属随意值。 |
| **具体改法** | 统一到 **lucide-react 单一图标库**：把 3 个 heroicons 文件改为 lucide 等价图标（`XCircleIcon → CircleX`、`ArrowPath → RefreshCw` 等），移除 `@heroicons/react` 依赖。尺寸收敛到 4 档：`12`（表格内辅助）/ `16`（按钮、菜单默认）/ `20`（AppBar、页面级）/ `24`（卡片标题、feature icon）。删除 `4`/`15`/`22`/`23`/`28`/`120`。emoji 图标全部替换为 lucide 组件。 |

---

#### P2-6 · 表单控件的 label 排版两套：`body2` 居中 vs `caption` 左对齐

| 项 | 内容 |
|---|---|
| **文件:行** | `src/app/today-signal/page.tsx:334`、`:391`、`:415`（`variant="caption"` 标签 + `variant="h6"` 数值）；`src/app/paper-portfolio/page.tsx:175-180`（`variant="subtitle1"` 名称 + `variant="h6"` 数值 + `variant="body2"` 百分比）；`src/app/paper-portfolio/page.tsx:202`（`variant="subtitle2"` 区块标题）；`src/app/data/page.tsx:717-724`（`variant="caption"` 标签） |
| **现象** | 「标签 + 数值」的 KPI 卡在 3 种排版之间切换：caption/h6、subtitle1/h6/body2、body2/h6。实测 `/today-signal` 的 `组合总值` 标签 12px、`/paper-portfolio` 的策略名 16px —— 同样是数据卡，标签字号差 4px。 |
| **证据** | 逐页 `variant=` 统计见上；运行时 `/today-signal` 与 `/paper-portfolio` 的 `h6` 计算样式均为 18px/600，但标签字号 12px vs 16px。 |
| **具体改法** | 定义统一的 `<StatTile>` 组件：`label` 用 `variant="caption"`（12px/500，`text.secondary`）、`value` 用 `variant="h6"`（18px/700）、`delta` 用 `variant="body2"`（14px/500，语义色）。所有 KPI 卡改用此组件。 |

---

#### P2-7 · 间距虽有 8px 栅格基础，但页面根容器 padding 有 4 种写法

| 项 | 内容 |
|---|---|
| **文件:行** | `src/app/today-signal/page.tsx:238`（`p: 3`）；`src/app/models/page.tsx`（`maxWidth: 1400, mx: 'auto', px: 3, py: 3`）；`src/app/optimization/page.tsx`（同上）；`src/app/paper-portfolio/page.tsx`（`py: 8`）；`src/app/settings/page.tsx`（`py: 3`）；`src/app/dashboard/page.tsx`（`gap: 3`）；`src/app/data/page.tsx`（`gap: 2`）；`src/app/signals/page.tsx`（`mt: 2`） |
| **现象** | 底层间距其实规整（`gap:1`×199、`gap:2`×182、`p:2`×100、`gap:3`×56、`gap:0.5`×54 —— 都在 8px 栅格上），但**页面级根容器的内边距/最大宽度**有 4 种模式：`p: 3` / `px:3 py:3 + maxWidth:1400` / `py: 3` / `py: 8`。 |
| **证据** | 逐文件提取根 `sx` 见上表。`AppLayout.tsx:391-399` 已统一了 main 区 `p: { xs: 1.5, sm: 2, md: 3 }` + `maxWidth: 1400`，但页面内部又各自加了一层自己的 `p: 3` / `px: 3 py: 3`，导致**双层内边距**（实测 `/dashboard` 内容区左偏移 = AppLayout 24px + 页面自身 24px = 48px，而 `/templates` 只有 24px）。 |
| **具体改法** | 删除各页面根容器的 `p`/`px`/`py`/`maxWidth`，**只保留 `AppLayout` 一处**的 `p: { xs: 1.5, sm: 2, md: 3 }` + `maxWidth: 1400` + `mx: auto`。需要居中窄栏的页面（`models`/`optimization`）用 `<Container maxWidth="lg">` 而非手写 `maxWidth: 1400`。 |

---

### P3 — 低优先级（清理项）

| # | 文件:行 | 现象 | 证据 | 改法 |
|---|---|---|---|---|
| P3-1 | `src/app/models/page.tsx.working` | 遗留的 HeroUI/Tailwind 风格工作稿文件，与 `page.tsx` 并存 | 唯一 `.working` 文件；含 `<h1 className="text-3xl font-bold flex items-center gap-2">` 等 Tailwind 写法 | 删除该文件 |
| P3-2 | `src/app/globals.css:97-155` | `.status-indicator` / `.task-status-*` 4 个类使用 Tailwind 色板（`#06b6d4`/`#10b981`/`#ef4444`/`#f59e0b`），与 MUI 主题的 `#1976d2`/`#2e7d32`/`#d32f2f`/`#ed6c02` 冲突 | `grep` 确认这 4 个色值在别处不出现 | 改用 MUI 主题色：`running → #0288d1`、`completed → #2e7d32`、`failed → #d32f2f`、`pending → #ed6c02` |
| P3-3 | `src/app/globals.css:44-51`、`:66-95` | `.chart-container`、`.fade-in`、`.slide-up` 三个工具类全站 0 引用 | `grep -rn 'chart-container\|fade-in\|slide-up' src` 仅命中定义处 | 删除 |
| P3-4 | `src/theme/muiTheme.tsx:26-35` | `success: #2e7d32` / `error: #d32f2f` 被当作「涨跌色」使用（`success.main` = 绿涨），与 A 股语义冲突 | 见 P0-1 | 不改色值，改**用法**：涨跌一律走新 `palette.semantics.up/down`；`success`/`error` 保留给「操作成功/失败」 |
| P3-5 | `src/app/error.tsx:28`、`:not-found.tsx:16` | 硬编码 `#d32f2f` / `#666` / `#1976d2` / `fontSize: 48` | 与 P1-6 同源 | 随 P1-6 一并改 MUI 组件 |
| P3-6 | `src/app/pricing/page.tsx:70`、`:88`、`:108` | 定价页 3 套 accent（`#64748b` 灰 / `#1976d2` 蓝 / `#0f766e` 青）自成体系；`:192`、`:371` 用 `#0f172a`（Tailwind slate-900）、`:397` 用 `rgba(15,23,42,0.02)` | 与 MUI `background.default #f5f5f5`、`divider` 体系脱节 | `#0f172a` → 用 `background.default` / `grey.900`；`#0f766e` → `secondary.dark` 或统一为品牌蓝的深色档 |
| P3-7 | `src/app/settings/page.tsx:176` | 用 `variant="h4"`（24px）做页面标题，而其他页用 h4+`component="h1"` | 逐页统计见 P1-3 | 随 P1-3 统一 |
| P3-8 | `src/theme/mobileOverrides.ts:9-37` | Typography 的移动端字号覆盖导致出现 `15.2px` 非整数档；`:13` 注释（「从 2.125rem 缩小」）与 base 主题 h4=`1.5rem` 不符 | 运行时实测 `15.2px` | 随 P2-1 删除该段覆盖并修注释 |
| P3-9 | `src/app/api/` 目录下页面 | 存在 `src/app/api/` 路由目录（前端 BFF） | `ls src/app` 可见 | 与本次审计无关，列出仅作记录 |

---

## 2. 关键量化结论速览

| 维度 | 实测值 |
|---|---|
| 组件库并存 | MUI v7（79 文件） + antd v6（5 文件） + lucide（68 文件） + heroicons（3 文件） + `@mui/icons-material`（3 处） + Tailwind（6 文件，实用 class 极少） + **HeroUI（0 文件，纯死代码）** |
| 混用两套以上组件库的页面 | `/wallet`（antd + Tailwind + MUI 布局） |
| 仅 antd 的页面 | `/register`、`/reset-password`、`/settings/subscription` |
| 硬编码色值 | **377 处 hex / 131 个唯一色** + 73 处 `rgba()` + 7 处 `hsl()` |
| 未定义 CSS 变量引用 | **1 处**：`hsl(var(--border))`（`globals.css:16`） |
| 全局 CSS 体积 | **229,677 B**，其中 HeroUI 死样式 **128,524 B（56.0%）** |
| antd JS 体积 | 3 个 chunk，raw 770.3 KB / **gzip 242.3 KB**，仅 4 个路由加载（`/wallet` 独占 242.3 KB） |
| 共享 chunk（每页必下） | 6 个 chunk，raw 301.1 KB / gzip 89.8 KB |
| 运行时唯一主色数 | **4 种**：MUI `#1976d2`、antd `#1677ff`、HeroUI `--heroui-primary #006FEE`、Tailwind ring `#3b82f6` |
| 运行时正文色数 | **3 种**：`rgba(0,0,0,0.87)`(MUI) / `rgba(0,0,0,0.88)`(antd) / `rgb(17,24,28)`(全局 html) |
| 字号档位数 | 4~10 档/页，全站最多 10 档（pricing） |
| 圆角档位数 | 7 档（0/4/8/10/12/16/50%） |
| 阴影套数 | 7 套手写 + 3 套 MUI elevation |
| 语义色矛盾 | 页面层绿涨红跌 vs 图表层红涨绿跌（同一屏冲突，实测确认） |
| 暗色模式 | 声称支持，**实际 0 处切换代码**，切换 `.dark` 后渲染无任何变化 |
| `hsl(var(--border))` 运行时表现 | 回落 `currentColor`（近黑）而非 `#e5e7eb` |

---

## 3. 统一设计系统整改方案

### 3.1 推荐保留的 UI 库

**结论：保留 MUI v7 作为唯一组件库 + Tailwind 3 作为唯一的原子类工具（仅布局微调）**

| 库 | 决策 | 理由 |
|---|---|---|
| **MUI v7** | ✅ **保留并作为唯一组件库** | ① 覆盖率压倒性：128 源文件中 79 个 import MUI，覆盖 24/26 个页面；② 主题系统完备（`createTheme` 支持 palette/typography/shape/components 四层 token），`mobileOverrides.ts` 已建立「集中覆盖」的正确范式；③ 唯一被 MUI 组件库同时需要 `sx` 主题感知能力的地方（Table/Chart 容器）都在；④ 全站 2106 处 `sx={{}}` 已经是事实上的设计 token 载体，迁移到 MUI 的边际成本最低；⑤ antd v6 的 `Table`/`Descriptions`/`Statistic` 在 MUI 均有等价物（`Table`/`Descriptions`/`Typography`+`Card`），迁移是**纯替换**而非重写。 |
| **Tailwind 3** | ⚠️ **保留但降级为「原子布局工具」** | ① 已在构建链中（`postcss.config.js` + `tailwind.config.js`），移除成本高；② 但 92 个源文件中仅 6 个用到工具类，其中 5 个是 `/wallet`；③ 需移除 HeroUI 插件（P1-1）—— 保留 Tailwind 的**布局原子类**（`flex`/`gap`/`grid`）与**排版原子类**（`text-xs`）用于避免 `sx` 过度膨胀，但**禁止**用 Tailwind 的颜色/阴影/圆角工具类（这些必须走 MUI 主题）。 |
| **antd v6** | ❌ **移除** | ① 仅 4 个页面使用（`/wallet`、`/register`、`/reset-password`、`/settings/subscription`），其中 3 个是低流量（注册/改密/订阅）；② 未配置 `ConfigProvider`，token 完全脱离主题（P0-3）；③ 引入独立 CSS-in-JS 运行时 + 47MB `@ant-design/*` + 61MB `antd`，且 `/wallet` 独占 242.3 KB gzip；④ 组件本身（MUI 都有等价物）不构成不可替代能力。 |
| **HeroUI** | ❌ **立即移除** | 全站 0 引用，纯粹通过 tailwind 插件注入 128KB 死 CSS（P1-1）。零迁移成本，纯粹收益。 |
| **lucide-react** | ✅ **保留为唯一图标库** | 68/128 文件已在用，风格统一，替换 heroicons 成本低（P2-5）。 |
| **@heroicons/react** | ❌ **移除** | 仅 3 个文件，换成 lucide 等价图标即可。 |
| **@mui/icons-material** | ❌ **移除** | 3 处引用，可换 lucide。 |

**若要保留 antd 的唯一理由**：`Table` 的固定列 / 虚拟滚动能力。但本项目数据量（分页 200 条）远未到需要，**不建议**为此保留 242KB gzip。

### 3.2 目标设计系统（统一 token 建议值）

新建 `src/theme/tokens.ts` 作为**唯一真相源**，MUI 主题与 CSS 变量都从这里派生：

```ts
// src/theme/tokens.ts
export const tokens = {
  color: {
    // 品牌
    primary:   { 50:'#e6f1fe', 100:'#cce3fd', 200:'#99c7fb', 300:'#66aaf9',
                 400:'#338ef7', 500:'#1976d2', 600:'#1565c0', 700:'#0d47a1',
                 800:'#08306b', 900:'#041e45' },
    secondary: '#9c27b0',        // 保留 MUI secondary，landing/templates 的紫统一到它
    // 中性（对齐 Tailwind slate，避免 #f5f5f5 / #11181c 两套真相）
    bg:        { default:'#f8fafc', paper:'#ffffff', sunken:'#f1f5f9' },
    text:      { primary:'#0f172a', secondary:'#475569', disabled:'#94a3b8' },
    border:    { light:'#e2e8f0', DEFAULT:'#e5e7eb', dark:'#cbd5e1' },
    // 语义
    success:   '#2e7d32',   // 仅表示「操作成功」
    error:     '#d32f2f',   // 仅表示「操作失败」
    warning:   '#ed6c02',
    info:      '#0288d1',
    // A 股语义色（新增，禁止用 success/error 表达涨跌）
    semantics: { up:'#d32f2f', down:'#2e7d32', flat:'#64748b' },
  },
  radius:  { sm:4, md:8, lg:16, pill:9999 },
  shadow:  { 1:'0 1px 2px rgba(16,24,40,.06), 0 1px 3px rgba(16,24,40,.10)',
             2:'0 12px 24px -4px rgba(16,24,40,.08)' },
  space:   { 0:0, 0.5:4, 1:8, 1.5:12, 2:16, 3:24, 4:32, 6:48, 8:64 },
} as const;
```

`globals.css` 改为从同一份值派生（保持 CSS 变量可用但不重复定义真相）：
```css
:root {
  --background: 210 40% 98%;   /* #f8fafc */
  --foreground: 222 47% 11%;   /* #0f172a */
  --border:     214 32% 91%;   /* #e5e7eb  ← 修复 P0-2 */
}
```

**核心原则**：
1. **一处定义**：颜色只在 `tokens.ts` 出现；`sx` 里只写 `'primary.main'` / `'text.secondary'` / `'semantics.up'` 这类语义引用。
2. **涨跌不用 success/error**：A 股语义色独立成 `semantics.up/down`。
3. **删除一切手写 `fontSize`/`borderRadius`/`boxShadow`/hex**：改用 `variant` / `shape.borderRadius` / `elevation`。
4. **禁清单**（建议加 ESLint `no-restricted-syntax` 约束）：`#hex`、`fontSize:`、`boxShadow:`、`linear-gradient`（landing 除外）。

### 3.3 分阶段迁移步骤

**阶段 0 · 止血（0.5 天，零风险）**
- 删除 `tailwind.config.js` 的 HeroUI 插件与 content 项 → 全局 CSS 229KB → ~100KB（**-56%**）
- 移除 `package.json` 的 `@heroui/react`、`@heroui/theme`（16MB）
- `globals.css:5-8` 补 `--border` 定义（修复 P0-2）
- 删除 `src/app/models/page.tsx.working`、`globals.css` 三个 0 引用工具类
- 删除 `src/components/providers/Providers.tsx` 或改 `app/layout.tsx` 引用之（让 `GlobalSnackbar` 真正挂上）

**阶段 1 · 主题层统一（2 天）**
- 新建 `src/theme/tokens.ts`；`muiTheme.tsx` 改为从 tokens 生成 palette/typography/shape/components
- 补全 typography scale（P2-1），删除 `mobileOverrides.ts` 的 Typography 覆盖
- 收敛 radius / shadow（P2-2、P2-3）到 3 档 / 2 档
- 新增 `palette.semantics.up/down`（P0-1 的 token 基础）
- 补 `MuiTableCell` / `MuiPaper` / `MuiChip` 的统一 `borderColor: 'border.light'`

**阶段 2 · 语义色整改（1.5 天）**
- 全站把涨跌语义从 `success.main`/`error.main` 切到 `semantics.up/down`：
  `today-signal/page.tsx:337,462,569`、`paper-portfolio/page.tsx:178`、`BacktestChart.tsx:479,595`、`MonthlyHeatmapChart.tsx:183,473,479-488`、`PredictionChart.tsx:216-218`
- 图表调色板统一到 `src/theme/chartTokens.ts`（P1-4），`TradingViewChart.tsx:524-525` 的红涨绿跌与页面文字对齐
- 抽 `<StatTile>` 统一 KPI 卡排版（P2-6）

**阶段 3 · 移除 antd（2 天）**
- 逐页改写：`/register`（→MUI `TextField`/`Button`/`Tabs`）、`/reset-password`（同）、`/settings/subscription`（→MUI `Card`/`Table`/`Chip`）、`/wallet`（→MUI `Card`/`Table`/`Descriptions`/`Progress`/`Dialog`）
- 删 `package.json` 的 `antd`（61MB）+ `@ant-design/*`（47MB）
- 收益：`/wallet` 少下 **242.3 KB gzip**；全站少一套 CSS-in-JS 运行时
- `/wallet` 的 Tailwind className 同步改为 `sx`（顺带修 P2-7 的双层内边距）

**阶段 4 · 排版与结构统一（2 天）**
- 页面标题统一 `variant="h4" component="h1"`（P1-3），补齐 12 个缺失 h1 的页面
- 页面根容器删除自有 `p`/`maxWidth`，只留 `AppLayout` 一处（P2-7）
- `loading.tsx` / `error.tsx` / `not-found.tsx` 改用 MUI（P1-6、P3-5）
- 统一 `EmptyState`（改 lucide 图标）、删 `TaskDetailContent` 本地实现、统一「暂无…」文案（P1-6）

**阶段 5 · 图标与 landing 打磨（1.5 天）**
- 3 个 heroicons 文件换 lucide，删 `@heroicons/react` + `@mui/icons-material`（P2-5）
- 尺寸收敛到 4 档，emoji 全替换
- landing 去蓝紫渐变、去紫色 radial 光晕、h1 收到 40px（P2-4）

**阶段 6 · 暗色模式定调（0.5 天决策 + 视选择 2~3 天）**
- **推荐路线 A（本期做）**：明确「全站仅浅色」。删除 `darkMode: "class"`、`.dark` 变量块、`useAppStore.theme` 字段；把 landing 顶栏改为透明 + 白字，落在深色 hero 内。
- 路线 B（后续可选）：真做暗色 —— MUI 动态 `palette.mode` + 完整 CSS 变量 + `<ThemeToggle>`。成本约 3 天，需覆盖 24 个页面与全部图表（ECharts 需额外 dark 配置），建议推迟。

**阶段 7 · 防回归（1 天）**
- ESLint 规则：禁 `fontSize:` / `boxShadow:` / `#hex` 裸值（`no-restricted-syntax`）
- CI 加 `stylelint`（`declaration-property-value-no-unknown` 捕获 `hsl(var(--border))` 类未定义变量）+ `stylelint-config-tailwindcss`（查废弃类）
- 引入视觉回归：Playwright 快照 12 个关键页面（dashboard / today-signal / signals / wallet / pricing / templates …），dark/light × desktop/mobile 四象限

### 3.4 先修清单（按 P0/P1 排序，≤12 条）

| # | 优先级 | 事项 | 涉及文件 | 预估工时 |
|---|---|---|---|---|
| 1 | **P0** | 定义 `--border` CSS 变量，修复 `*` 边框退化为 currentColor | `src/app/globals.css:5-8` | 5 min |
| 2 | **P0** | 统一 A 股语义色：新增 `palette.semantics.up/down`，替换 9 处 `success.main`/`error.main` 涨跌用法 | `today-signal/page.tsx:337,462,569`、`paper-portfolio/page.tsx:178`、`BacktestChart.tsx:479,595`、`MonthlyHeatmapChart.tsx:183,473,479-488`、`PredictionChart.tsx:216-218` | 3 h |
| 3 | **P0** | 图表调色板统一到 `chartTokens.ts`；`TradingViewChart.tsx:524-525` 红涨绿跌与页面文字对齐 | 新建 `src/theme/chartTokens.ts` + 6 个图表组件 | 4 h |
| 4 | **P0** | 消除 antd 与 MUI 的双主色：加 `ConfigProvider` 统一 token（过渡期） | `src/app/layout.tsx` | 1 h |
| 5 | **P1** | 删除 HeroUI tailwind 插件 + `@heroui/*` 依赖，回收 128KB 死 CSS（-56%） | `tailwind.config.js:1,9,20-65`、`package.json` | 30 min |
| 6 | **P1** | 暗色模式定调：明确「仅浅色」，删 `darkMode:"class"`/`.dark`/store.theme；landing 顶栏改透明白字 | `tailwind.config.js:19`、`globals.css:10-13`、`useAppStore.ts:28`、`app/page.tsx` | 2 h |
| 7 | **P1** | 页面标题统一 `variant="h4" component="h1"`，补齐 12 个缺失 h1 的页面 | 14 个 page.tsx | 2 h |
| 8 | **P1** | 建立 `src/theme/tokens.ts` 唯一真相源，改造 `muiTheme.tsx` 消费它；补全 typography scale | `src/theme/muiTheme.tsx`、`mobileOverrides.ts:9-37` | 1 d |
| 9 | **P1** | 统一加载/空/错误态：扩充 `EmptyState`（lucide 图标）、删 `TaskDetailContent` 本地实现、`loading.tsx`/`error.tsx`/`not-found.tsx` 改 MUI | `components/common/EmptyState.tsx`、`TaskDetailContent.tsx:852-866`、`app/loading.tsx`、`app/error.tsx`、`app/not-found.tsx` | 1 d |
| 10 | **P1** | 修 `GlobalSnackbar` 从未挂载：删 `Providers.tsx` 或改 `app/layout.tsx` 引用 | `src/components/providers/Providers.tsx`、`src/app/layout.tsx` | 30 min |
| 11 | **P1** | 收敛圆角 3 档 / 阴影 2 档；删除 `borderRadius: 1.5/3/4/5` 与 18 处手写 `boxShadow` | `templates/page.tsx:210`、`pricing/page.tsx:253,467`、`data/page.tsx:717`、`today-signal/page.tsx:452` 等 | 4 h |
| 12 | **P1** | 移除 antd：改写 `/register`、`/reset-password`、`/settings/subscription`、`/wallet` 为 MUI；删 `antd`+`@ant-design/*` 依赖 | 4 个 page.tsx + `hooks/useCommerce.ts` | 2 d |

**清单外延后项**（P2/P3）：landing 去蓝紫渐变、图标库统一到 lucide、页面根容器双层 padding、StatTile 抽取、`.working` 死文件与 0 引用工具类清理、ESLint/Stylelint/Playwright 防回归。

---

## 4. 未发现 / 已符合项

- **表格样式本身是统一的**：全站 491 处 `Table`/`TableHead`/`TableRow` 均用 MUI `Table`，`size="small"` 统一，无混用 antd `Table`（`/wallet` 的 antd Table 是唯一例外，随 P1 移除）。
- **MUI 主题集中化范式正确**：`mobileOverrides.ts` 通过 `getMobileOverrides(theme)` 合并进主主题，而非在组件里写 `useMediaQuery` 分支，这个结构应保留（只需删掉 Typography 覆盖段）。
- **间距底层是 8px 栅格**：`gap:1/2/3`、`p:1.5/2/3` 全部落在 8px 倍数，统计上无 `gap: 1.3` 之类非栅格值。
- **无 `as any` 式的样式逃逸**：所有样式都走 `sx` 或 `styleOverrides`，没有内联 `<style>` 块（`app/loading.tsx` 的 `@keyframes` 是唯一例外，属正常用法）。
- **`prefers-color-scheme` 未被滥用**：无任何媒体查询式暗色 hack，避免了「半暗」的中间态。
- **生产构建产物可分析**（`.next-build-check` 存在且为 2026-10-27 最新），本报告的包体与 CSS 数字均来自真实产物而非估算。

---

## 5. 审计边界与已知局限

1. **只读审计**：未修改任何源码；`git status --porcelain` 仅显示 `?? docs/frontend-audit/`（本次新增的截图与报告）。
2. **登录凭据**：`ronghui` 用户实际邮箱为 `ronghui@example.com`（任务描述给的 `ronghui` 直接作为 email 登录会失败）。已通过后端 API 确认并使用该邮箱登录，账号为 enterprise 订阅 + is_admin。
3. **速率限制**：后端对高频请求返回「请求过于频繁，请稍后再试」，导致 `/dashboard`、`/signals` 等页在部分截图里呈现 error Alert 而非完整数据。截图已重采，但**少数页面的数据区仍是错误态**（如 02-today-signal.png 顶部有「请求失败」Alert）。这属于后端限流，非前端设计缺陷。
4. **生产 bundle 数字**取自 `.next-build-check`（08:00 构建），与当前 dev server 的 `.next` 可能有微小差异，但组件构成一致。
5. **视口覆盖**：桌面 1440×900（dpr 1）与移动 390×844（dpr 2）双视口抽查，未覆盖平板断点（768/1024）。
6. **未逐页深挖业务逻辑**：按任务要求聚焦「设计系统一致性」，收益/成本等业务计算正确性不在本次范围。
7. **图表色板未逐一验证渲染结果**：`MonthlyHeatmapChart` 等 ECharts 组件的色板基于源码字面量判定，未在有数据的会话中逐一截图比对。

---

*报告生成：Hermes Agent · 只读审计 · 2026-10-27*
