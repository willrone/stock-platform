# 前端整体布局 / 导航 / 信息架构 审计报告

- 审计日期：2026-10-27
- 审计范围：`frontend/src/components/layout/AppLayout.tsx`、`frontend/src/app/{layout,loading,error,not-found}.tsx`、`frontend/src/components/mobile/*`、`frontend/src/theme/{muiTheme,mobileOverrides}.ts`、`frontend/src/app/globals.css`
- 审计方式：只读。浏览器实跑（`browser_exec`，隔离 session `layout-audit`），逐断点 DOM 几何测量 + 截图取证，未修改任何源码
- 被测环境：前端 `http://localhost:3000`（dev），后端 `http://127.0.0.1:18082`（健康）
- 登录：账号 `ronghui`（实际邮箱 `ronghui@example.com`，后端不接受 `ronghui` 作为 email）、密码 `XMPzFEZDFpvV`。通过 API 取 token 后注入 `localStorage`（`access_token` / `refresh_token` / `user_info`），未在浏览器中输入密码

## 结论摘要

**整体布局评估：不合格，需重构全局导航壳。**

核心问题不是"某处样式不好看"，而是**导航信息架构与实现方式根本不匹配**：16 个一级导航项被平铺进顶栏按钮组，没有侧边栏、没有分组、没有面包屑、没有二级入口。这在 1200–1440px（绝大多数笔记本）区间会直接把顶栏撑成 159px 高、标签竖切成 4 行、logo 压成 6 个单字竖排 —— 这是 P0 级别的可用性阻断，不是打磨项。

叠加 3 个 P0 级功能缺陷：**退出登录不清 token**（`auth_token` 键名错误，会话实际未登出）、**管理端入口刷新后消失**（zustand store 未从 localStorage 反序列化，`user` 恒为 `null`）、**`/profile` 与 `/account-settings` 两条路由 404**。

设计体系层面，`globals.css` 是一套**完全未被使用的死代码**（10 个类全部 0 引用），且其中 `* { border-color: hsl(var(--border)) }` 引用了从未定义的 `--border` 变量，导致该声明整体失效、边框颜色回落到 `currentColor`（实测 `rgba(0,0,0,0.87)` 近黑）。项目实际运行在 MUI + HeroUI + Tailwind 三套样式体系叠加之上，spacing/radius/shadow 无单一真源。

---

## 问题清单

### P0 阻断

#### 1. 顶栏 16 项平铺，1200–1440px 区间顶栏撑高 159px、菜单标签竖切成 4 行
- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:245-271`（顶栏菜单容器）、`:63-145`（`menuItems` 数组，16 项平铺无分组）
- **现象描述**（实测 DOM 几何，`document.querySelector('.MuiToolbar-root')`）：
  | 视口宽 | 顶栏高 | 单按钮宽 | 标签行数 | 页面横向溢出 |
  |---|---|---|---|---|
  | 1920 | 64px | 93px | 1 | 否 |
  | 1600 | 64px | 76px | 2 | 否 |
  | 1440 | **86px** | 65px | 2 | 否 |
  | 1366 | **159px** | 64px | **4** | **是**（scrollWidth 1332 > 1274） |
  | 1280 | **159px** | 64px | **4** | **是**（scrollWidth 1332 > 1274） |
  | 1200 | **159px** | 64px | **4** | **是**（scrollWidth 1332 > 1200） |

  按钮 `white-space: normal`（实测 `getComputedStyle(button).whiteSpace === 'normal'`），无 `whiteSpace: 'nowrap'`，也无 `flexShrink: 0`。`Range.getClientRects()` 实测「数据管理」在 1280px 下占 2 个 rect（y=21 与 y=46），1366px 下占 4 个 rect —— 中文标签被压成竖排多行。
  顶栏 `flexGrow: 1` 的菜单容器（实测宽 1072px）在 1280px 下已超出可用宽度，把右侧操作区（连接状态 Chip + 铃铛 + 头像，实测 right=1332）挤出视口，页面出现 58px 横向滚动条。
- **复现路径**：登录后把浏览器窗口宽度调至 1200–1440px（1366×768 笔记本最典型）→ 观察顶栏高度与「数据管理」「系统监控」等标签是否竖排 → 横向滚动页面查看右侧头像是否被切掉。
- **建议改法**：改为**左侧固定侧边栏 + 顶栏只保留面包屑/搜索/账号区**的经典结构。若必须保留顶栏平铺，则需要：① 给 `Button` 加 `sx={{ whiteSpace: 'nowrap', flexShrink: 0 }}`；② 把 `menuItems` 按业务域分组为「交易/研究（预测分析、策略回测、策略信号、今日信号、纸面信号、多策略纸面盘）」「平台（任务管理、模型管理、数据管理、策略模板、参数调优、超参优化）」「系统（系统监控、系统设置、充值订单）」，分组折叠进 `Menu` / `Popover`；③ `lg` 断点从 1200px 上调到 1600px 以上，并给顶栏加 `overflow-x: auto` 兜底。**推荐方案是引入 `Drawer variant="permanent"` 侧边栏**，因为 16 个一级入口在任何断点下平铺都不可用。
- **截图文件名**：`16-nav-1366.png`、`17-nav-1200.png`、`06-nav-1280-overflow.png`、`02-dashboard-1440.png`

#### 2. Logo「股票预测平台」被压成 6 个单字竖排
- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:233-242`（Logo 容器 `display:flex` + `Typography variant="h6"` 无 `whiteSpace`/`flexShrink` 保护）
- **现象描述**：1280px 实测 Logo 标题元素宽 **32px**、高 **154px**；`Range.getClientRects()` 返回 **6 个独立 rect**（y = 6 / 31 / 57 / 83 / 108 / 134，每个 16×18），即「股/票/预/测/平/台」6 个字各占一行。Logo 外层 Box 实测宽 48px，即 6 个汉字被挤在 32px 宽的柱子里。1366px 截图左上角可直接看到竖排字形。
- **复现路径**：视口宽设为 1280–1440px → 观察左上角平台名称是否被压成竖排单字。
- **建议改法**：给 Logo `Typography` 加 `sx={{ whiteSpace: 'nowrap', flexShrink: 0 }}`，并给外层 Box 加 `minWidth` 保护；或把品牌名在 `lg` 以下用 `display: {xs:'none', lg:'block'}` 隐藏、只留图标（图标本身已有 `BarChart3`）。同时给品牌区加 `flexShrink: 0` 防止被菜单容器挤压。
- **截图文件名**：`16-nav-1366.png`、`06-nav-1280-overflow.png`

#### 3. 退出登录不清除会话 token，登录态实际未退出
- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:200-204`
- **现象描述**：`handleUserAction('logout')` 执行 `localStorage.removeItem('auth_token')`，但登录流程写入的键是 **`access_token`**（`frontend/src/app/login/page.tsx:46`）。实测点击「退出登录」后：
  - URL 跳到 `/login` ✓
  - `localStorage` 键集合前后**完全不变**：`["refresh_token","user_info","access_token"]`
  - `localStorage.getItem('access_token')` 仍返回有效 token
  - 随后手动访问 `/dashboard` → **成功进入仪表板**，数据正常加载（显示"1 总任务数 / audit_smoke_rsi"）
  正确做法见 `frontend/src/services/api.ts:58-66` 的 `clearStoredAuth()`，它删的是正确的三个键，但退出登录流程没有复用它。
- **复现路径**：登录 → 右上角头像 → 退出登录 → 观察地址栏已到 `/login` → 手动输入 `localhost:3000/dashboard` → 页面直接进入仪表板并显示数据（未重新认证）。
- **建议改法**：把 `AppLayout` 的 logout 分支改为复用 `api.ts` 已有的 `clearStoredAuth()`，删除 `access_token` / `refresh_token` / `user_info` 三键；或至少把 `removeItem('auth_token')` 改为 `removeItem('access_token')` + `removeItem('refresh_token')`。同时应调用 `useAppStore.getState().setUser(null)` 清理内存态，并 `router.replace('/login')`（非 `push`，避免用户后退回到已登录页）。
- **截图文件名**：`03-user-menu.png`（退出菜单项取证）

#### 4. 管理端入口「充值订单」在任何刷新后都不出现
- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:143`（`adminOnly: true`）、`:254` 与 `:358`（`item.adminOnly || user?.is_admin` 过滤）、`frontend/src/stores/useAppStore.ts:52-60`（store 初始 `user: null`，**无 persist / rehydrate 中间件**）
- **现象描述**：`useAppStore` 用 `create()` + `devtools()` 创建，**没有 persist**，全仓库无任何 `useAppStore` 的持久化/反序列化调用（仅 `login/page.tsx:25`、`register/page.tsx:30` 在登录时 `setUser`）。因此页面刷新后 `user` 恒为 `null`。
  实测：注入 `localStorage.user_info = {"is_admin": true, ...}` 后刷新 `/dashboard`，`localStorage` 中 `is_admin` 读取为 `true`，但顶栏按钮数实测 **15** 个，`includes('充值订单')` → **false**。`/admin/payments` 页面本身能直接访问并渲染「充值订单管理」，但**导航中无任何入口**。
  附带证据：`AppLayout.tsx:202` 删除的是 `auth_token`，说明 `auth_token` 是历史遗留键名，代码中已无任何写入方 —— 佐证 store/token 键名管理存在系统性脱节。
- **复现路径**：以管理员登录 → F5 刷新任意内页 → 观察顶栏「系统设置」右侧是否出现「充值订单」→ 不出现；`/admin/payments` 只能靠手输 URL 进入。
- **建议改法**：给 `useAppStore` 加 zustand `persist` 中间件（`storage: createJSONStorage(() => localStorage)`，`partialize` 只持久化 `user` / `isAuthenticated`），或在 `AppLayout` 挂载时用 `useEffect` 从 `localStorage.getItem('user_info')` 解析并 `setUser()` 恢复（注意 store 的 `User` 接口用 `name` 字段，而后端返回 `username`，需做映射）。补上 `/admin/payments` 之外的第二个可见入口（侧边栏分组内）。
- **截图文件名**：`02-dashboard-1440.png`（15 项、无充值订单）

#### 5. 用户菜单两项全部 404
- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:195`（`router.push('/profile')`）、`:198`（`router.push('/account-settings')`）
- **现象描述**：实测点击「个人资料」→ URL 变为 `/profile` → 主区域渲染 `not-found.tsx`：「404 / 页面不存在 / 您访问的页面不存在或已被移除」；直接访问 `/account-settings` 结果相同。文件系统确认 `src/app/profile/` 与 `src/app/account-settings/` **均不存在**（`ls` 报 No such file or directory）。
- **复现路径**：登录 → 右上角头像 → 点「个人资料」→ 看到 404 页；返回后再点「账户设置」→ 同样 404。
- **建议改法**：二选一 —— ① 新建 `src/app/profile/page.tsx` 与 `src/app/account-settings/page.tsx` 真实页面；② 若本期不实现，先移除这两个 `MenuItem`（或 `disabled`），避免暴露死链。同时 `not-found.tsx:20` 的「返回仪表板」`href="/dashboard"` 与菜单项 key `'/'` 不一致，404 页回跳链路和导航语义应统一。
- **截图文件名**：`04-profile-404.png`、`05-notfound-404.png`

---

### P1 严重

#### 6. `仪表板` 菜单项 key 是 `/`，实际路由是 `/dashboard`，导致高亮永不生效
- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:65`（`key: '/'`）、`:257` 与 `:361`（`const isActive = pathname === item.key`）
- **现象描述**：菜单项 `仪表板` 的 `key` 为 `'/'`，而仪表板真实路由是 `/dashboard`（`src/app/dashboard/page.tsx`）。`/` 是落地页（`src/app/page.tsx:133` 在已登录时 `router.replace('/dashboard')`）。
  实测在 `/dashboard` 页：顶栏 `MuiButton-contained`（激活态）按钮集合为 **空数组 `[]`** —— 15 个菜单项无一高亮。移动端抽屉同样：`.Mui-selected` 元素为 `[]`。
  实测点击「仪表板」按钮 → `router.push('/')` → `/` 重定向到 `/dashboard` —— 绕了一圈才到达目标页。
  路由扫描证实其他菜单项 `key` 均能匹配（`/tasks`、`/models`、`/data`、`/monitoring`、`/predictions`、`/backtest`、`/templates`、`/tasks/tuning`、`/signals`、`/today-signal`、`/paper-signal`、`/paper-portfolio`、`/optimization`、`/settings` 全部实测 `act` 正确），**仅「仪表板」一条失效**。
- **复现路径**：访问 `/dashboard` → 观察顶栏无任何菜单项呈选中态（蓝色实心态）；打开移动端抽屉 → 无高亮行。
- **建议改法**：把 `menuItems` 首项的 `key` 改为 `'/dashboard'`，并把 `isActive` 判定改为带前缀的匹配以覆盖子路由：`const isActive = pathname === item.key || pathname.startsWith(item.key + '/')`（`/tasks/tuning` 需要同时点亮「任务管理」和「参数调优」时，另加 `matchPaths` 字段显式声明父子关系）。落地页 `/` 建议从菜单中移除，改为「返回首页」次级链接。
- **截图文件名**：`02-dashboard-1440.png`

#### 7. `/tasks/create`、`/tasks/compare`、`/tasks/[id]`、`/wallet`、`/settings/subscription` 无任何导航入口
- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:63-145`（`menuItems` 缺这些项）
- **现象描述**：对 `src/app` 全量 `find . -name page.tsx` 得到 27 条路由，与 `menuItems` 的 16 个 key 求差集，缺 11 条（`admin/payments` 虽有 key 但因问题 4 被过滤掉）。实测逐条访问：`/wallet`（钱包中心）、`/settings/subscription`（订阅）、`/tasks/create`（创建任务）、`/tasks/compare`（配置对比）均**能正常渲染但无导航可达路径** —— 只能手输 URL 或从页面内部跳转（实测 `/tasks/create` 的 `act` 为 `[]`，`/wallet` 仅被 `pricing/page.tsx:216` 引用）。
  `/admin`（无 `page.tsx`，只有 `payments/`）实测渲染 404 —— 父路径无索引页。
- **复现路径**：登录 → 手输 `localhost:3000/wallet` → 页面正常渲染，但顶栏无「钱包」项 → 返回顶栏无法再次到达该页（除手输 URL 外无路径）。
- **建议改法**：把「钱包」并入用户菜单（Avatar 下拉），「订阅管理」放进「系统设置」分组，「创建任务」作为「任务管理」的子项/主 CTA 按钮，「配置对比」挂到任务详情页内联。`/admin` 增加 `src/app/admin/page.tsx` 做管理员索引页。**同时必须补上 16→N 的分组结构**，否则补完入口后顶栏会更挤。
- **截图文件名**：`02-dashboard-1440.png`（对照缺失项）

#### 8. 移动端抽屉 320px 面板内内容硬编码 256px，右侧留 64px 空白
- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:341`（`<Box sx={{ width: 256, p: 2 }}>`）、`frontend/src/theme/mobileOverrides.ts:194-203`（`MuiDrawer` 移动端 `width: '85%', maxWidth: 320`）
- **现象描述**：`mobileOverrides` 把 Drawer 纸面宽度设为 `85% / maxWidth 320`，但 `AppLayout` 内的内容 Box 写死 `width: 256`。390px 视口实测：纸面宽 **320px**、内容 Box 宽 **256px** → 右侧 **64px 死白**（`contentOverflow: false`，内容不溢出，但视觉上菜单项只占面板 80% 宽）。
  更严重的是断点边界跳变：实测 599px → 纸面 320px；**600px → 纸面 256px**；601px → 256px。`MuiDrawer` 覆盖只写在 `down('sm')`（<600px），而 600–1199px 区间 MUI 默认纸面宽 240px 却被内容 Box 的 256px 撑开 —— 断点处抽屉宽度从 320 突降到 256。
- **复现路径**：视口调至 390px → 打开汉堡菜单 → 观察菜单项右侧大片空白；再逐像素调至 600px 观察抽屉宽度突变。
- **建议改法**：删除 `AppLayout.tsx:341` 的固定 `width: 256`，改为 `sx={{ width: '100%', p: 2 }}`；`MuiDrawer` 覆盖的断点从 `down('sm')` 扩到 `down('lg')`（与 `AppLayout:339` 的 `display: {lg:'none'}` 对齐），统一为 `width: '85%', maxWidth: 320`。
- **截图文件名**：`19-mobile-390-drawer-open.png`、`19-mobile-390-drawer-open.png`

#### 9. 小屏手机（360×640）抽屉底部菜单项被裁切
- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:335-388`（`Drawer` 无 `PaperProps` 高度/滚动配置）、`frontend/src/theme/mobileOverrides.ts:194-203`
- **现象描述**：390×844 实测抽屉纸面高 844px、15 项菜单总高约 810px，最后一项「系统设置」bottom = 786px（勉强可见）。在 **360×640** 实测：纸面高 **640px**、`scrollHeight` **810px**，最后一项「系统设置」y=739 / **bottom=786** → **超出视口 146px，被完全裁切**，且纸面未设 `overflowY: 'auto'`（实测 `paperScrollH > paperH` 但无法滚动到菜单底部，用户无法触达「系统设置」）。
- **复现路径**：视口设为 360×640（iPhone SE 尺寸）→ 打开汉堡菜单 → 观察底部是否能看到「系统设置」→ 看不到且无法滚动。
- **建议改法**：给 `Drawer` 加 `PaperProps={{ sx: { overflowY: 'auto' } }}`，并让菜单区在内容超高时自身滚动（`Box` 包 `List` 加 `overflowY:'auto', flex:1`）；抽屉头部（Logo + 关闭按钮）建议 `position: sticky; top: 0` 保持可见。同时考虑把 15 个平铺项分组后条目数可降到 8–9 行。
- **截图文件名**：`13-mobile-360-drawer.png`

#### 10. 登录 / 注册 / 定价页套用完整已登录应用外壳
- **文件路径:行号**：`frontend/src/app/layout.tsx:24-29`（根布局无条件包裹 `AppLayout`）
- **现象描述**：根布局对所有路由无条件渲染 `AppLayout`，没有按路由做「公开页 / 应用页」分组。实测 `/login`、`/register`、`/pricing` 三个公开页均带 **15 个已登录导航按钮**（仪表板…系统设置）、**已连接状态 Chip**、**用户头像**和**页脚**。
  受影响后果：① 未登录访客看到一堆点进去会 404 或被重定向的功能入口（`/profile`、`/account-settings` 如问题 5）；② 登录页顶部出现「已连接」WebSocket 状态，但对未认证用户毫无意义；③ 视觉上「量化研究平台 / AI 驱动的量化回测与因子分析工具」（登录卡片内的品牌名）与导航栏「股票预测平台」**是两套不同的品牌名称**，同一屏出现两个品牌。
- **复现路径**：退出后访问 `localhost:3000/login` → 观察登录卡片上方是否有一排已登录导航。
- **建议改法**：拆分根布局为两套：新建 `src/app/(app)/layout.tsx` 放 `AppLayout` 承载所有需鉴权页面，公开页（login/register/pricing/reset-password）留在 `src/app/` 根级不放 `AppLayout`；或用 `usePathname()` 在 `AppLayout` 内对公开路径 `return <>{children}</>`。同时统一品牌名 —— 登录卡片的「📈 量化研究平台」与导航栏「股票预测平台」必须二选一。
- **截图文件名**：`01-login.png`

---

### P2 一般

#### 11. 全局 CSS 变量 `--border` 从未定义，`* { border-color }` 整条规则失效
- **文件路径:行号**：`frontend/src/app/globals.css:15-17`
- **现象描述**：`globals.css` 定义了 `--background` / `--foreground`（`:root`，第 5-8 行），但**从未定义 `--border`**。实测 `getComputedStyle(document.documentElement).getPropertyValue('--border')` 返回**空字符串**。因此 `border-color: hsl(var(--border))` 是非法值，浏览器按 `currentColor` 处理：实测新建 `div` 设 `border-width:1px; border-style:solid` 无显式 border-color 时，`borderColor` 计算值为 **`rgba(0, 0, 0, 0.87)`**（= 文字色近黑）而非分隔线灰。
  实际影响有限：MUI 组件自带 `borderColor: 'divider'`（实测 `rgba(0,0,0,0.12)`）优先级更高，未被覆盖。但任何**非 MUI 的原生元素**（`hr`、`table` 的 `border`、原生 `fieldset`/`legend`、Tailwind 预设的 `border` 类）都会拿到近黑边框。
  注：`.dark` 块（第 10-13 行）同样只切 `--background`/`--foreground`，而 MUI 主题 `palette.mode` 硬编码 `'light'`（`muiTheme.tsx:15`），深色模式在两套体系间不一致。
- **复现路径**：浏览器控制台执行 `getComputedStyle(document.documentElement).getPropertyValue('--border')` → 返回空；再对任一原生带 border 元素读 `borderColor` → `rgba(0,0,0,0.87)`。
- **建议改法**：要么补全 `:root { --border: 0 0% 87%; }`（与 MUI `divider` 对齐）+ `.dark { --border: 0 0% 100%; }`；要么直接删除这条 `*` 通配规则（当前无任何代码依赖 `--border`，实测 `--border` 全仓库仅此一处引用），改由 MUI `theme.palette.divider` 统一供值。同时统一深色模式入口：让 `muiTheme` 用 `useMediaQuery('(prefers-color-scheme: dark)')` 或 `.dark` class 驱动 `palette.mode`，避免 globals.css 与 MUI 各有一套互不感知的深色变量。
- **截图文件名**：`02-dashboard-1440.png`

#### 12. `globals.css` 全部工具类为死代码，与 MUI 体系并存造成双轨制
- **文件路径:行号**：`frontend/src/app/globals.css:43-155`
- **现象描述**：对 `.chart-container`、`.card-shadow`、`.card-shadow-lg`、`.fade-in`、`.slide-up`、`.status-indicator`（含 4 个状态变体）、`.task-status-pending/running/completed/failed` 逐个全仓库检索（排除 `__tests__`），**引用数全部为 0**。即整个 `globals.css`（除 body 背景/滚动条外）是未使用代码。
  同时项目叠加了 **4 套样式体系**：MUI v7、antd v6、@heroui/react、Tailwind 3 —— Tailwind 的 `content` 扫描 `src/**` 且 `heroui` 插件注入了独立的 `--heroui-*` 变量组（实测 `:root` 同时存在 `--background`、`--foreground`、`--heroui-background`、`--heroui-foreground` 等两组变量）。`globals.css` 的 `body { color: hsl(var(--foreground)); background: hsl(var(--background)); }` 被 MUI `CssBaseline` 覆盖（实测 body 计算色 `rgba(0,0,0,0.87)`、背景 `rgb(245,245,245)` = MUI `background.default`），说明 globals.css 的 body 规则也不生效。
- **复现路径**：`grep -rn "card-shadow\|status-indicator\|task-status-" src/ --include=*.tsx` → 0 结果；DevTools 面板 Elements → `:root` 可同时看到 `--background` 与 `--heroui-background` 两组变量。
- **建议改法**：删除 `globals.css` 中的死工具类（保留 `body` 基础与滚动条即可）；全局样式以 MUI `theme` 为唯一真源 —— spacing 用 `theme.spacing`、radius 用 `theme.shape.borderRadius`、shadow 用 `theme.shadows`；antd 用 `ConfigProvider` + `theme` token 对齐 MUI 主色 `#1976d2`（antd 默认主色 `#1677ff`，与 HeroUI 的 `#006FEE` 又不同，当前是三套主色并存）；Tailwind 若保留则只用于零星 utility，不再自建 token。
- **截图文件名**：`02-dashboard-1440.png`

#### 13. 圆角 / 阴影 / 字号三套并行，无单一真源
- **文件路径:行号**：`frontend/src/theme/muiTheme.tsx:86-88`（`shape.borderRadius: 8`）、`:98-104`（`MuiCard` 硬编码 `boxShadow: '0 2px 8px rgba(0,0,0,0.1)'`）、`frontend/src/theme/mobileOverrides.ts:86-94`（`MuiCard` 移动端 `borderRadius: 12`）、`:131-139`（`MuiPaper` 移动端 `borderRadius: 12`）、`frontend/src/app/globals.css:60-64`（`.card-shadow-lg` 另一套阴影）
- **现象描述**：圆角有 3 个值：MUI `shape.borderRadius = 8`、移动端覆盖 `Card/Paper = 12`、`globals.css` 的 `.chart-container` 用 `12px`。阴影有 3 套：`MuiCard` 硬编码 `0 2px 8px rgba(0,0,0,0.1)`、`globals.css` 的 `.card-shadow`（`0 1px 3px 0 rgb(0 0 0/0.1), 0 1px 2px -1px rgb(0 0 0/0.1)`）与 `.card-shadow-lg`、以及 `login/page.tsx:113` 的 `boxShadow: '0 4px 12px rgba(0,0,0,0.1)'`（登录卡片又是第 4 套）。
  更关键的是 **`error.tsx` / `not-found.tsx` / `loading.tsx` 三个全局边界页完全绕开了 MUI**，全部用内联 `style` 对象写死：实测 404 页 `h1` = `48px / 700 / rgb(211,47,47)`、`h2` = `20px`、`p` = `14px / rgb(102,102,102)`、返回按钮 `padding: 8px 24px; borderRadius: 8px; background: rgb(25,118,210)`。这些硬编码值恰好等于 MUI theme 的当前值，但**换成主题就完全不会跟随**（`muiTheme.tsx` 的 `h1` 是 `2.5rem/40px`，404 页的 48px 与主题的 40px 已经不一致）。同理 `loading.tsx:18` 硬编码 `borderTopColor: '#1976d2'`、`error.tsx:28` 硬编码 `color: '#d32f2f'`。
- **复现路径**：读 404 页计算样式（`getComputedStyle(h1)` → `48px`）并对比 `muiTheme.tsx` 的 `h1: 2.5rem`；在 DevTools 改 `theme.shape.borderRadius` 观察内联样式页面不响应。
- **建议改法**：把 `error.tsx` / `not-found.tsx` / `loading.tsx` 全部改写为 MUI 组件（`Container` + `Paper` + `Typography variant="h1"` + `Button` + `CircularProgress`），删除所有内联 `style` 对象与硬编码色值（`#1976d2` → `primary.main`，`#d32f2f` → `error.main`）；`loading.tsx` 的手写 `@keyframes` 改为 `<CircularProgress/>`。阴影统一进 `theme.shadows`（`shadows[1]` 卡片 / `shadows[8]` 浮层），移除 `MuiCard` 的硬编码 `boxShadow` 覆盖。
- **截图文件名**：`15-notfound.png`、`05-notfound-404.png`、`14-loading-state.png`

#### 14. 移动端断点三套标准不一致（sm=600 / md=900 / lg=1200 混用）
- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:227`（`display: {xs:'block', lg:'none'}` 菜单按钮）、`:248`（`display: {xs:'none', lg:'flex'}` 桌面菜单）、`:339`（`Drawer` 的 `display: {lg:'none'}`）、`:395`（主区 `p: {xs:1.5, sm:2, md:3}`）；`frontend/src/theme/mobileOverrides.ts` 混用 `down('sm')`（第 12/17/22/27/32/46/89/98/103/134/143/153/163/172/185/197 行）与 `down('md')`（第 124 行）；`frontend/src/app/tasks/page.tsx:444`、`:469` 用 `{xs:'block', md:'none'}` / `{xs:'none', md:'block'}`
- **现象描述**：导航壳用 `lg`(1200) 切换桌面/移动、内容区 padding 用到 `md`(900)、`mobileOverrides` 的字号/圆角/对话框覆盖用 `sm`(600)、任务页的表格/卡片切换用 `md`(900)、`MuiTableCell` 压缩用 `down('md')`。结果是同一屏内多套规则各自生效。
  实测可验证的错配：**768–1199px 平板区间用 `md` 断点的抽屉**（实测 768/900/1024/1199px 纸面均 256px），而 `MuiDrawer` 的移动端宽度覆盖只写在 `down('sm')`（<600px），所以 600–1199px 区间拿不到移动端宽度规则 —— 抽屉宽度在 599→600px 从 320 突变到 256（见问题 8）。同时 `mobileOverrides` 的 `MuiTypography` 小字号只在 <600px 生效，但平板 768–1199px 已经进入"移动端导航形态"（汉堡菜单），字号却是桌面尺寸。
- **复现路径**：视宽在 599/600/601px 之间切换并打开抽屉观察宽度跳变；在 768px 与 1200px 分别观察导航形态与正文字号是否匹配。
- **建议改法**：在 `muiTheme.tsx` 显式声明自定义断点（如 `xs:0, sm:600, md:900, lg:1200`）并在全项目统一引用；确立"导航形态断点 = `lg`(1200)、内容密度断点 = `md`(900)、触控密度断点 = `sm`(600)"三层语义，各组件按语义选断点而非就近取值；`mobileOverrides` 里 `MuiDrawer` 的宽度规则从 `down('sm')` 改为 `down('lg')`。
- **截图文件名**：`13-mobile-360-drawer.png`、`19-mobile-390-drawer-open.png`

#### 15. 通知铃铛是死控件，点击无任何反馈
- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:285-289`（`onClick={clearNotifications}` 仅 `setNotifications(0)`）、`:209-211`
- **现象描述**：实测点击铃铛后，DOM 中 `.MuiMenu-root` / `.MuiPopover-root` / `.MuiDrawer-root` / `.MuiSnackbar-root` 数量**全部为 0** —— 不打开任何面板、不弹提示。通知计数 `notifications` 只在收到 `system:alert` 的 warning/error 时自增（`:170-174`），但没有任何 UI 能查看历史通知，点铃铛即静默清零。
  另：该 `IconButton` 无 `aria-label`（实测 18 个顶栏按钮中仅汉堡菜单有 `aria-label="menu"`，其余 17 个 `ariaLabel` 为 `null`），屏幕阅读器无法区分铃铛/头像/15 个菜单按钮。
- **复现路径**：登录 → 顶栏点击铃铛图标 → 观察无任何面板或提示出现。
- **建议改法**：接入 `Menu`/`Popover` 展示通知列表（时间、级别、消息、跳转链接），点击铃铛打开面板而不是清零；清零动作移入面板内「全部已读」。补 `aria-label`：铃铛 `aria-label="通知"`、头像 `aria-label="用户菜单"`、桌面导航容器加 `aria-label="主导航"`（并用 `<nav>` 包裹提升可及性）。
- **截图文件名**：`03-user-menu.png`

#### 16. 页脚年份硬编码 2025，通知状态徽章无实际入口
- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:417`（`股票预测平台 ©2025 - ...`）
- **现象描述**：实测页脚文本为「股票预测平台 ©2025 - 基于AI的智能投资决策系统」，当前系统时间为 2026 年。年份写死不会随时间更新。
  另：品牌名在系统内有三种叫法 —— 导航栏/页脚「股票预测平台」、登录卡「📈 量化研究平台」、`metadata.title`「股票预测平台」、`.env.local` 的 `NEXT_PUBLIC_APP_NAME=股票预测平台`。其中「量化研究平台 / AI 驱动的量化回测与因子分析工具」与产品主名不符（见问题 10）。
- **复现路径**：任意内页滚到底部 → 观察年份；对比登录页卡片标题。
- **建议改法**：年份改为运行时计算（`` `©${new Date().getFullYear()}` ``，或用 `NEXT_PUBLIC_APP_NAME` 拼装）；统一品牌名，删除 `.env.local` 中未被引用的 `NEXT_PUBLIC_APP_NAME`（实测除 `metadata` 硬编码外无代码消费它）或真正接入。
- **截图文件名**：`02-dashboard-1440.png`

---

### P3 优化项

#### 17. 顶栏菜单项无分组、图标重复，16 个入口视觉上完全等权
- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:63-145`
- **现象描述**：`menuItems` 16 项平铺，无分组标题、无分隔线。图标统计（`grep` 计数）：`Signal` 被 4 项复用（策略信号 / 今日信号 / 纸面信号 / 多策略纸面盘）、`Sparkles` 被 3 项复用（策略模板 / 参数调优 / 超参优化）。在 64–93px 的按钮宽度下，用户只能靠 2–4 个汉字辨认，无法靠图标区分。
  信息架构上，"参数调优"（`/tasks/tuning`）是"任务管理"（`/tasks`）的**子页面**却被平级提升为一级导航；"今日信号"（`/today-signal`）、"纸面信号"（`/paper-signal`）、"策略信号"（`/signals`）三个同族功能也各占一个一级入口。
- **复现路径**：观察顶栏（或移动端抽屉）的 15 项 —— 4 个信号塔图标、3 个星形图标相邻排列，无任何分组线索。
- **建议改法**：按业务域重组为 3–4 组（交易决策 / 研究与回测 / 平台运维 / 账户），组内二级项用 `Menu` 或侧边栏可展开段承载；`/tasks/tuning` 降级为「任务管理」下的二级页；三个 signal 页合并为「信号中心」入口页内再分流。为每项配唯一图标（信号类可用 `Signal`/`Radio`/`ChartNoAxesColumn`/`Wallet` 区分）。
- **截图文件名**：`16-nav-1366.png`、`19-mobile-390-drawer-open.png`

#### 18. 主内容区最大宽度 1400px 与顶栏全宽不对齐，宽屏下视觉脱节
- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:396`（`maxWidth: '1400px'`）、`:415`（页脚 `maxWidth: '1400px'`）、`:218-220`（`AppBar` 无宽度约束，`Toolbar` 全宽）
- **现象描述**：1920px 实测：顶栏 Toolbar 宽 **1920px**、`main` 宽 **1400px** 且 `x=260`（居中）。顶栏的 logo 从 x=24 开始，而内容区从 x=260 开始 —— **品牌区与内容区左边界完全不对齐**；页脚（`maxWidth: 1400px`）与主内容对齐，**顶栏是三者中唯一脱离对齐线的**。在 1366/1440px 这类常见宽度下问题被顶栏撑高掩盖，1920px 宽屏才暴露。
- **复现路径**：视口设 1920×900 → 观察顶栏 logo 左缘（x≈24）与下方卡片左缘（x≈260）的错位。
- **建议改法**：给 `AppBar` 内的 `Toolbar` 加与 `main` 一致的 `maxWidth: 1400px; mx: 'auto'`，使顶栏/内容/页脚共用同一栅格左缘。
- **截图文件名**：`07-nav-1920.png`

#### 19. 无面包屑，层级信息完全缺失
- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:213-332`（顶栏内无任何层级指示）
- **现象描述**：全仓库检索 `Breadcrumb` **零命中** —— `AppLayout` 未引入 MUI `Breadcrumbs`，也没有任何自定义面包屑实现。用户从顶栏直接进 `/tasks/tuning` 或 `/paper-portfolio` 时，顶栏只有高亮的平级按钮（且见问题 6/7，部分页面连高亮都没有），无法得知"我在哪个模块的哪一层"。
  现有层级线索只有两个且都不完整：① 顶栏按钮高亮（缺前缀匹配，子路由失焦）；② 各页面自己渲染的 `h1`（如「参数调优台 🎛️」「多策略纸面盘（策略赛马）」）—— 但**顶栏区没有承担层级指示**，页面标题出现在内容区而非导航区，导致「面包屑 + 页面标题」的两段式结构缺失。
- **复现路径**：访问 `/tasks/tuning` → 观察顶栏：仅「参数调优」一个按钮高亮，无"任务管理 > 参数调优"的路径提示。
- **建议改法**：在顶栏左侧（Logo 之后）加 `<Breadcrumbs>`，数据源为 `menuItems` 构成的路由树（父级显式声明，如 `/tasks/tuning` 的 `parent: '/tasks'`）；配合问题 6 的前缀匹配规则，面包屑与高亮可共用同一棵树。`<nav aria-label="面包屑">` + 末级加 `aria-current="page"`。
- **截图文件名**：`02-dashboard-1440.png`

#### 20. 顶栏按钮组无键盘可达性与语义化包裹
- **文件路径:行号**：`frontend/src/components/layout/AppLayout.tsx:245-271`
- **现象描述**：桌面菜单是一个无语义 `<Box>`（`display:flex`），15 个 `<Button>` 直接平铺，无 `<nav>` / `role="navigation"` 包裹，无 `aria-label`。Tab 键遍历时焦点会在 15 个导航按钮间逐个停留（含移动端隐藏的 `xs:'none'` 组 —— 该组在 `lg` 以下 `display:none`，因此不可聚焦，行为正确），但屏幕阅读器无法把它们识别为导航组。
  另：桌面菜单与移动端抽屉是**两份独立的 `menuItems.map` 渲染**（`:253-270` 和 `:357-385`），高亮逻辑（`pathname === item.key`）重复了两份 —— 修一处必漏另一处，问题 6 的高亮失效正是因为这两份都用了错误的 `'/'` key。
- **复现路径**：用键盘 Tab 遍历顶栏 → 焦点逐个落在 15 个按钮上；用 DevTools Accessibility 面板查看顶部区域，无 navigation landmark。
- **建议改法**：桌面菜单与抽屉菜单抽成同一个 `<NavMenu variant="horizontal|vertical" />` 组件（同一份高亮逻辑、同一份 `menuItems`），外层包 `<nav aria-label="主导航">`；给当前页按钮加 `aria-current="page"`。
- **截图文件名**：`02-dashboard-1440.png`

---

## 截图索引

| 文件 | 尺寸 | 说明 |
|---|---|---|
| `01-login.png` | 1440×900 | 登录页 —— 顶部带完整已登录导航（问题 10） |
| `02-dashboard-1440.png` | 1440×900 | 1440px 仪表板 —— 顶栏 86px、标签 2 行；无菜单高亮（问题 1/6） |
| `03-user-menu.png` | 1440×900 | 用户下拉菜单 —— 个人资料/账户设置/退出登录（问题 3/5） |
| `04-profile-404.png` | 1440×900 | `/profile` → 404（问题 5） |
| `05-notfound-404.png` | 1440×900 | 404 页 —— 内联样式硬编码（问题 13） |
| `06-nav-1280-overflow.png` | 1280×900 | 1280px —— 顶栏 159px、4 行标签、横向溢出（问题 1/2） |
| `07-nav-1920.png` | 1920×900 | 1920px —— 顶栏与内容区左边界错位（问题 18） |
| `10-mobile-admin-payments-overflow.png` | 390×844 | `/admin/payments` 390px 横向溢出（问题 7） |
| `11-mobile-dashboard.png` | 390×844 | 390px 仪表板 |
| `13-mobile-360-drawer.png` | 360×640 | 抽屉 —— 底部「系统设置」被裁切（问题 9） |
| `14-loading-state.png` | 1440×900 | 加载态 —— 内联样式硬编码（问题 13） |
| `15-notfound.png` | 1440×900 | 404 页全页 |
| `16-nav-1366.png` | 1366×900 | 1366px —— 顶栏 159px、logo 6 字竖排（问题 1/2） |
| `17-nav-1200.png` | 1200×900 | 1200px（lg 断点）—— 顶栏 159px（问题 1） |
| `18-mobile-390-topbar.png` | 390×844 | 390px 顶栏 |
| `19-mobile-390-drawer-open.png` | 390×844 | 390px 抽屉 —— 右侧 64px 死白（问题 8） |

---

## 整体布局评估结论

**不合格，需重构全局导航壳（AppLayout + 根布局），不是局部打磨能解决的。**

1. **导航壳形态本身有问题**：16 个一级入口平铺进顶栏按钮组，无侧边栏、无分组、无面包屑。这在 1200–1440px 直接产出"顶栏 159px 高 + 中文标签竖切 4 行 + logo 压成 6 个单字竖排 + 页面横向溢出"——在 1366×768 这个最主流的笔记本尺寸上，导航区本身就吃掉 20% 视口高度。根因是**信息架构未分层**（`/tasks/tuning` 这类子页面被提升为一级项，三个 signal 页各占一级入口），叠加**实现方式未适配**（无 `whiteSpace: nowrap`、无 `flexShrink: 0`、无 `minWidth` 保护）。正确解法是引入 `Drawer variant="permanent"` 侧边栏 + 分组折叠 + 面包屑，顶栏只保留品牌区 + 面包屑 + 账号区。

2. **导航壳之外有三个 P0 功能缺陷**：退出登录删错键名导致会话未真正登出（`auth_token` vs `access_token`，`AppLayout.tsx:202`）；zustand store 无 persist 导致管理员入口刷新即消失（`useAppStore.ts:52`）；`/profile` 与 `/account-settings` 两条菜单项路由不存在、点击必 404（`AppLayout.tsx:195,198`）。这三条与布局无关，但同属"全局导航壳"职责范围，且用户感知上比视觉问题更严重。

3. **设计体系四轨并行、无单一真源**：MUI v7 / antd v6 / @heroui/react / Tailwind 3 同时在跑，主色就有 `#1976d2` / `#1677ff` / `#006FEE` 三个值；`globals.css` 的 10 个工具类实测**零引用**且其中 `--border` 变量从未定义导致 `* { border-color }` 规则整体失效（回落到近黑 `rgba(0,0,0,0.87)`）；圆角 8/12 与阴影 4 套并行；`error.tsx`/`not-found.tsx`/`loading.tsx` 三个全局边界页完全绕开 MUI 用内联硬编码色值，主题一变就跟不上。

4. **响应式断点三套标准混用**（sm/md/lg 各自为政），已产生可复现的错配：抽屉宽度在 599→600px 从 320 突降到 256；768–1199px 平板区已用移动端导航形态却拿不到移动端字号与抽屉宽度规则。

5. **可访问性有基础缺口**：无面包屑、无 `<nav>` 语义包裹、17/18 个顶栏按钮无 `aria-label`、铃铛是死控件。

建议顺序：先修 5 个 P0（会话与死链，影响功能正确性）→ 引入侧边栏重构导航壳并落地面包屑与分组（问题 1/2/6/17/19/20）→ 统一设计 token 与断点（问题 11/12/13/14）→ 补可访问性与细节（问题 15/16/18）。

## 先修清单（按 P0/P1 排序，12 条）

| # | 严重度 | 问题 | 文件:行号 | 一句话改法 |
|---|---|---|---|---|
| 1 | **P0** | 退出登录不清 token，会话未真正登出 | `AppLayout.tsx:200-204` | 改用 `api.ts` 的 `clearStoredAuth()` 删 `access_token`/`refresh_token`/`user_info`，并 `setUser(null)` + `router.replace('/login')` |
| 2 | **P0** | 管理员入口「充值订单」刷新后消失 | `AppLayout.tsx:143,254,358` + `stores/useAppStore.ts:52` | 给 store 加 zustand `persist`（或 mount 时从 `localStorage.user_info` 反序列化 `setUser`），补 `is_admin` 映射 |
| 3 | **P0** | `/profile`、`/account-settings` 两条菜单必 404 | `AppLayout.tsx:195,198` | 新建两个真实页面，或本期先移除/`disabled` 这两个 `MenuItem` |
| 4 | **P0** | 顶栏 16 项平铺，1200–1440px 顶栏 159px、标签竖切 4 行、横向溢出 | `AppLayout.tsx:245-271, 63-145` | 改 `Drawer variant="permanent"` 侧边栏 + 分组；过渡期至少加 `whiteSpace:'nowrap'` + `flexShrink:0` + `lg` 断点上移 |
| 5 | **P0** | Logo「股票预测平台」压成 6 个单字竖排 | `AppLayout.tsx:233-242` | Logo `Typography` 加 `whiteSpace:'nowrap'; flexShrink:0` + 外层 `minWidth`，或 `lg` 以下只留图标 |
| 6 | **P1** | 「仪表板」key 是 `/` 而真实路由 `/dashboard`，高亮永不生效 | `AppLayout.tsx:65,257,361` | key 改 `/dashboard`；`isActive` 改前缀匹配 `pathname === key \|\| pathname.startsWith(key+'/')` |
| 7 | **P1** | 5 条页面无导航入口（wallet/subscription/tasks-create/tasks-compare/tasks-id） | `AppLayout.tsx:63-145` | 钱包并入用户菜单、订阅并入设置分组、创建任务做任务管理子项、对比挂任务详情页；`/admin` 补索引页 |
| 8 | **P1** | 移动抽屉 320px 面板内容硬编码 256px，右侧 64px 死白 | `AppLayout.tsx:341` + `mobileOverrides.ts:194-203` | 删固定 `width:256` 改 `width:'100%'`；Drawer 覆盖断点 `down('sm')` → `down('lg')` |
| 9 | **P1** | 360×640 抽屉底部「系统设置」被裁切且无法滚动 | `AppLayout.tsx:335-388` | `Drawer` 加 `PaperProps={{sx:{overflowY:'auto'}}}`，菜单区 `flex:1` 自滚，头部 sticky |
| 10 | **P1** | 登录/注册/定价页套用完整已登录外壳 | `app/layout.tsx:24-29` | 拆 `(app)` 路由组承载 `AppLayout`，公开页移出；统一品牌名 |
| 11 | **P2** | `--border` 未定义 → `* { border-color }` 整条失效回落到近黑 | `globals.css:15-17` | 补 `--border` 或删除该通配规则，统一由 `theme.palette.divider` 供值 |
| 12 | **P2** | 三个全局边界页用内联硬编码色值绕开 MUI 主题 | `app/error.tsx:28,39` `app/not-found.tsx:16-27` `app/loading.tsx:18,24` | 全部改写为 MUI 组件（`Typography variant="h1"`/`Button`/`CircularProgress`），删内联 `style` 与硬编码色 |

### 建议同期处理（不占用先修额度）

- P1 #15 通知铃铛死控件 + 17 个按钮缺 `aria-label`；P1 #19 补面包屑；P2 #12 清理 `globals.css` 死代码与四套样式体系；P2 #14 统一断点语义；P3 #16 页脚年份硬编码 2025；P3 #18 顶栏与内容区左边界错位；P3 #20 抽 `<NavMenu>` 消除桌面/移动两份重复渲染。
