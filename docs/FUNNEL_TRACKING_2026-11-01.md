# 真实用户漏斗埋点（2026-11-01）

> 状态：**已上线**（后端埋点 + 前端埋点 + 报告 CLI/API 全通）
> 关联：台账 `MULTI_ASSET_LEDGER_2026-10-31.md` §12.13 平台策略全貌（运营闭环缺口）

---

## 一、为什么做这件事

平台当前的真实状态（2026-11-01 核过）：

| 项 | 数值 |
|---|---|
| 注册用户 | 23 个，**全是测试账号** |
| 真实用户 | **0** |
| 真实收入 | **¥0** |
| 纸面盘净值点 | 0（定时任务未加载） |

策略侧已经挖得很深（因子/暴露/执行/基准/换源全部证伪，见台账 §八），
但**"用户卡在哪一环"完全没有数据**。没有埋点时，
"用户注册了不看信号""看了信号不去付费""点了升级没付钱"这三种流失无法区分，
只能靠猜 —— 等真用户进来再补埋点，就连"漏在哪一步"都答不出来。

所以这次补的是**运营闭环的度量能力**，不是新的策略。

---

## 二、事件注册表（唯一事实源）

**同一个事件名只能有一个来源**，否则报告会重复计数。前后端各埋一半，但互不重叠：

| 事件 | 来源 | 触发时机 | 为什么归这一边 |
|---|---|---|---|
| `signup` | **后端** | `/auth/register` 成功 | 服务端才知道的事实；前端报错就丢了 |
| `login` | **后端** | `/auth/login` 成功 | 同上 |
| `quota_exhausted` | **后端** | 配额打满抛 429 前 | 真实付费墙（用户想用但用不了），只有服务端知道 |
| `first_screen` | 前端 | dashboard 挂载 | UI 触点 |
| `view_signal` | 前端 | 今日信号 / 纸面信号**真正加载出信号后** | 不是"进了页面"，是"看到了内容" |
| `view_pricing` | 前端 | 价格页挂载 | 付费墙触点（看价 ≠ 点升级） |
| `click_upgrade` | 前端 | 点套餐 CTA / 点"去钱包充值" | 升级意图 |
| `wallet_open` | 前端 | 钱包页挂载 | 走到付费动作 |

代码位置：
- 后端注册表：`backend/app/services/funnel/funnel_service.py`（`ALLOWED_EVENTS`）
- 前端类型：`frontend/src/services/funnel.ts`（`FunnelEventName`）

**改事件名必须同时改这两处**，否则前端埋点会被后端静默丢弃（返回 `ok=false`）。

---

## 三、数据模型

`backend/app/models/funnel_models.py` → 表 `funnel_events`（只追加）

| 列 | 说明 |
|---|---|
| `user_id` | 已识别用户；注册前事件为 NULL（外键 → users.id，删号置 NULL） |
| `anon_id` | 匿名设备标识（前端 localStorage），**归并后仍保留**以便追溯 |
| `event` | 事件名 |
| `props` | JSON 字符串，>2KB 截断为 `{"_truncated":true}` |
| `session_id` | 标签页会话标识 |
| `source` | `backend` / `frontend` |
| `created_at` | naive UTC（与项目其余表一致） |

### 匿名 → 已识别的归并

注册前的行为（看了价格页、点了升级）也要能连到注册后的账号上。
机制：前端把 `anon_id` 一起带上（注册请求体 + 每次埋点请求），
后端在**首次带身份**的埋点到达时，把该 `anon_id` 下所有 `user_id IS NULL` 的事件
归并到该用户。

→ E2E 已验证：匿名 `view_pricing` → 注册 → 归并后
`anonymous_users=0`、`paywall.view_pricing_users=1`、顺序转化不丢。

---

## 四、报告口径

`funnel_service.build_report(session, days=30)` —— CLI 与 API **共用同一个函数**，
口径不会漂移。

### 4.1 漏斗主链路（顺序转化）

```
注册 → 首屏 → 首条信号 → 付费墙触点 → 升级点击
```

| 字段 | 含义 |
|---|---|
| `users` | **触达**人数：窗口内触发过该事件的人数（不要求完成前序步骤） |
| `converted` | **顺序转化**人数：触发了该事件**且**完成了前面所有步骤的人数 |
| `conv_from_prev` | 本步 `converted` / 上一步 `converted` |
| `conv_from_start` | 本步 `converted` / 注册人数 |

两个数都要看：`users` 看"这一步有多少人摸到"，`converted` 看"这条路走到底的还剩几个"。
只报一个会得出相反的结论。

### 4.2 留存

- cohort = 窗口内有 `signup` 事件、且**有真实 user_id** 的用户（匿名不算）
- `d1` = 注册**次日**有任意事件的用户占比
- `d7` = 注册后 1~7 天内有任意事件的用户占比

### 4.3 付费墙（单列，不进漏斗主链路）

`view_pricing_users` / `click_upgrade_users` / `wallet_open_users` / `quota_exhausted_users`。

`quota_exhausted` 是**服务端事实**（回测次数打满），它比"看了价格页"更接近真实付费意愿，
所以单列统计，不混进前端触点。

---

## 五、怎么读

```bash
# CLI（不需要管理员 token，后端没跑也能读）
cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/funnel_report.py
cd backend && env -u PYTHONPATH .venv-py313/bin/python3 scripts/funnel_report.py --days 7 --json

# API（仅管理员）
GET /api/v1/funnel/report?days=30
```

`POST /api/v1/funnel/track` 是**公开**端点（注册前也要能埋），带 token 时自动归并身份；
report 仅管理员（403），未登录 401。

---

## 六、埋点铁律

1. **埋点永远不能影响业务**：所有异常吞掉；不重试、不排队、不缓存到内存。
   丢一条埋点远好过拖慢首屏。
2. **页面挂载类事件每会话只上报一次**（`trackEventOnce`）——
   React StrictMode 在开发态会双跑 effect，不去重会直接翻倍。
3. **服务端事实只由服务端记**（注册/登录/配额），前端不重复上报。
4. **匿名行为必须带 `anon_id`**，否则归并断链，注册前路径全丢。

---

## 七、验证证据（2026-11-01 实跑）

| 项 | 命令 | 结果 |
|---|---|---|
| 后端单测 | `pytest tests/unit/test_funnel_service.py -q` | **6 passed**（写入/拒收/归并/顺序漏斗/D1-D7/截断清洗） |
| 后端 E2E | `python3 scripts/test_funnel_e2e.py` | **27 项全过**（真起 uvicorn + 真注册 + 真埋点 + 真读报告 + 权限 403/401 + CLI/API 口径一致） |
| 前端单测 | `npx jest src/__tests__/funnel.test.ts` | **6 passed**（标识持久化/请求体/Authorization/once 语义/失败不抛错） |
| 前端类型 | `npm run type-check` | 通过 |
| 前端构建 | `npm run build` | 通过 |

E2E 用独立临时库（`~/.hermes/cache/scratch/funnel_e2e.db`），不碰线上 `app.db`。

---

## 八、已知边界

- **报表只覆盖 web 端**。微信/QQ 渠道的访问不经过 Next.js 前端，不产生埋点。
- **触点粒度是页面级**，没有做点击热图；要看"用户在信号页停留多久"需要另加计时埋点。
- **事件是流水表，不做去重**（除 anon_id 归并）。同一用户多次看信号会留多条，
  报告按**人数**统计（distinct user），不受影响；按次数分析要自己数 `daily`。
- **当前漏斗只有 1 个真实用户也没有数据**：23 个测试账号是 2026-10 之前建的，
  埋点上线后**不会**回填历史行为。要看漏斗形状，需要**新的真实用户**。
