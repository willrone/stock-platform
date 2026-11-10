# 获客与付费转化页前端审计报告

**审计范围**：`frontend/src/app/` 下 `page.tsx`（落地页）、`login/`、`register/`、`reset-password/`、`pricing/`、`settings/subscription/`、`wallet/`，以及 `frontend/src/services/commerce/`
**审计方式**：`browser_exec` 真实浏览器渲染（登录态 + 未登录态），配合 CDP 几何测量与后端契约探测
**环境**：前端 http://localhost:3000，后端 http://127.0.0.1:18082，账号 ronghui（enterprise 档、is_admin=1）
**日期**：2026-10-27
**性质**：只读审计，未修改任何源码。审计中产生的测试用户 / 测试订单已从 `backend/data/app.db` 清理还原。

---

## 结论速览

这组页面的**文案与信息架构做得比预期好**（落地页的"讲难听的那部分"、定价页的逐项对照表，都是有真内容的，不是 AI 模板）。但**整条付费转化链路在运行时是断的**：

> 落地页 → 注册 → 定价 → 支付 → 钱包 → 充值 → 后台确认到账 → 套餐开通

这条链路上有 **6 个 P0**，其中任意一个单独存在都足以让付费转化归零：

1. 钱包页 4 个核心接口全部因响应封装不匹配而失败，余额恒显示 0.00
2. 钱包页「充值」按钮是个空操作（点击隐藏表单）
3. 充值表单的金额校验规则永远不通过，**没有任何金额能提交**
4. 忘记密码 / 重置密码两个模式**完全不可用**（运行时 TypeError）
5. 定价页的支付请求 plan_id 契约错误，付费档位**永远调不通**
6. 后台「确认到账」接口路径错误，管理员**永远无法确认收款**（钱收了，套餐不会开通）

**"收了钱套餐不变"是这套代码里被明确修过一次的 bug**（`payment_service.py` 里写着「否则『收了钱但套餐没变』」），但前端两处调用把这条链路又切断了。

---

## 问题清单

### P0 — 阻断级（付费转化链路上必有一环断掉）

---

#### P0-1｜钱包页所有数据加载失败，余额恒显示 0.00 元

| 项 | 内容 |
|---|---|
| **严重度** | **P0** |
| **位置** | `frontend/src/app/wallet/page.tsx:120-134`（调用方）<br>`frontend/src/services/api.ts:193-201`（拦截器根因）<br>`frontend/src/services/api.ts:306-314`（`apiRequest.get` 根因） |
| **截图** | `06-wallet-loggedin.png` |

**现象描述**

钱包页 4 个核心接口全部请求失败。实际 API 返回余额 10.00 元（`balance_yuan: 10`），页面却显示「当前余额 **0.00 元**」「账户状态 **余额不足**」「本月累计支出 0.00 元」，并弹出「加载数据失败」toast（连续弹两次）。

**根因（已精确定位）**

`/commerce/*` 系列端点返回的是**裸 JSON**（`{user_id, balance_cents, balance_yuan, ...}`），而后端只有**错误路径**被 `ErrorHandlingMiddleware` 包成 `{success, message, data, timestamp}`（`backend/app/middleware/error_handling.py:68`）。前端 `api.ts:194` 的响应拦截器写的是：

```ts
if (!data.success) {
  return Promise.reject(new Error(data.message || '请求失败'));
}
```

裸 JSON 没有 `success` 字段 → `!data.success` 为 true → **所有成功响应被当成业务失败丢弃**。实测拦截器行为：

```
rawKeys: ['user_id','balance_cents','balance_yuan','credited_cents','total_spent_cents','updated_at']
hasSuccess: false  →  interceptorWouldReject: true  →  interceptorMessage: '请求失败'
```

`apiRequest.get` 随后再取 `res.data.data`（也是 undefined）形成第二道拦截。**双重失败**。

**受影响的端点（实测全部 200 但被前端丢弃）**：
`/commerce/balance`、`/commerce/usage/summary`、`/commerce/billing/history`、`/commerce/payment/orders`、`/commerce/payment/config`、`/commerce/pricing/rules`、`/billing/subscription`、`/billing/usage`

> 注：`/api/v1/dashboard/stats`、`/api/v1/billing/checkout`、`/api/v1/commerce/deposit` 是**有**封装的（走的是别的响应路径），所以表现不一致 —— 这正是这个 bug 难被发现的原因。

**复现步骤**
1. 登录 ronghui
2. 访问 http://localhost:3000/wallet
3. 观察：余额「0.00 元」、状态「余额不足」、右下角弹「加载数据失败」×2
4. 对照：DevTools 里 `GET /api/v1/commerce/balance` 实际返回 `{"balance_cents":1000,"balance_yuan":10,...}`

**具体改法**

三选一（推荐 A，改动面最小且能一次性修好全部 8 个端点）：

- **A（改前端拦截器，兼容裸 JSON）**：`api.ts:193-201` 改成
  ```ts
  // 兼容裸 JSON 与 {success,data} 两种后端契约
  const enveloped = data && typeof data === 'object' && 'success' in data;
  if (enveloped && !data.success) {
    return Promise.reject(new Error(data.message || '请求失败'));
  }
  ```
  同时 `apiRequest.get` 的取值改为 `enveloped ? res.data.data : res.data`，`apiRequest.post` 同理（`api.ts:331-336`）。**注意 `data.data === undefined` 时当前实现会返回 `null` 并丢弃真实空值**，也要一起改。
- **B（改后端）**：在 `ErrorHandlingMiddleware.dispatch` 里对**成功响应**也做统一包装（`backend/app/middleware/error_handling.py:27-30`）。这是更干净的做法，但要回归全部端点。
- **C（改 commerce 端点）**：给 `/commerce/*` 单独加 `response_model=StandardResponse[...]` 包装。改动分散，不推荐。

---

#### P0-2｜钱包页「充值」按钮是空操作，点击无任何反应

| 项 | 内容 |
|---|---|
| **严重度** | **P0** |
| **位置** | `frontend/src/app/wallet/page.tsx:284-292`（按钮）<br>`frontend/src/app/wallet/page.tsx:356-390`（被点击的表单） |
| **截图** | `07-wallet-deposit-click.png` |

**现象描述**

钱包页右上角「充值」是全页唯一的转化 CTA。点击后**没有任何视觉反馈**：没有弹窗、没有表单展开、没有滚动、没有任何提示。用户会以为按钮坏了而反复点击。

**根因**

按钮的 onClick 是：
```tsx
onClick={() => document.getElementById('deposit-form')?.click()}
```
它 click 的是一个 `<Form>` 元素（不是 submit 按钮），而这个 Form 被 `style={{ display: 'none' }}` 隐藏（`wallet/page.tsx:361`）。实测：

```
deposit-form: {tag: 'FORM', display: 'none', visible: false, rect: 0x0}
点击「充值」后 → {display:'none', visible:false, modalOpen:false, modalText:null}
```

对 `<form>` 元素调 `.click()` 不会触发表单提交（form 本身没有默认提交行为），且它被 `display:none` 隐藏着 —— 即使触发了也看不见。

**复现步骤**
1. 登录后访问 /wallet
2. 点击「账户状态」卡片里的蓝色「充值」按钮
3. 页面零变化

**具体改法**

把充值做成受控的 Modal（antd `Modal` + `Form`，`open` state 驱动），删除那个 `display:none` 的隐藏 Form：

```tsx
const [depositOpen, setDepositOpen] = useState(false);
// 按钮
<Button type="primary" block icon={<CreditCard size={16}/>}
  onClick={() => setDepositOpen(true)}>充值</Button>
// 弹窗
<Modal open={depositOpen} onCancel={() => setDepositOpen(false)}
  title="充值" footer={null} destroyOnClose>
  <Form form={depositForm} onFinish={handleDeposit} layout="vertical">
    ...
  </Form>
</Modal>
```

顺带修 P0-3 时，把金额校验也一起放进这个 Modal。

---

#### P0-3｜充值金额校验规则永远不通过，任何金额都提交不了

| 项 | 内容 |
|---|---|
| **严重度** | **P0** |
| **位置** | `frontend/src/app/wallet/page.tsx:363-379` |
| **截图** | `16-wallet-deposit-submit.png` |

**现象描述**

即使把 P0-2 的按钮改成能打开表单，金额也**填不进去**。输入 10（等于下限）、88、100、500、1000、10.5，全部报同一个错误：

> 充值范围 ¥10 ~ ¥5000

这个提示是**误导性的** —— 88 / 100 / 500 明明在 ¥10~¥5000 区间内。

**根因**

规则写的是 `{ type: 'number', min, max, message }`（`wallet/page.tsx:368-375`），但 antd v6 的 `<Input type="number">` 表单值是**字符串**（DOM input 的 value 永远是 string）。而 antd v6 依赖的 `@rc-component/async-validator@6.0.0` 的 `types.number` 是：

```js
// node_modules/@rc-component/async-validator/lib/rule/type.js:52
number(value) {
  if (isNaN(value)) { return false; }
  return typeof value === 'number';   // ← 字符串必然 false
}
```

`type: 'number'` 先因 `typeof value === 'number'` 不成立而失败；由于 min/max 与 type 写在**同一条规则对象**里，共用同一条 `message`，最终展示出来的就是「充值范围」这句 —— 真实原因（类型不匹配）被错误文案掩盖了。

**复现步骤**
1. 登录后访问 /wallet
2. DevTools 里把 `#deposit-form` 的 `display` 改成 `block`（绕过 P0-2 的隐藏问题）
3. 在金额框输入 `88`，点「创建充值订单」
4. 唯一错误：`充值范围 ¥10 ~ ¥5000`（88 明明合法）
5. 换成 `10` / `500` / `123456789012` —— 全部同一个错误

**具体改法**

用 antd 的 `InputNumber`（值是真正的 number），或保留 `Input` 但加 `normalize` 把字符串转数字：

```tsx
<Form.Item
  name="amount"
  label="充值金额（元）"
  rules={[
    { required: true, message: '请输入充值金额' },
    {
      type: 'number',
      min: (payConfig?.min_deposit_cents ?? 1000) / 100,
      max: (payConfig?.max_deposit_cents ?? 500000) / 100,
      message: `充值范围 ¥${...} ~ ¥${...}`,
    },
  ]}
  normalize={(v) => (v === '' || v === null ? undefined : Number(v))}
>
  <InputNumber min={10} max={5000} step={10} addonAfter="元" style={{width:'100%'}} />
</Form.Item>
```

同时**把 type 与 min/max 拆成两条规则**，让类型错误和范围错误能分别给出正确文案。

---

#### P0-4｜忘记密码 / 重置密码两个模式完全不可用（运行时 TypeError）

| 项 | 内容 |
|---|---|
| **严重度** | **P0** |
| **位置** | `frontend/src/app/reset-password/page.tsx:28`（`handleRequestReset`）<br>`frontend/src/app/reset-password/page.tsx:51`（`handleResetPassword`）<br>`frontend/src/app/reset-password/page.tsx:121,153`（`onFinish` 绑定） |
| **截图** | `19-reset-password-request-sent.png`、`20-reset-password-token-mode.png` |

**现象描述**

「发送重置链接」点了没反应；带 `?token=` 的重置模式点「重置密码」同样没反应。**两个模式都是死的**，且控制台无任何用户可见的错误提示 —— 页面看起来像"点了没生效"。

**根因**

两个 handler 都声明为 `async (e: React.FormEvent<HTMLFormElement>)` 并调用 `e.preventDefault()`，但它们被绑定到 antd `<Form onFinish={...}>`（`page.tsx:121` 和 `page.tsx:153`）。**antd 的 `onFinish` 回调第一个参数是 `values` 对象，不是 DOM 事件**。于是 `e.preventDefault` 是 `undefined` → 抛 `TypeError`。

同时，handler 内部用 `new FormData(e.currentTarget)` 取值 —— `e.currentTarget` 在 antd 语境下也是 undefined，即使 `preventDefault` 不报错也取不到任何字段值。

**实测（两种模式都复现）**

```
forgot 模式:  unhandledrejection: e.preventDefault is not a function
reset 模式:   unhandledrejection: e.preventDefault is not a function
按钮 loading 状态: {text:'发送重置链接', disabled:false, hasSpin:false}  ← handler 根本没跑起来
```

对比：后端接口是好的，**直接调 API 能拿到 token**：
```
POST /api/v1/auth/forgot-password {"email":"ronghui@example.com"}
→ 200 {"message":"如果该邮箱存在，密码重置链接将发送到邮箱","reset_token":"k8kGK9emG..."}
```

**复现步骤**
1. 访问 http://localhost:3000/reset-password
2. 输入 `ronghui@example.com`，点「发送重置链接」
3. 页面无变化（无成功提示、无错误提示）；Console 有 `e.preventDefault is not a function`
4. 访问 http://localhost:3000/reset-password?token=xxx 填两个密码提交 → 同样无反应

**具体改法**

改成 antd `onFinish` 的正确签名，直接用 `values`：

```tsx
const handleRequestReset = async (values: { email: string }) => {
  setRequestLoading(true); setRequestError('');
  try {
    const res = await fetch(`${API_BASE}/api/v1/auth/forgot-password`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ email: values.email }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || '发送失败');
    // 开发/测试环境下后端会直接返回 token，提示用户以便调试
    setRequestSent(true);
  } catch (err) { setRequestError(err.message || '发送失败，请稍后重试'); }
  finally { setRequestLoading(false); }
};
```
`handleResetPassword` 同理改成 `async (values: {password: string; confirmPassword: string})`，并把两个 `Form.Item` 的 `dependencies={['password']}` 校验（register 页已经这么写了，reset 页漏了）补上。

> 顺带：`register/page.tsx:79` 的 `submit(values)` 用法是对的，**register 页没有这个问题** —— 说明这是 reset-password 页的孤立笔误。

---

#### P0-5｜定价页付费档位的支付请求 plan_id 契约错误，永远调不通

| 项 | 内容 |
|---|---|
| **严重度** | **P0** |
| **位置** | `frontend/src/app/pricing/page.tsx:172-175`（前端传参）<br>`backend/app/services/billing/stripe_service.py:80-92`（后端查表） |
| **截图** | `10-pricing-free-cta-loggedin.png` |

**现象描述**

登录用户在定价页点「开通信号版」（¥199），页面弹出：

> ⚠️ 自动支付通道暂未开通，请到「钱包 → 充值」下单后联系客服确认。

这个提示**掩盖了真实错误**。真实错误是 400「套餐不存在或已停用」，而且是**前端传参错误**，不是 Stripe 没配置。

**根因**

前端传的是**档位名** `plan.id ∈ {free, pro, enterprise}`（`pricing/page.tsx:174`），后端查的是**主键 UUID**：

```python
# backend/app/services/billing/stripe_service.py:84
plan = session.query(SubscriptionPlan).filter(
    SubscriptionPlan.id == plan_id,          # ← 比的是 id（UUID），不是 name
    SubscriptionPlan.is_active.is_(True),
).first()
```

而 `subscription_plans` 表里 `id` 是 UUID（`f5f643b9-...`），`name` 才是 `free/pro/enterprise`。实测对照：

| plan_id 传入 | 结果 |
|---|---|
| `'pro'` | 400 `套餐不存在或已停用` |
| `'enterprise'` | 400 `套餐不存在或已停用` |
| `'f5f643b9-38e8-441c-a579-91664963dcb7'`（真 UUID） | 400 `该套餐尚未配置有效的 Stripe Price ID` ← 换对参数后才走到真正的 Stripe 报错 |

也就是说，**即使把 Stripe 配好，前端也依然调不通**。这个 bug 会一直存在到有人手工修好。

**复现步骤**
1. 登录任意用户，访问 /pricing
2. 点「开通信号版」
3. 弹出「自动支付通道暂未开通…」（误导性提示）
4. DevTools 可见 `POST /api/v1/billing/checkout` 返回 400 `套餐不存在或已停用`

**具体改法**

前端改为先查后端拿真实 plan id，或后端接受 name。推荐后端加一层兼容（更稳，两种传法都支持）：

```python
# backend/app/services/billing/stripe_service.py:_get_plan
plan = (
    session.query(SubscriptionPlan)
    .filter(
        or_(
            SubscriptionPlan.id == plan_id,
            SubscriptionPlan.name == plan_id,   # ← 兼容 'pro'/'enterprise'
        ),
        SubscriptionPlan.is_active.is_(True),
    )
    .first()
)
```

同时**前端应把真实错误透出**，不要一律显示「通道暂未开通」：

```tsx
} catch (e) {
  setError(e instanceof Error ? e.message : '自动支付通道暂未开通，请到「钱包 → 充值」下单后联系客服确认。');
}
```

---

#### P0-6｜后台「确认到账」接口路径错误，管理员永远无法确认收款

| 项 | 内容 |
|---|---|
| **严重度** | **P0** |
| **位置** | `frontend/src/app/admin/payments/page.tsx:84,116,129,143`（4 处 URL 缺前缀）<br>`backend/app/api/v1/commerce.py:281,292,330,356`（真实路径，带 `/commerce` 前缀） |
| **截图** | `25-admin-payments.png` |

**现象描述**

管理员进「充值订单」页，**列表永远是空的**（显示「暂无充值订单」、「待确认订单 0」、「待确认金额 ¥0.00」），尽管数据库里有 3 条已付订单和（当时的）1 条待确认订单。用户转账了，管理员在这个页面上**看不到订单、点不了「确认到账」**，套餐永远不会开通。

**根因**

后端 commerce 路由带 `/commerce` 前缀（`backend/app/api/v1/commerce.py:29`），但前端 admin 页调的是 `/admin/payment/orders`（少了 `/commerce`）。实测：

```
GET /api/v1/admin/payment/orders?limit=3          → 404 Not Found
GET /api/v1/commerce/admin/payment/orders?limit=3  → 200 {orders:[...], count:N}   ← 正确路径，有数据
POST /api/v1/admin/payment/orders/PO-XXX/confirm  → 404 Not Found
```

同页第 95 行 `apiRequest.get('/commerce/payment/config')` 用了正确前缀，所以「收款方式配置」区块能显示 —— 这就更迷惑了：同页面一半通一半不通。

**复现步骤**
1. 用 is_admin=1 的账号（如 ronghui）登录
2. 访问 http://localhost:3000/admin/payments
3. 显示「暂无充值订单」，与数据库实际条数不符
4. DevTools：`GET /api/v1/admin/payment/orders` → 404

**具体改法**

4 处 URL 全部补前缀：

```tsx
const res = await apiRequest.get<{ orders: Order[] }>(`/commerce/admin/payment/orders${q}`);
await apiRequest.post(`/commerce/admin/payment/orders/${orderNo}/confirm?remark=后台确认到账`);
await apiRequest.post(`/commerce/admin/payment/orders/${orderNo}/cancel?remark=管理员取消`);
await apiRequest.post('/commerce/admin/payment/config', { ... });
```

> 这条 P0 与 P0-1 一起构成"收了钱套餐不变"的完整链路：用户提交订单 → 管理员在后台**看不到**这单 → 永远不确认 → 订单永远 pending → 套餐永远不升级。**即使 P0-3 修好了、用户能下单，钱也进不了系统。**

---

### P1 — 严重（影响转化效率或数据可信度）

---

#### P1-1｜定价页三个档位的 CTA 全部显示「免费注册」，¥199/¥599 档没有付费语义

| 项 | 内容 |
|---|---|
| **严重度** | **P1** |
| **位置** | `frontend/src/app/pricing/page.tsx:341-347`（按钮文案分支） |
| **截图** | `02-pricing-loggedout.png` |

**现象描述**

未登录访问 /pricing，三张价格卡的按钮**全是「免费注册」**，包括 ¥199 的「信号版」和 ¥599 的「专业版」。用户点 ¥599 的卡，看到的是「免费注册」—— 这要么让人以为付费版也免费，要么直接失去信任。

**根因**

```tsx
) : !loggedIn ? (
  '免费注册'          // ← 未登录时，所有档位（哪怕 ¥599）都显示这个
) : (
  plan.cta            // ← 只有登录后才用各自的 cta
)
```

未登录分支硬编码了 `'免费注册'`，把 `plan.cta` 覆盖掉了（`plan.cta` 本身分别是「免费注册 / 开通信号版 / 联系销售」，配置是对的）。

**复现步骤**
1. 清空 localStorage，访问 http://localhost:3000/pricing
2. 观察三张卡的按钮：全部「免费注册」
3. 登录后再看：「当前套餐 / 开通信号版 / 联系销售」—— 文案才正确

**具体改法**

未登录时按档位给出差异化文案：

```tsx
) : !loggedIn ? (
  plan.id === 'free' ? '免费注册' : `登录后开通 ${plan.name}`
) : (
  plan.cta
)
```

---

#### P1-2｜「联系销售」按钮把已登录用户送去注册页

| 项 | 内容 |
|---|---|
| **严重度** | **P1** |
| **位置** | `frontend/src/app/pricing/page.tsx:163-166` |
| **截图** | `24-pricing-enterprise-cta.png` |

**现象描述**

已登录用户点「专业版」的「联系销售」按钮，被重定向到 `/register`（创建账号页）。语义完全错位 —— 企业级客户最不可能需要的动作是"再注册一个账号"。

**根因**

```tsx
if (plan.id === 'enterprise') {
  router.push('/register');   // ← 无论是否已登录
  return;
}
```

而 `choosePlan` 开头已经有 `if (!loggedIn) { router.push('/login'); return; }`，说明「未登录」的分支逻辑是有的，只是 enterprise 分支没区分登录态。

**复现步骤**
1. 登录任意账号
2. 访问 /pricing，点「专业版」的「联系销售」
3. URL 变成 `/register`，页面显示「创建账号」

**具体改法**

企业版本来也不该走自助购买，改成弹出联系表单或 mailto：

```tsx
if (plan.id === 'enterprise') {
  setContactOpen(true);   // 弹窗收集联系方式，或 window.location.href = 'mailto:sales@...'
  return;
}
```
若暂不实现联系表单，至少**已登录用户不要跳注册页**，改为跳 `/pricing#contact` 或展示"请联系客服"提示。

---

#### P1-3｜定价页价格与数据库/已收订单金额三方不一致

| 项 | 内容 |
|---|---|
| **严重度** | **P1** |
| **位置** | `frontend/src/app/pricing/page.tsx:85,106`（硬编码价格）<br>`backend/app/api/v1/commerce.py:242-254`（实际扣费金额） |
| **截图** | `09-pricing-loggedin.png` |

**现象描述**

定价页硬编码「信号版 ¥199 / 每月」「专业版 ¥599 / 每月」，但：

| 来源 | pro | enterprise |
|---|---|---|
| 定价页前端硬编码 | ¥199 | ¥599 |
| `subscription_plans` 表（`monthly_price_cents`） | 9900 分 = **¥99** | 49900 分 = **¥499** |
| 历史上已收订单 `payment_orders.amount_cents` | 19900 分 = **¥199** | — |

**订了 pro 却按 ¥99 扣、标价 ¥199** —— 这三处对不上，用户看到的价格和实际扣的钱不是一回事，客服会被告到。

**复现步骤**
1. 访问 /pricing，读取「信号版 ¥199 每月」
2. `sqlite3 backend/data/app.db "select name, monthly_price_cents from subscription_plans;"` → pro = 9900
3. 对比 `payment_orders` 已有订单 amount_cents = 19900

**具体改法**

价格不要在前端硬编码，改为从后端拉：

```tsx
// frontend/src/app/pricing/page.tsx
const [plansData, setPlansData] = useState<Plan[]>(plans);
useEffect(() => {
  apiRequest.get<{plans: BackendPlan[]}>('/billing/plans').then(r => {
    setPlansData(plans.map(p => ({
      ...p,
      price: `¥${(r.plans.find(b=>b.name===p.id)?.monthly_price_cents ?? 0)/100}`,
    })));
  });
}, []);
```
后端需补一个返回 `subscription_plans` 列表的公开端点（当前没有）。短期兜底：**至少把前端硬编码值改成与 DB 一致**，并把年付价（99000 / 499000）也展示出来（现在完全没有年付选项）。

---

#### P1-4｜充值下单时不带 plan_id，订单不会关联套餐，充值到账后套餐不会自动开通

| 项 | 内容 |
|---|---|
| **严重度** | **P1** |
| **位置** | `frontend/src/app/wallet/page.tsx:145-148`（请求体缺 `plan_id`）<br>`backend/app/services/commerce/payment_service.py:266-290`（按 order.plan_id 升级套餐） |
| **截图** | `16-wallet-deposit-submit.png` |

**现象描述**

后端 `payment_service.py` 里有一段专门的逻辑：确认到账时读 `order.plan_id`，据此自动把用户 `subscription_tier` 升上去（代码注释写着「否则『收了钱但套餐没变』」）。但前端下单时**根本没传 `plan_id`**：

```tsx
const res = await apiRequest.post<DepositResult>('/commerce/deposit', {
  amount_cents: Math.round(values.amount * 100),
  remark: '用户充值',        // ← 只有备注，没有 plan_id
});
```

实测下单后数据库里的订单 `plan_id` 为空：

```
PO-MQNWUDCW|029c80e1-...|10000|pending|        ← plan_id 空
```

**结果**：即使 P0-3 / P0-6 都修好、钱真的确认到账了，**用户的套餐也不会升级**，付费墙会把付费用户挡在外面。而定价页白纸黑字写着「充值到账后系统会**自动开通对应套餐**（free / pro / enterprise）」—— 这句话目前是假的。

**复现步骤**
1. 登录后（绕过 P0-2 显示表单），提交一笔充值
2. `sqlite3 backend/data/app.db "select order_no, plan_id from payment_orders order by created_at desc limit 1;"` → `plan_id` 为空
3. 管理员确认到账 → `payment_service.py:268` 的 `if _plan_id:` 不成立 → 不升级套餐

**具体改法**

1. 充值 Modal 增加套餐选择（至少能选「充值对应哪个套餐」），下单时带上：
   ```tsx
   const res = await apiRequest.post<DepositResult>('/commerce/deposit', {
     amount_cents: Math.round(values.amount * 100),
     remark: '用户充值',
     plan_id: selectedPlanId,   // ← free/pro/enterprise
   });
   ```
2. 从定价页跳转到钱包时带上意图：`/wallet?plan=pro`，钱包页读取 `searchParams.get('plan')` 预选档位。

---

#### P1-5｜定价页把 Markdown 语法 `**...**` 直接渲染成正文

| 项 | 内容 |
|---|---|
| **严重度** | **P1** |
| **位置** | `frontend/src/app/pricing/page.tsx:437` |
| **截图** | `09-pricing-loggedin.png` |

**现象描述**

「关于付款方式」提示框里，页面上原样显示：

> 充值到账后系统会**自动开通对应套餐**（free / pro / enterprise），不需要另外发邮件申请。

`**` 号直接露在页面上。这是**关于付费的关键承诺句**，出现在最需要专业感的位置却带着未渲染的 Markdown 标记 —— 典型的"AI 生成的通用模板"痕迹。

**复现步骤**
1. 访问 /pricing，滚动到「关于付款方式」区块
2. 可见字面量 `**自动开通对应套餐**`

**具体改法**

改用 MUI 的强调样式，或直接去掉标记：

```tsx
<Typography variant="body2" sx={{ fontWeight: 700 }}>
  自动开通对应套餐（free / pro / enterprise）
</Typography>
```

---

#### P1-6｜落地页 / 登录页 / 定价页全部被套在已登录 App 外壳里，未登录访客看到 15 项后台菜单

| 项 | 内容 |
|---|---|
| **严重度** | **P1** |
| **位置** | `frontend/src/app/layout.tsx:26`（`<AppLayout>` 无条件包裹）<br>`frontend/src/components/layout/AppLayout.tsx:63-145`（15 项 `menuItems`）<br>`frontend/src/components/layout/AppLayout.tsx:276-282`（WebSocket 状态 chip） |
| **截图** | `01-landing-loggedout.png`、`03-login.png` |

**现象描述**

未登录访客访问落地页 / 登录页 / 注册页 / 定价页，顶端都是同一条**后台管理导航**：仪表板、任务管理、模型管理、数据管理、系统监控、预测分析、策略回测、策略模板、参数调优、策略信号、今日信号、纸面信号、多策略纸面盘、超参优化、系统设置 —— 共 15 项。右上角还挂着一个「已连接 / 未连接」的 **WebSocket 状态指示器**。

这带来三个问题：
1. **首屏被占掉**：900px 高的视口里，导航条就吃掉 64px，落地页 Hero 的 h1 顶部在 267px。
2. **访客误入后台**：点任意菜单项 → 被重定向到 /login（实测「任务管理」→ `/login`），但用户已经"点了一个功能菜单"，却得到一个登录框，转化体感很差。
3. **登录页出现「未连接」chip**：登录页上显示 WebSocket 连接状态，对访客毫无意义，且暗示系统坏了。

**复现步骤**
1. 清空 localStorage
2. 访问 http://localhost:3000/ ，观察顶栏 15 项菜单 + 右上角「未连接」
3. 点「任务管理」→ 跳 /login

**具体改法**

在 `layout.tsx` 里按路由分流，未登录场景用精简的营销页外壳：

```tsx
// frontend/src/app/layout.tsx
const PUBLIC_ROUTES = ['/login', '/register', '/reset-password'];
export default function RootLayout({children}) {
  // ... 
  <AppLayout>{children}</AppLayout>
}
```
更实际的做法：把 `AppLayout` 拆成 `AppShell`（后台，含侧栏与 WS chip）与 `MarketingShell`（顶栏只有 Logo + 登录/注册/定价），按 pathname 或登录态选择；**未登录状态下，落地页/定价页/登录页一律用 MarketingShell**。落地页首屏可回收 64px，同时把「未连接」chip 限制在 AppShell 内。

---

#### P1-7｜全站没有一条导航能到定价页、钱包页或订阅页

| 项 | 内容 |
|---|---|
| **严重度** | **P1** |
| **位置** | `frontend/src/components/layout/AppLayout.tsx:63-145`（menuItems 无 pricing/wallet/subscription）<br>`frontend/src/app/settings/page.tsx`（无相关入口） |
| **截图** | `01-landing-loggedout.png` |

**现象描述**

`menuItems` 里 15 项，**没有 `/pricing`、没有 `/wallet`、没有 `/settings/subscription`**。`/settings` 页里也没有任何指向订阅/钱包的链接（grep 无结果）。

后果：
- 付费用户想查看订阅状态、充值余额，**必须手敲 URL**
- 落地页 hero 里的「看定价」按钮在 `page.tsx:503` 是通往 `/pricing` 的**唯一入口** —— 但那个按钮在页面第 4 屏深处（top=1788px）
- 想从定价页去钱包充值，只能靠 P0-5 那个错误提示框里的「去钱包充值」按钮

**复现步骤**
1. 登录后点顶栏任意菜单，逐项查看 → 无「定价」「钱包」「订阅」
2. 直接访问 /wallet（已登录）→ 顶栏没有任何高亮项指示当前位置

**具体改法**

在 `menuItems` 中补两项（并把 `/pricing` 放在靠前位置）：

```tsx
{ key: '/pricing', icon: CreditCard, label: '定价' },
{ key: '/wallet', icon: Wallet, label: '钱包' },
```
并在 `/settings` 页加一个「订阅管理」入口指向 `/settings/subscription`（该页本身已在 `pricing/page.tsx:227` 被引用，但无处可达）。

---

#### P1-8｜限流按代理 IP 计数，全站用户共用一个 60 次/分钟 桶

| 项 | 内容 |
|---|---|
| **严重度** | **P1** |
| **位置** | `backend/app/middleware/rate_limiting.py:143-150`（client_id 取法）<br>`frontend/next.config.js:24-45`（rewrites 未设 X-Forwarded-For）<br>`frontend/src/services/api.ts:274-277`（429 无退避重试） |
| **截图** | `06-wallet-loggedin.png`（页面级 toast） |

**现象描述**

审计过程中**实际把登录功能打挂了**：连续访问几次后，登录页提交直接返回「请求过于频繁，请稍后再试」，且**等待 60 秒后仍然失败**（小时级窗口 1000 次已耗尽，`x-ratelimit-remaining-hour: 0`）。

根因是限流键的取法：
```python
forwarded_for = request.headers.get("X-Forwarded-For")
if forwarded_for: return str(forwarded_for.split(",")[0].strip())
client_host = getattr(request.client, "host", "unknown")   # ← 代理 IP
return str(client_host)
```
前端所有 API 都走 Next.js rewrites 代理到后端（`next.config.js` 的 `/api/v1/:path*` → 后端），请求的来源 IP 对后端而言**都是 Next.js 服务器的 IP**。`next.config.js` 没有配置 `X-Forwarded-For` 透传（虽然 Next 内部 `base-server.js:509` 会 `??=` 补一个，但审计中实测浏览器端可以自带 `X-Forwarded-For` 头并被后端采纳 —— 说明这个头在链路上是客户端可控的）。

结果：**所有在线用户共享一个 60/min + 1000/h 的配额**。一个人的高频操作会让所有人（包括付费客户的登录、充值）一起被锁。

**复现步骤**
1. 连续快速访问 /wallet 等多接口页面
2. 观察 `x-ratelimit-remaining-hour` 归零
3. 此后**任何**页面登录都返回 429「请求过于频繁，请稍后再试」

**具体改法**

1. 后端：按**用户 ID**（已认证时）而非 IP 计数；未认证时才用 IP，并把阈值放宽（登录/注册端点单独设更宽松的桶）。
2. 前端 `api.ts`：429 增加指数退避重试（当前只对 401 做了 refresh 重试，429 直接 reject）。
3. 部署层：Next.js rewrites 显式设置 `headers: {'X-Forwarded-For': '$remoteAddress'}`，并在后端**只信任来自已知反代的该头**（否则可被客户端伪造绕过限流 —— 实测确实可绕过）。

---

### P2 — 中等（打磨项）

---

#### P2-1｜登录页无密码可见性切换、无 autocomplete，密码管理器不友好

| 项 | 内容 |
|---|---|
| **严重度** | **P2** |
| **位置** | `frontend/src/app/login/page.tsx:140-148` |
| **截图** | `03-login.png`、`13-login-register-tab.png` |

**现象描述**

登录页（以及它内嵌的注册 Tab）三个输入框的实测属性：

| 字段 | autocomplete | 密码可见性切换 |
|---|---|---|
| email | **无** | — |
| password（登录） | **无** | **无** |
| password（注册 Tab） | **无** | **无** |

用户输错密码只能盲打、无法确认；浏览器/1Password 不会自动填充（无 `autocomplete="current-password"`）；移动端键盘也不会调出正确的键盘类型。

对比：独立的 `register/page.tsx`（antd）**做对了** —— `autoComplete="email" / "username" / "new-password"` 齐全，antd 的 `Input.Password` 自带可见性切换。所以这是 MUI 版登录页的独立缺陷。

**具体改法**

```tsx
<TextField name="email" label="邮箱" type="email" required fullWidth size="small"
  autoComplete="email" />
<TextField name="password" label="密码" type="password" required fullWidth size="small"
  autoComplete="current-password"
  InputProps={{
    endAdornment: (
      <InputAdornment position="end">
        <IconButton aria-label="切换密码可见性" onClick={() => setShowPwd(s=>!s)} edge="end">
          {showPwd ? <VisibilityOff size={18}/> : <Visibility size={18}/>}
        </IconButton>
      </InputAdornment>
    ),
  }}
  inputProps={{ type: showPwd ? 'text' : 'password' }}
/>
```
注册 Tab 同理，`autoComplete="new-password"`。并给两个 Tab 内的 `username` 加上 `autoComplete="username"`。

---

#### P2-2｜登录页内嵌注册 Tab 缺少「确认密码」，且注册后 store 未映射用户字段

| 项 | 内容 |
|---|---|
| **严重度** | **P2** |
| **位置** | `frontend/src/app/login/page.tsx:165-178`（缺确认密码）<br>`frontend/src/app/login/page.tsx:91`（`setUser(data.user)` 缺映射） |
| **截图** | `13-login-register-tab.png`、`21-login-tab-register-result.png` |

**现象描述**

两个问题：

**(a) 缺确认密码**。登录页的「注册」Tab 只有 email/username/password 三个字段，独立注册页有四个。同一平台两条注册路径的字段要求不一致 —— 走 Tab 注册的用户输错密码无感知。

**(b) `setUser(data.user)` 未做字段映射**。登录路径（`page.tsx:53-60`）做了完整映射：
```tsx
setUser({ id, name: username||name||email, email, role: tier||'free', is_admin, subscription_tier });
```
但注册 Tab（`page.tsx:91`）是 `setUser(data.user)` **原样传入**。后端返回的是 `{id, email, username, subscription_tier, is_admin}`（`backend/app/api/v1/auth.py:95-103`），**没有 `name` 字段**。实测注册后 store 里 `name` 为 undefined → 顶栏头像永远显示通用的 lucide `User` 图标而不是用户名；`role` 也为 undefined。

**具体改法**

(a) 给注册 Tab 加确认密码字段（复用独立注册页的校验逻辑）。
(b) 抽出公共映射函数，两处共用：
```tsx
// 抽到 lib/auth.ts
export const mapAuthUser = (u: any) => ({
  id: u.id || '',
  name: u.username || u.name || u.email || '',
  email: u.email || '',
  role: u.subscription_tier || 'free',
  is_admin: Boolean(u.is_admin),
  subscription_tier: u.subscription_tier,
});
```
> 顺带：`login/page.tsx:91` 与 `register/page.tsx:54` 也是同样的问题 —— `register/page.tsx:54` 用了 `setUser(data.user as unknown as ...)` 强转绕过类型检查，运行时同样缺 `name`/`role`。**三个注册入口、两种写法、同一类 bug**，值得统一。

---

#### P2-3｜钱包页的余额进度条算法无意义，数字与图形不一致

| 项 | 内容 |
|---|---|
| **严重度** | **P2** |
| **位置** | `frontend/src/app/wallet/page.tsx:247-252`（余额条）<br>`frontend/src/app/wallet/page.tsx:309-313`（用量条） |
| **截图** | `06-wallet-loggedin.png` |

**现象描述**

余额卡下面挂了一条进度条，百分比算的是：
```tsx
percent={balance ? (balance.balance_cents / 100 / 100) * 100 : 0}
```
`balance_cents/100` 是元，再 `/100*100` = 元数本身。所以余额 10 元 → 进度条显示 **10%**、余额 1000 元 → **1000%**。这个数字**没有任何含义**（既不是占某个上限的比例，也不是占充值额的比例），却用了一个标准的进度条视觉，用户会误读为"额度已用/已充"。

用量条同理：`percent={Math.min(100, (data.count / 1000) * 100)}` —— 硬编码 1000 次为分母，与实际配额无关（DB 里 pro 是 500 次、free 是 50 次、enterprise 不限）。

**具体改法**

余额条要表达"离下一个档位/最低充值还差多少"，或干脆删掉。改成有语义的形式：

```tsx
// 例如：按最低充值档位 ¥100 显示充值进度
<Progress percent={Math.min(100, ((balance?.balance_cents ?? 0) / 10000) * 100)} format={() => ''} />
```
或者直接去掉余额进度条，把那 8px 空间还给信息密度。用量条的分母应改为**当前套餐的真实配额**（`monthly_backtest_limit`）。

---

#### P2-4｜钱包页"账单历史"跳转到不存在的路由，落到 404 页

| 项 | 内容 |
|---|---|
| **严重度** | **P2** |
| **位置** | `frontend/src/app/wallet/page.tsx:326,346`（`router.push('/commerce/billing/history')`） |
| **截图** | `08-wallet-billing-history-404.png` |

**现象描述**

「账单历史」卡片右上角的「查看全部」和列表下方超过 5 条时的「查看更多」，都跳 `/commerce/billing/history`。该路径**没有对应的 Next.js 路由**（`frontend/src/app/` 下无 `commerce/` 目录），点击后进入 404 页（"页面不存在 / 返回仪表板"）。

这是**把后端 API 路径当成了前端路由** —— 典型的路径混淆。

**具体改法**

真正的历史已经在当前页展示了（`billingHistory.slice(0,5)`）。二选一：
- 简单：去掉「查看全部」按钮，或改成"显示更多"（前端分页，`apiRequest.get('/commerce/billing/history?limit=20&offset=20')`）
- 完整：新建 `frontend/src/app/wallet/history/page.tsx` 真正实现一个账单历史页，并把按钮指向 `/wallet/history`

---

#### P2-5｜钱包页"账户状态"卡片语义混乱，把免费用户也标成「余额不足」

| 项 | 内容 |
|---|---|
| **严重度** | **P2** |
| **位置** | `frontend/src/app/wallet/page.tsx:271-293` |
| **截图** | `06-wallet-loggedin.png` |

**现象描述**

第三张卡叫「账户状态」，红点 + 文字「余额不足」。但这个判定只有 `balance_cents > 0`（`wallet/page.tsx:275`），**没有考虑用户的套餐层级**。体验版（free）用户按定价页说明「体验版不需要任何付款」（`pricing/page.tsx:434`），本就不需要余额，却会看到一个红色的「余额不足」警告 —— 免费用户第一眼看到的是一个红色告警，转化体感极差。

另外卡片标题「账户状态」下面的内容却是一个「充值」按钮，标题与内容不符。

**具体改法**

按套餐与余额给出分层文案：

```tsx
const isFreeTier = tier === 'free';
<Statistic
  title="账户状态"
  value={isFreeTier ? '体验版' : (balance && balance.balance_cents > 0 ? '可用' : '余额不足')}
/>
```
并把该卡的标题改为「余额与充值」，让标题与内容对齐。

---

#### P2-6｜定价页缺少 FAQ、退款政策、联系方式、发票说明

| 项 | 内容 |
|---|---|
| **严重度** | **P2** |
| **位置** | `frontend/src/app/pricing/page.tsx`（全文 487 行，无 FAQ 区块） |
| **截图** | `09-pricing-loggedin.png` |

**现象描述**

实测 pricing 页文本，付费决策所需的常见信息**全部缺失**：

```
hasFAQ: false    hasAnnual: false   hasRefund: false
hasTrial: false  hasCancel: false   hasInvoice: false
priceCount: ['¥0', '¥199', '¥599']
```

即：无 FAQ、无年付/季付选项、无退款政策、无试用期说明、无取消订阅说明、无发票说明。「关于付款方式」只说了怎么付钱，没说**付了能退吗、什么时候能用、怎么取消、怎么开票**。

考虑到整个平台卖的是"每天收盘后那 20 分钟的判断"这种**时效性极强的服务**，用户第一个想问的一定是"信号不准能退吗" —— 这恰恰没有答案。

**具体改法**

在「关于付款方式」之后、「信任背书」之前加一个 FAQ 区块（用 MUI `Accordion`），至少覆盖：
- 什么时候能收到信号？（T+1 收盘后 17:45）
- 信号不准可以退款吗？退款政策
- 怎么取消订阅？到期前可用吗？
- 支持开发票吗？
- 支付后多久能开通？
- 有没有试用/体验期？

同时补上年付档位（DB 里 `yearly_price_cents` 已有值：pro 99000 / enterprise 499000，前端完全没用上）。

---

#### P2-7｜落地页首屏 CTA 语义含糊 + 页脚重复

| 项 | 内容 |
|---|---|
| **严重度** | **P2** |
| **位置** | `frontend/src/app/page.tsx:260-280`（Hero CTA）<br>`frontend/src/app/page.tsx:649-661`（页脚）<br>`frontend/src/components/layout/AppLayout.tsx:405-420`（外壳页脚） |
| **截图** | `01-landing-loggedout.png` |

**现象描述**

**(a) Hero 主 CTA 叫「开始使用」，点进去是 `/login`**（实测 `URL after 开始使用: /login`）。一个新访客看到「开始使用」却落到登录页（而非注册页），是转化路径上多绕的一步。而同一个页面的第二屏 CTA 反而是「免费注册看今日信号」→ `/register`（`page.tsx:495`）—— 同一页面两个 CTA 指向不同落点，措辞也不一致。

**(b) 页脚渲染了两遍**。落地页自己有一个页脚（`page.tsx:649-661`），AppLayout 外壳还有一个（`AppLayout.tsx:405-420`），实测全文出现 **2 次** `©`：

```
'股票预测平台 © 2025 — 基于 AI 的智能投资决策系统'   ← page.tsx 自己画的
'股票预测平台 ©2025 - 基于AI的智能投资决策系统'      ← AppLayout 外壳画的
```

文案还不同（一个用「，」一个用「-」，一个写 2025 一个也是 2025 但格式不同）。当前是 2026 年 10 月，版权年份还写着 2025。

**具体改法**

(a) Hero 主 CTA 改为「免费注册」→ `/register`（新用户的主路径），次 CTA 保留「查看策略模板」；或保留「开始使用」但按登录态分流。
(b) 删掉 `page.tsx:649-661` 的独立页脚（外壳已经有了），或反过来从外壳里去掉。年份改为动态：`new Date().getFullYear()`。

---

#### P2-8｜落地页 Hero 右侧是纯装饰占位图，观感像未完成的模板

| 项 | 内容 |
|---|---|
| **严重度** | **P2** |
| **位置** | `frontend/src/app/page.tsx:305-347` |
| **截图** | `01-landing-loggedout.png` |

**现象描述**

Hero 右侧 320×320 的区域里是一个渐变圆角方块 + 一个 120px 的 `TrendingUp` 图标（透明度 0.15）+ 一个虚线内框 + 一个 64px 的 `LineChart` 图标（透明度 0.1）。**没有任何真实产品截图**。

在一个主张「我们把口径都摊开写出来」的页面上，首屏放一个假图表占位，会削弱"我们是做事的人"这一印象。这也是"AI 生成的通用模板"的典型特征：图标堆砌 + 渐变方块 + 装饰性圆点背景（`page.tsx:157-180` 两个 radial-gradient 圆）。

**具体改法**

换成真实资产，二选一：
- 贴一张纸面账本净值曲线截图（用 echarts 静态导出成 PNG，或直接内联一个轻量 SVG 走势图）
- 贴「今日信号」页的实际输出截图（含 4 条线的持仓与买/卖标记）

如果暂时没有素材，**宁可留白**，也不要放装饰性图标方块 —— 留白比假图更可信。

---

### P3 — 轻微

---

#### P3-1｜注册页 `requiredMark={false}` 与 antd 实际渲染冲突，必填标记语义不清

| 项 | 内容 |
|---|---|
| **严重度** | **P3** |
| **位置** | `frontend/src/app/register/page.tsx:79` |
| **截图** | `14-register-validation.png` |

`requiredMark={false}` 传给了 antd `<Form>`，但 antd 仍给每个必填项的 label 加了 `ant-form-item-required` 类（实测 4 个 label 全部 `hasAsterisk: true`）。antd v6 的 `::after` content 实测是 `":"`（不是 `*`），颜色 `rgba(0,0,0,0.88)` 而非红色。

即：**"关闭必填标记"这个意图在 antd v6 下没生效**，label 上仍带着必填后缀，视觉上又不像传统的红色星号。实测 `requiredMark={false}` 与渲染结果不自洽。

**具体改法**：删掉 `requiredMark={false}`（让它显示标准的必填标记），或若确实要隐藏，改用 `requiredMark="optional"`。同时确认 antd v6 的 `:` 后缀样式是否符合设计预期 —— 若不符合，全局统一成红色 `*`。

---

#### P3-2｜注册页错误提示层级混乱：服务端错误显示在卡片顶部，字段错误显示在字段下方

| 项 | 内容 |
|---|---|
| **严重度** | **P3** |
| **位置** | `frontend/src/app/register/page.tsx:78`（全局 Alert）<br>`frontend/src/app/register/page.tsx:101-117`（字段级校验） |
| **截图** | `17-register-duplicate-email.png`、`18-register-password-mismatch.png` |

实测两种错误的呈现位置不同：

| 错误类型 | 呈现位置 | 实例 |
|---|---|---|
| 客户端字段校验 | 字段正下方 | 「两次输入的密码不一致」「用户名长度为 2–50 个字符」 |
| 服务端错误 | 卡片顶部（远离出错的字段） | 「该邮箱已注册」 |

「该邮箱已注册」出现在卡片顶部、离邮箱输入框很远（中间还隔着用户名字段），用户很难把错误和输入框对应起来。

**具体改法**：把服务端错误映射到对应字段，让错误就地显示：

```tsx
<Form.Item name="email" ... rules={[...,
  { validator: async (_, v) => {
      try { await doRegister(v); } catch (e) {
        if (String(e.message).includes('邮箱')) throw new Error(e.message);
        throw e;
      }
    }}
]}>
```
或保留全局 Alert，但把邮箱/用户名的服务端错误分别绑定到 `Form.Item` 的 `error` 属性。

---

#### P3-3｜`services/commerce/commerce.service.ts` 是一层无用的抽象

| 项 | 内容 |
|---|---|
| **严重度** | **P3** |
| **位置** | `frontend/src/services/commerce/commerce.service.ts`（全 118 行）<br>`frontend/src/hooks/useCommerce.ts:5`（唯一调用方） |
| **截图** | — |

`commerceService` 唯一被 `hooks/useCommerce.ts` 使用，而 `useCommerce` 本身在本次审计的 7 个目标页面里**没有一处被引用**。这层 service 重复定义了 `apiRequest` 的调用（同样的路径、同样的错误处理），唯一的"增量"是 `deposit()` / `refund()` 两个薄包装。

同时 `frontend/src/services/api/commerce.ts` 定义了 `BalanceResponse` / `BillingRecord` 等类型，但 `wallet/page.tsx` 和 `settings/subscription/page.tsx` 都在**各自文件里重新定义了一套同名类型**（`wallet/page.tsx:28-85` 定义了 7 个 type，`services/api/commerce.ts` 定义了 4 个），两者字段还有细微差异（如 `UsageEvent` 有 `total_cost_yuan`，`UsageSummaryResponse.events` 有 `total_cost` 但没有 `total_cost_yuan`）。

**具体改法**

要么删掉这层 service（让 `useCommerce` 直接用 `apiRequest`），要么把它扶正成唯一入口：让 `wallet/page.tsx` / `settings/subscription/page.tsx` **从 `services/api/commerce.ts` 导入类型**，删掉页面内的重复定义。当前两套类型并存，是 P0-1 那类契约漂移的温床。

---

#### P3-4｜订阅页的"账单记录"永远是空的 Empty 状态，与钱包页的账单历史割裂

| 项 | 内容 |
|---|---|
| **严重度** | **P3** |
| **位置** | `frontend/src/app/settings/subscription/page.tsx:222-233` |
| **截图** | `11-subscription.png` |

订阅页底部固定渲染一个 `<Empty description="账单记录将由支付服务同步" />`，**不查询任何数据**。而钱包页有真实的「账单历史」（`/commerce/billing/history`）。两个页面对"账单"的定义不一致：订阅页说"将由支付服务同步"（永远不会有，因为 Stripe 没装），钱包页才是真相来源。

同时订阅页顶部的"本月用量"（回测次数 / 策略数量 / 并发任务）来自 `/dashboard/stats`，该端点实测**返回 500**（`'AsyncSession' object has no attribute 'query'`，后端 bug），于是页面静默回退到 `defaultUsage` 硬编码值（50/20/3）—— enterprise 用户的 `max_strategies` 实际是**不限量**，页面却显示 `策略数量 0 / 20`。

**具体改法**

1. 订阅页的"账单记录"改为直接调 `/commerce/billing/history` 渲染真实列表（与钱包页复用同一数据源和组件），而不是写死 Empty。
2. 用量数据改调 `/billing/subscription`（该端点实测 200 且返回 `subscription_tier`），或在 `/dashboard/stats` 的 500 时**显式提示"用量数据暂不可用"**，而不是静默展示假配额。后端那个 `AsyncSession` 的 500 也应单独修（不在本次前端审计范围，但值得单开一条）。

---

## 先修清单（按 P0/P1 排序，12 条）

| # | 严重度 | 问题 | 位置 | 截图 |
|---|---|---|---|---|
| 1 | **P0** | 钱包页 4 个接口因响应封装不匹配全部失败，余额恒显示 0.00 | `wallet/page.tsx:120-134` + `services/api.ts:193-201` | `06-wallet-loggedin.png` |
| 2 | **P0** | 钱包页「充值」按钮是空操作（点击隐藏表单），无任何反馈 | `wallet/page.tsx:284-292, 356-390` | `07-wallet-deposit-click.png` |
| 3 | **P0** | 充值金额校验永远不通过（`type:'number'` vs 字符串），任何金额都提交不了 | `wallet/page.tsx:363-379` | `16-wallet-deposit-submit.png` |
| 4 | **P0** | 忘记密码 / 重置密码两个模式完全不可用（`e.preventDefault is not a function`） | `reset-password/page.tsx:28, 51, 121, 153` | `19-`, `20-reset-password-*.png` |
| 5 | **P0** | 定价页支付请求 plan_id 传的是档位名、后端查的是 UUID，永远 400 | `pricing/page.tsx:172-175` | `10-pricing-free-cta-loggedin.png` |
| 6 | **P0** | 后台「确认到账」4 处 URL 缺 `/commerce` 前缀 → 404，管理员永远无法确认收款 | `admin/payments/page.tsx:84,116,129,143` | `25-admin-payments.png` |
| 7 | **P1** | 充值下单不带 `plan_id` → 订单不关联套餐 → 确认到账也不升级套餐 | `wallet/page.tsx:145-148` | `16-wallet-deposit-submit.png` |
| 8 | **P1** | 定价页三个档位 CTA 全显示「免费注册」，¥199/¥599 无付费语义 | `pricing/page.tsx:341-347` | `02-pricing-loggedout.png` |
| 9 | **P1** | 定价页价格硬编码 ¥199/¥599，与 DB（¥99/¥499）和已收订单三方不一致 | `pricing/page.tsx:85,106` | `09-pricing-loggedin.png` |
| 10 | **P1** | 限流按代理 IP 计数，全站共用一个桶；审计中实际打挂了登录 | `rate_limiting.py:143-150` + `next.config.js` | `06-wallet-loggedin.png` |
| 11 | **P1** | 营销页被套在后台 App 外壳里：15 项菜单 + WebSocket「未连接」chip，未登录访客点菜单被弹到登录页 | `layout.tsx:26` + `AppLayout.tsx:63-145, 276-282` | `01-landing-loggedout.png`, `03-login.png` |
| 12 | **P1** | 全站无导航可达定价页/钱包页/订阅页；「联系销售」把已登录用户送去注册页 | `AppLayout.tsx:63-145` + `pricing/page.tsx:163-166` | `24-pricing-enterprise-cta.png` |

> 另需单独排期：**P1-5**（定价页漏出 `**` 标记）与 **P1-6**（落地页页脚重复、年份写 2025）改动极小，可随第 8/9 条一起改。

---

## 总体评价

**文案与信息架构有真东西，运行时却全线崩塌。** 落地页的"我们先讲难听的那部分"、定价页的逐项对照表、把 0.976 换成可复现的 0.8193 —— 这些都不是 AI 通用模板能写出来的东西，作者显然懂自己的产品。**但这套页面的付费转化链路在运行时是断的**：钱包余额永远 0.00、充值按钮点了没反应、金额填不进去、忘记密码全线报错、付费档位调不通、后台永远看不到订单。6 个 P0 任意一个单独存在都足以让转化归零，而它们全部同时存在 —— 加上定价页把 ¥199 显示成"免费注册"、营销页套着后台菜单、导航里根本没有钱包入口，这套东西**目前不像是一个真的在收钱的产品的用户界面**。

**修复顺序建议**：第 1、2、3、6 条是纯粹的接口契约与状态管理错误，合计改动约 100 行、无设计决策，可以一次性合；第 4、5、7 条是"契约写错"类问题，同样是确定性修复；第 8-12 条才是体验与信息层。**在第 1/2/3/6 条修好之前，不建议再对定价页和落地页的文案做任何打磨** —— 现在改文案是在给一条断掉的管道贴标签。

**一点保留**：这组页面里有 3 套 UI 库（MUI v7 / antd v6）、3 套图表库并存，本次审计的 7 个目标页面里 MUI 与 antd 各占一半，同一个转化漏斗上"登录页（MUI）→ 注册页（antd）→ 定价页（MUI）→ 钱包页（antd）"交替出现，表单控件、圆角、阴影、焦点态全不一致。**这不单是审美问题** —— P0-4 那个 `e.preventDefault is not a function` 正是因为 register 页用 antd（正确写法 `submit(values)`）、reset-password 页用 antd 但写成了 event 签名，而两页长得几乎一样，review 时极难发现。**建议这 7 个页面在修 bug 的同一轮里统一到 MUI**（antd v6 的 `Input.Password` 有内置可见性切换，迁移时需自行补 `InputAdornment`），否则下一轮还会出同类的"看起来一样、行为不同"的坑。
