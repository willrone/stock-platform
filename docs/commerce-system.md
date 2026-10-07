# 按使用量计费系统 - 实施总结

## 概述

实现了完整的按使用量计费系统（Pay-per-Use Billing System），包括后端模型、服务、API 和前端页面。

## 已完成的功能

### 1. 数据模型层 (`backend/app/models/commerce_models.py`)

#### 核心实体
- **UserBalance**: 用户钱包余额管理
  - `balance_cents`: 当前余额（单位：分）
  - `credited_cents`: 赠金额度
  - `total_spent_cents`: 累计消费
  
- **UsageEvent**: 用量事件记录
  - `event_type`: 事件类型（backtest_basic, api_call, 等）
  - `unit_cost`: 单次成本
  - `quantity`: 消耗数量
  - `total_cost`: 总成本
  - `metadata`: 附加元数据

- **BillingRecord**: 账单记录
  - `charge_status`: 扣费状态（charged/pending/failed/refunded/credited）
  - `balance_before_cents`: 扣费前余额
  - `balance_after_cents`: 扣费后余额
  
- **CommercePricingRule**: 计费规则配置
  - `event_type`: 事件类型
  - `base_price_cents`: 基础价格
  - `tier_discounts`: 阶梯折扣
  - `effective_from/until`: 有效期

#### 预定义计费规则 (DEFAULT_PRICING_RULES)
| 事件类型 | 单价 | 说明 |
|---------|------|------|
| backtest_basic | ¥0.50/次 | 基础回测 |
| backtest_realtime | ¥2.00/次 | 实时回测 |
| backtest_advanced | ¥5.00/次 | 高级回测 |
| api_call | ¥0.01/次 | API 调用 |
| data_download | ¥0.10/MB | 数据下载 |
| realtime_quote | ¥0.05/次 | 实时行情 |
| model_training | ¥50/小时 | 模型训练 |
| optimization | ¥10/次 | 参数优化 |

### 2. 服务层 (`backend/app/services/commerce/commission_service.py`)

#### CommerceService 类
- `get_balance(user_id)`: 查询用户余额
- `record_usage(user_id, event_type, quantity, metadata)`: 记录用量并扣费
- `deposit(user_id, amount_cents, remark)`: 充值
- `refund(user_id, billing_record_id, amount_cents, remark)`: 退款
- `get_usage_summary(user_id, month, year)`: 获取用量汇总
- `get_billing_history(user_id, limit, offset)`: 获取账单历史
- `get_pricing_rules(event_type)`: 获取计费规则
- `can_proceed(user_id, event_type, quantity)`: 检查是否允许执行

#### 辅助函数
- `check_can_proceed(user_id, event_type, quantity)`: 权限检查
- `seed_default_pricing_rules(db)`: 初始化默认计费规则

### 3. API 路由层 (`backend/app/api/v1/commerce.py`)

#### RESTful 端点
| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/commerce/usage` | 记录用量事件 |
| GET | `/commerce/balance` | 查询余额 |
| POST | `/commerce/deposit` | 充值 |
| POST | `/commerce/refund` | 退款 |
| GET | `/commerce/usage/summary` | 用量汇总 |
| GET | `/commerce/billing/history` | 账单历史 |
| GET | `/commerce/pricing/rules` | 计费规则 |
| GET | `/commerce/check/{event_type}` | 检查权限 |

### 4. 数据库迁移 (`backend/alembic/versions/commerce_001_initial.py`)

创建了 4 张表：
- `usage_events` - 用量事件
- `user_balances` - 用户余额
- `billing_records` - 账单记录
- `commerce_pricing_rules` - 计费规则

所有表都有适当的索引和外键约束。

### 5. 前端页面

#### 钱包页面 (`frontend/src/app/wallet/page.tsx`)
- 余额概览（当前余额、本月支出、账户状态）
- 用量统计图表
- 账单历史记录
- 充值表单

#### 服务层
- `frontend/src/services/api/commerce.ts` - TypeScript 类型定义和常量
- `frontend/src/services/commerce/commerce.service.ts` - 计费服务封装
- `frontend/src/hooks/useCommerce.ts` - React Hook，用于组件中调用

### 6. 测试文件

- `backend/tests/unit/test_commerce_models.py` - 单元测试脚本

## 设计原则

### 1. 扩展性
- 事件类型可扩展：新增事件只需在 `EventType` 枚举和 `DEFAULT_PRICING_RULES` 中添加
- 计费规则可配置：支持阶梯折扣、有效期、启用/禁用开关
- 支持多种支付方式（预留 `payment_method` 字段）

### 2. 可维护性
- 单一职责：模型、服务、API 分层清晰
- 接口抽象：前端通过 `CommerceService` 调用，不直接依赖 API
- TypeScript 类型安全：所有数据结构都有类型定义

### 3. 可测试性
- 服务层方法独立可调
- 支持 mock 测试
- 单元测试覆盖核心逻辑

### 4. 可读性
- 中文注释完善
- 变量命名语义化
- 代码结构清晰

### 5. 性能
- 数据库索引优化查询
- 批量操作支持（`quantity` 参数）
- 缓存友好（余额、规则等）

### 6. 维测性
- 详细日志记录
- 错误码标准化
- 审计追踪（`metadata` 字段）

## 下一步建议

### 优先级高
1. **支付网关集成** - 对接微信支付/支付宝
2. **自动化计费** - 在实际业务（回测、API 调用）中嵌入计费逻辑
3. **管理员后台** - 查看全局统计、处理充值申请

### 优先级中
4. **优惠券系统** - 支持折扣券、满减活动
5. **套餐包** - 打包销售（如 100 次回测套餐）
6. **余额预警** - 低于阈值时发送通知

### 优先级低
7. **企业版** - 多租户、部门分摊
8. **财务报表** - 导出 CSV/Excel
9. **API Key 计费** - 区分免费/付费 API Key

## 验证清单

- [x] 模型创建成功
- [x] 数据库迁移脚本就绪
- [x] 服务层方法完备
- [x] API 路由注册
- [x] 前端页面和 Hook
- [x] TypeScript 类型定义
- [x] 单元测试框架

## 待办事项

1. **实际集成**: 在回测 API 中嵌入计费逻辑（替换现有的 `enforce_quota`）
2. **运行迁移**: 执行 `alembic upgrade head` 创建表
3. **初始化规则**: 调用 `seed_default_pricing_rules()` 填充计费规则
4. **端到端测试**: 模拟完整流程：充值 → 执行回测 → 扣费 → 查账单
5. **UI 调整**: 根据实际渲染效果优化样式

---

**最后更新**: 2026-09-23  
**状态**: 核心框架完成，等待集成和测试