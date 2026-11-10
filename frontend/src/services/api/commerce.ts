/**
 * 按使用量计费 API 服务 —— 类型定义
 *
 * 提供计费事件记录、余额查询、充值、账单历史等接口的**响应类型**。
 *
 * ⚠️ 响应形态说明（与后端 /commerce/* 路由保持一致）：
 *   /commerce/* 下的端点**尚未统一为 {success, data, message, timestamp} 标准信封**，
 *   大多数端点直接返回业务对象本身（裸 payload）。前端 api.ts 的归一化层
 *   （见 @/services/api 的 normalizeResponse / unwrap）已兼容两种形态，
 *   因此这里声明的类型一律描述「业务数据本身」，不含信封字段。
 *
 * 类型口径以 backend/app/api/v1/commerce.py 的 Pydantic 模型为准：
 *   - BalanceResponse        ← commerce.py: BalanceResponse
 *   - UsageSummaryResponse   ← commerce.py: UsageSummaryResponse
 *   - BillingRecord          ← commerce.py: BillingRecordResponse
 *   - UsageRecordResponse    ← commerce.py: UsageRecordResponse
 *   - PricingRuleResponse    ← commerce.py: PricingRuleResponse
 *   - DepositResponse        ← commerce.py: DepositResponse
 */

/** 余额（后端 BalanceResponse） */
export interface BalanceResponse {
  user_id: string;
  balance_cents: number;
  balance_yuan: number;
  credited_cents: number;
  total_spent_cents: number;
  /** 后端为 Optional[str]，可能为 null */
  updated_at: string | null;
}

/**
 * 单个用量事件汇总（后端 events 字典的 value）
 * 注意：后端字段名是 total_cost_cents（不是 total_cost）。
 */
export interface UsageEventSummary {
  count: number;
  total_cost_cents: number;
  total_cost_yuan: number;
}

/** 用量汇总（后端 UsageSummaryResponse） */
export interface UsageSummaryResponse {
  month: number;
  year: number;
  total_spent_cents: number;
  total_spent_yuan: number;
  /** key = 事件类型，value = 该类型的用量汇总 */
  events: Record<string, UsageEventSummary>;
}

/** 账单记录（后端 BillingRecordResponse） */
export interface BillingRecord {
  id: string;
  user_id: string;
  usage_event_id: string;
  amount_cents: number;
  amount_yuan: number;
  charge_status: string;
  payment_method?: string | null;
  balance_before_cents?: number | null;
  balance_after_cents?: number | null;
  remark?: string | null;
  charged_at?: string | null;
  created_at?: string | null;
}

/** 用量记录响应（后端 UsageRecordResponse） */
export interface UsageRecordResponse {
  event_id: string;
  event_type: string;
  unit_cost: number;
  quantity: number;
  total_cost: number;
  billing_status: string;
  message: string;
}

/** 计费规则（后端 PricingRuleResponse） */
export interface PricingRuleResponse {
  event_type: string;
  base_price_cents: number;
  base_price_yuan: number;
  discounted_price_cents: number | null;
  discounted_price_yuan: number | null;
  unit_name: string | null;
  description: string | null;
  is_active: boolean;
}

/** 充值下单响应（后端 DepositResponse，2026-10-25 起为订单制，不立即加分） */
export interface DepositResponse {
  balance_cents: number;
  balance_yuan: number;
  credited_cents: number;
  /** 到账后才有的账单 ID（订单制下为 null） */
  billing_record_id: string | null;
  order_no: string | null;
  /** 订单状态：pending / paid */
  order_status: string | null;
  amount_cents: number | null;
  payment_method: string | null;
  payee_account: string | null;
  qr_image_url: string | null;
  instructions: string | null;
}

/** 退款响应（后端 POST /commerce/refund 的返回体） */
export interface RefundResponse {
  refund_record_id: string;
  amount_yuan: number;
  status: string;
}

/** 是否可执行的检查结果（后端 GET /commerce/check/{event_type}） */
export interface CanExecuteResponse {
  can_proceed: boolean;
  message: string;
  event_type: string;
  estimated_cost_yuan: number;
}

/** 充值订单（后端 PaymentOrder.to_dict） */
export interface PaymentOrder {
  id: string;
  order_no: string;
  user_id: string;
  amount_cents: number;
  amount_yuan: number;
  status: 'pending' | 'paid' | 'cancelled' | 'refunded';
  payment_method: string | null;
  plan_id: string | null;
  user_remark: string | null;
  admin_remark: string | null;
  billing_record_id: string | null;
  created_at: string | null;
  confirmed_at: string | null;
}

/** 收款配置（后端 PaymentConfig.to_dict + 上下限） */
export interface PaymentConfig {
  method_name: string;
  payee_account: string;
  qr_image_url: string;
  instructions: string;
  is_enabled: boolean;
  updated_at?: string | null;
  min_deposit_cents?: number;
  max_deposit_cents?: number;
}

/**
 * 计费服务接口契约。
 *
 * 与 CommerceService（@/services/commerce/commerce.service）保持一一对应，
 * 由该类 implements 本接口，编译期即可发现两处定义漂移。
 */
export interface CommerceAPI {
  /** 记录用量（扣费） */
  recordUsage: (
    event_type: string,
    quantity?: number,
    metadata?: Record<string, unknown>
  ) => Promise<UsageRecordResponse>;

  /** 余额查询 */
  getBalance: () => Promise<BalanceResponse>;

  /** 充值（下单，订单制） */
  deposit: (amount: number, remark?: string) => Promise<DepositResponse>;

  /** 退款 */
  refund: (billing_record_id: string, amount: number) => Promise<RefundResponse>;

  /** 用量汇总 */
  getUsageSummary: (month?: number, year?: number) => Promise<UsageSummaryResponse>;

  /** 账单历史 */
  getBillingHistory: (limit?: number, offset?: number) => Promise<BillingRecord[]>;

  /** 计费规则 */
  getPricingRules: (event_type?: string) => Promise<PricingRuleResponse[]>;

  /** 检查是否可以执行 */
  checkCanExecute: (event_type: string) => Promise<CanExecuteResponse>;
}

// 事件类型常量
export const EVENT_TYPES = {
  BACKTEST_BASIC: 'backtest_basic',
  BACKTEST_REALTIME: 'backtest_realtime',
  BACKTEST_ADVANCED: 'backtest_advanced',
  API_CALL: 'api_call',
  DATA_DOWNLOAD: 'data_download',
  REALTIME_QUOTE: 'realtime_quote',
  MODEL_TRAINING: 'model_training',
  OPTIMIZATION: 'optimization',
} as const;

// 事件名称映射
export const EVENT_NAMES: Record<string, string> = {
  [EVENT_TYPES.BACKTEST_BASIC]: '基础回测',
  [EVENT_TYPES.BACKTEST_REALTIME]: '实时回测',
  [EVENT_TYPES.BACKTEST_ADVANCED]: '高级回测',
  [EVENT_TYPES.API_CALL]: 'API 调用',
  [EVENT_TYPES.DATA_DOWNLOAD]: '数据下载',
  [EVENT_TYPES.REALTIME_QUOTE]: '实时行情',
  [EVENT_TYPES.MODEL_TRAINING]: '模型训练',
  [EVENT_TYPES.OPTIMIZATION]: '参数优化',
};
