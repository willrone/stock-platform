/**
 * 按使用量计费 API 服务
 *
 * 提供计费事件记录、余额查询、充值、账单历史等接口调用
 */

export interface BalanceResponse {
  user_id: string;
  balance_cents: number;
  balance_yuan: number;
  credited_cents: number;
  total_spent_cents: number;
  updated_at: string;
}

export interface UsageSummaryResponse {
  month: number;
  year: number;
  total_spent_cents: number;
  total_spent_yuan: number;
  events: Record<
    string,
    {
      count: number;
      total_cost: number;
      total_cost_yuan: number;
    }
  >;
}

export interface BillingRecord {
  id: string;
  amount_cents: number;
  charge_status: string;
  payment_method?: string;
  remark?: string;
  charged_at?: string;
  created_at: string;
}

export interface UsageRecordResponse {
  event_id: string;
  event_type: string;
  unit_cost: number;
  quantity: number;
  total_cost: number;
  billing_status: string;
  message: string;
}

export interface CommerceAPI {
  // 记录用量（扣费）
  recordUsage: (
    event_type: string,
    quantity?: number,
    metadata?: Record<string, any>
  ) => Promise<UsageRecordResponse>;

  // 余额查询
  getBalance: () => Promise<BalanceResponse>;

  // 充值
  deposit: (amount: number, remark?: string) => Promise<void>;

  // 退款
  refund: (billing_record_id: string, amount: number) => Promise<void>;

  // 用量汇总
  getUsageSummary: (month?: number, year?: number) => Promise<UsageSummaryResponse>;

  // 账单历史
  getBillingHistory: (limit?: number, offset?: number) => Promise<BillingRecord[]>;

  // 计费规则
  getPricingRules: (event_type?: string) => Promise<any[]>;

  // 检查是否可以执行
  checkCanExecute: (
    event_type: string
  ) => Promise<{ can_proceed: boolean; message: string; estimated_cost_yuan: number }>;
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
