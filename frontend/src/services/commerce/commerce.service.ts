/**
 * 按使用量计费服务层
 *
 * 负责协调计费、余额、账单、用量记录等核心业务逻辑
 * 遵循单一职责原则，便于单元测试和维护
 *
 * ⚠️ 响应形态：/commerce/* 后端端点多数返回**裸 payload**（未统一为
 * {success, data, message, timestamp} 信封）。前端 @/services/api 的归一化层
 * （normalizeResponse / unwrap）已兼容两种形态，因此这里直接按「业务数据本身」
 * 声明类型即可，不需要在这里做信封拆包。
 */

import { apiRequest } from '@/services/api';
import type {
  BalanceResponse,
  BillingRecord,
  CanExecuteResponse,
  CommerceAPI,
  DepositResponse,
  PricingRuleResponse,
  RefundResponse,
  UsageRecordResponse,
  UsageSummaryResponse,
} from '@/services/api/commerce';

export class CommerceService implements CommerceAPI {
  private baseURL = '/commerce';

  /**
   * 记录用量事件并扣费
   * @param event_type - 事件类型（如：backtest_basic, api_call 等）
   * @param quantity - 消耗数量，默认 1
   * @param metadata - 额外元数据
   * @returns 用量记录响应
   */
  async recordUsage(
    event_type: string,
    quantity = 1,
    metadata?: Record<string, unknown>
  ): Promise<UsageRecordResponse> {
    return apiRequest.post<UsageRecordResponse>(`${this.baseURL}/usage`, {
      event_type,
      quantity,
      metadata,
    });
  }

  /**
   * 获取用户余额
   * @returns 余额信息
   */
  async getBalance(): Promise<BalanceResponse> {
    return apiRequest.get<BalanceResponse>(`${this.baseURL}/balance`);
  }

  /**
   * 用户充值（下单）
   *
   * ⚠️ 2026-10-25 起为订单制：下单不立即加余额，需管理员确认到账后才生效。
   * @param amount - 充值金额（元）
   * @param remark - 充值备注
   * @returns 充值订单信息（订单号、收款指引等）
   */
  async deposit(amount: number, remark?: string): Promise<DepositResponse> {
    const amount_cents = Math.round(amount * 100);
    return apiRequest.post<DepositResponse>(`${this.baseURL}/deposit`, {
      amount_cents,
      remark,
    });
  }

  /**
   * 退款
   * @param billing_record_id - 原始账单 ID
   * @param amount - 退款金额（元）
   */
  async refund(billing_record_id: string, amount: number): Promise<RefundResponse> {
    const amount_cents = Math.round(amount * 100);
    return apiRequest.post<RefundResponse>(`${this.baseURL}/refund`, {
      billing_record_id,
      amount_cents,
    });
  }

  /**
   * 获取用量汇总
   * @param month - 月份（可选，默认当月）
   * @param year - 年份（可选，默认当年）
   */
  async getUsageSummary(month?: number, year?: number): Promise<UsageSummaryResponse> {
    const params = new URLSearchParams();
    if (month) {
      params.append('month', month.toString());
    }
    if (year) {
      params.append('year', year.toString());
    }
    const query = params.toString();
    return apiRequest.get<UsageSummaryResponse>(
      query ? `${this.baseURL}/usage/summary?${query}` : `${this.baseURL}/usage/summary`
    );
  }

  /**
   * 获取账单历史
   * @param limit - 限制条数
   * @param offset - 偏移量
   */
  async getBillingHistory(limit = 20, offset = 0): Promise<BillingRecord[]> {
    const params = new URLSearchParams();
    params.append('limit', limit.toString());
    params.append('offset', offset.toString());
    return apiRequest.get<BillingRecord[]>(`${this.baseURL}/billing/history?${params}`);
  }

  /**
   * 获取计费规则
   * @param event_type - 事件类型筛选
   */
  async getPricingRules(event_type?: string): Promise<PricingRuleResponse[]> {
    const params = new URLSearchParams();
    if (event_type) {
      params.append('event_type', event_type);
    }
    const query = params.toString();
    return apiRequest.get<PricingRuleResponse[]>(
      query ? `${this.baseURL}/pricing/rules?${query}` : `${this.baseURL}/pricing/rules`
    );
  }

  /**
   * 检查是否可以执行操作
   * @param event_type - 事件类型
   */
  async checkCanExecute(event_type: string): Promise<CanExecuteResponse> {
    return apiRequest.get<CanExecuteResponse>(
      `${this.baseURL}/check/${encodeURIComponent(event_type)}`
    );
  }
}

// 创建单例服务
export const commerceService = new CommerceService();
