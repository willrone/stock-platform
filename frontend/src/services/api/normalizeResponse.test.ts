/**
 * 响应归一化兼容层单元测试
 *
 * 覆盖后端两种响应形态：
 *   1. 标准信封 { success, data, message, timestamp }
 *   2. 裸 payload（/commerce/*、/trading/* 尚未统一信封）
 *
 * 详见 @/services/api 顶部的临时兼容层说明。
 */

import {
  normalizeResponse,
  unwrap,
  extractBackendErrorMessage,
  buildErrorMessage,
  ApiRequestError,
} from '../api';

describe('normalizeResponse 归一化层', () => {
  describe('标准信封（保持既有语义）', () => {
    it('原样保留 success / message / timestamp / data', () => {
      const body = {
        success: true,
        message: '获取成功',
        data: { strategies: [{ key: 'honest_42f' }], count: 1 },
        timestamp: '2026-10-27T00:00:00',
      };

      const normalized = normalizeResponse(body);

      expect(normalized.success).toBe(true);
      expect(normalized.message).toBe('获取成功');
      expect(normalized.timestamp).toBe('2026-10-27T00:00:00');
      expect(normalized.data).toBe(body.data);
      // 标准信封不提升业务字段到顶层，避免改变 res.data 语义
      expect((normalized as unknown as Record<string, unknown>).strategies).toBeUndefined();
    });

    it('success=false 时不提升字段，且仍为失败态', () => {
      const normalized = normalizeResponse({
        success: false,
        message: '余额不足',
        data: null,
        timestamp: '2026-10-27T00:00:00',
      });

      expect(normalized.success).toBe(false);
      expect(normalized.message).toBe('余额不足');
    });

    it('缺 timestamp 时补齐 ISO 时间戳', () => {
      const normalized = normalizeResponse({ success: true, message: 'ok', data: 1 });
      expect(typeof normalized.timestamp).toBe('string');
      expect(normalized.timestamp.length).toBeGreaterThan(0);
    });
  });

  describe('裸 payload（不判失败，包成信封）', () => {
    it('/trading/strategies 形态：顶层与 data 嵌套都能取到业务字段', () => {
      const body = {
        strategies: [{ key: 'honest_42f' }],
        count: 1,
        tier: 'free',
        tier_label: '体验版',
        allowed_strategies: ['honest_42f'],
        delay_days: 1,
        include_execution_detail: false,
      };

      const normalized = normalizeResponse(body);

      // 不判失败
      expect(normalized.success).toBe(true);
      expect(normalized.message).toBe('OK');
      expect(typeof normalized.timestamp).toBe('string');

      // 写法一：res.data.xxx（业务字段被提升到顶层）
      const top = normalized as unknown as Record<string, unknown>;
      expect(top.strategies).toBe(body.strategies);
      expect(top.tier).toBe('free');
      expect(top.delay_days).toBe(1);

      // 写法二：res.data.data.xxx（data 嵌套保留）
      expect((normalized.data as Record<string, unknown>).strategies).toBe(body.strategies);
      expect((normalized.data as Record<string, unknown>).tier).toBe('free');
    });

    it('/commerce/payment/orders 形态：{orders, count}', () => {
      const body = { orders: [{ order_no: 'PO1' }], count: 1 };

      const normalized = normalizeResponse(body);

      expect(normalized.success).toBe(true);
      expect((normalized.data as Record<string, unknown>).orders).toBe(body.orders);
      expect((normalized as unknown as Record<string, unknown>).orders).toBe(body.orders);
    });

    it('/auth/login 形态：裸 token 对象', () => {
      const body = {
        access_token: 'at',
        refresh_token: 'rt',
        token_type: 'bearer',
        user: { id: 'u1' },
      };

      const normalized = normalizeResponse(body);

      expect(normalized.success).toBe(true);
      expect((normalized.data as Record<string, unknown>).access_token).toBe('at');
      expect((normalized as unknown as Record<string, unknown>).token_type).toBe('bearer');
    });

    it('/commerce/balance 形态：裸 Pydantic 模型', () => {
      const body = {
        user_id: 'u1',
        balance_cents: 1234,
        balance_yuan: 12.34,
        credited_cents: 0,
        total_spent_cents: 0,
        updated_at: '2026-10-27T00:00:00',
      };

      const normalized = normalizeResponse(body);

      expect(normalized.success).toBe(true);
      expect((normalized.data as Record<string, unknown>).balance_yuan).toBe(12.34);
    });

    it('裸 payload 自带 data 字段时保留后端原值，不被覆盖', () => {
      const body = { data: [1, 2, 3], count: 3 };

      const normalized = normalizeResponse(body);

      expect((normalized.data as Record<string, unknown>).data).toEqual([1, 2, 3]);
    });
  });

  describe('非对象裸 payload（数组 / 标量）', () => {
    it('数组包成 data', () => {
      const normalized = normalizeResponse([{ id: 'a' }, { id: 'b' }]);

      expect(normalized.success).toBe(true);
      expect(normalized.data).toHaveLength(2);
    });

    it('字符串包成 data', () => {
      const normalized = normalizeResponse('pong');
      expect(normalized.success).toBe(true);
      expect(normalized.data).toBe('pong');
    });

    it('null 包成 data', () => {
      const normalized = normalizeResponse(null);
      expect(normalized.success).toBe(true);
      expect(normalized.data).toBeNull();
    });
  });
});

describe('unwrap 提取业务数据', () => {
  it('从标准信封取 data', () => {
    expect(unwrap<{ id: string }>({ success: true, message: 'ok', data: { id: 'x' } })).toEqual({
      id: 'x',
    });
  });

  it('从归一化后的裸 payload 响应体取业务数据', () => {
    const normalized = normalizeResponse({ strategies: [], count: 0 });
    expect(unwrap<{ count: number }>(normalized).count).toBe(0);
  });

  it('裸 payload 直接传入时原样返回', () => {
    expect(unwrap<{ orders: number[] }>({ orders: [1] })).toEqual({ orders: [1] });
  });

  it('axios 响应对象会先取 .data 再解包', () => {
    const axiosLike = {
      data: { success: true, message: 'ok', data: { id: 'y' } },
      status: 200,
      config: {},
    };
    expect(unwrap<{ id: string }>(axiosLike)).toEqual({ id: 'y' });
  });

  it('null / undefined 安全返回', () => {
    expect(unwrap(null)).toBeNull();
    expect(unwrap(undefined)).toBeUndefined();
  });
});

describe('错误信息提取与构造', () => {
  it('提取 FastAPI HTTPException 的 detail', () => {
    expect(extractBackendErrorMessage({ detail: '免费用户只能访问 1 条策略线' })).toBe(
      '免费用户只能访问 1 条策略线'
    );
  });

  it('提取标准信封失败时的 message', () => {
    expect(extractBackendErrorMessage({ success: false, message: '余额不足' })).toBe('余额不足');
  });

  it('提取 422 校验错误数组并拼接', () => {
    const detail = [
      { loc: ['body', 'amount_cents'], msg: 'field required', type: 'value_error.missing' },
    ];
    expect(extractBackendErrorMessage({ detail })).toBe('amount_cents: field required');
  });

  it('无内容时返回 undefined', () => {
    expect(extractBackendErrorMessage(undefined)).toBeUndefined();
    expect(extractBackendErrorMessage({})).toBeUndefined();
  });

  it('按状态码生成中文文案，并合并后端 detail', () => {
    expect(buildErrorMessage(429)).toBe('请求过于频繁，请稍后重试');
    expect(buildErrorMessage(403, '需要升级套餐')).toBe('权限不足，无法执行该操作：需要升级套餐');
    expect(buildErrorMessage(500)).toBe('服务器内部错误');
    expect(buildErrorMessage(undefined)).toBe('请求失败');
  });

  it('后端 detail 与状态码文案重复时不重复拼接', () => {
    // 后端限流中间件返回的 detail 就是这句文案
    expect(buildErrorMessage(429, '请求过于频繁，请稍后再试')).toBe('请求过于频繁，请稍后重试');
  });

  it('ApiRequestError 保留 status 且是 Error 实例', () => {
    const err = new ApiRequestError('请求过于频繁，请稍后重试', 429);
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(ApiRequestError);
    expect(err.status).toBe(429);
    expect(err.message).toBe('请求过于频繁，请稍后重试');
  });
});
