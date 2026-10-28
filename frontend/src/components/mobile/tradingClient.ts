/**
 * 移动端请求封装（trading 系端点专用）
 *
 * 为什么不用 src/services/api.ts：
 *   1. /trading/* 端点后端返回的是**裸对象**（没有 success/data 信封），
 *      而 /api/v1/data/* 等端点返回的是**标准信封**。api.ts 由其他代理
 *      正在并行修改（加响应信封归一化），本文件不依赖它的具体行为，
 *      自己同时兼容两种形态 —— 无论对方最终怎么实现，页面都能正常渲染。
 *   2. 需要把后端 HTTPException 的 `detail` 透传给用户（api.ts 目前只取
 *      `message`），否则用户只会看到「请求失败」这种无信息文案。
 *
 * 如果后续 src/services/api.ts 的信封归一化上线，可以把本文件删掉，
 * 改回 apiRequest.get —— 调用点已经在各页面集中封装为 tradingGet。
 */

export type TradingError = Error & {
  status?: number;
  detail?: unknown;
};

const API_BASE = '/api/v1';

const readToken = (): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }
  return window.localStorage.getItem('access_token');
};

/** 用 refresh_token 换一个新的 access_token（401 时兜底一次） */
const refreshAccessToken = async (): Promise<string | null> => {
  if (typeof window === 'undefined') {
    return null;
  }
  const refreshToken = window.localStorage.getItem('refresh_token');
  if (!refreshToken) {
    return null;
  }
  try {
    const res = await fetch(`${API_BASE}/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    if (!res.ok) {
      return null;
    }
    const body = await res.json();
    const payload = body?.data || body;
    const token = payload?.access_token;
    if (!token) {
      return null;
    }
    window.localStorage.setItem('access_token', token);
    if (payload?.refresh_token) {
      window.localStorage.setItem('refresh_token', payload.refresh_token);
    }
    return token;
  } catch {
    return null;
  }
};

/** 从后端错误体里抽出人类可读的信息（detail 优先） */
export const extractErrorDetail = (body: unknown): string | null => {
  if (!body || typeof body !== 'object') {
    return null;
  }
  const candidate = body as Record<string, unknown>;
  if (typeof candidate.detail === 'string' && candidate.detail) {
    return candidate.detail;
  }
  if (typeof candidate.message === 'string' && candidate.message) {
    return candidate.message;
  }
  return null;
};

const buildError = (status: number, body: unknown, fallback: string): TradingError => {
  const detail = extractErrorDetail(body);
  const message = detail || fallback;
  const err = new Error(message) as TradingError;
  err.status = status;
  err.detail = detail ?? body;
  return err;
};

/**
 * GET /api/v1/<path>
 *
 * - 自动带 access_token
 * - 401 时尝试用 refresh_token 换新 token 并重试一次
 * - 同时兼容「标准信封」{success,message,data} 与「裸对象」两种响应形态
 * - 错误时抛出带 status / detail 的 Error（detail 直接来自后端）
 */
export async function tradingGet<T>(path: string, options?: { retryOn401?: boolean }): Promise<T> {
  const doFetch = async (token: string | null): Promise<Response> =>
    fetch(`${API_BASE}${path}`, {
      method: 'GET',
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      cache: 'no-store',
    });

  let token = readToken();
  let res = await doFetch(token);

  if (res.status === 401 && options?.retryOn401 !== false) {
    const fresh = await refreshAccessToken();
    if (fresh) {
      token = fresh;
      res = await doFetch(token);
    }
  }

  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }

  if (!res.ok) {
    throw buildError(res.status, body, `请求失败（HTTP ${res.status}）`);
  }

  // 标准信封 → 取 data；裸对象 → 原样返回
  if (body && typeof body === 'object' && 'success' in (body as Record<string, unknown>)) {
    const envelope = body as Record<string, unknown>;
    if (envelope.success === false) {
      throw buildError(res.status, body, '请求失败');
    }
    return (envelope.data ?? null) as T;
  }

  return body as T;
}

/**
 * 计算两个日期之间相差的天数（用于「数据已过期」守卫）。
 * dateStr 支持 'YYYY-MM-DD' 或 ISO 字符串。
 */
export const daysSince = (dateStr?: string | null, now: Date = new Date()): number | null => {
  if (!dateStr) {
    return null;
  }
  const d = new Date(dateStr.length <= 10 ? `${dateStr}T00:00:00` : dateStr);
  if (Number.isNaN(d.getTime())) {
    return null;
  }
  const start = new Date(d.getFullYear(), d.getMonth(), d.getDate());
  const end = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  return Math.round((end.getTime() - start.getTime()) / 86400000);
};
