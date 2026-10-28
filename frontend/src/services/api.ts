/**
 * API服务层
 *
 * 封装所有与后端API的通信，包括：
 * - HTTP请求配置
 * - 错误处理
 * - 响应拦截
 * - 请求重试
 *
 * ─────────────────────────────────────────────────────────────────────────
 * ⚠️ 临时兼容层：响应信封归一化（后端统一信封后应删除本段逻辑）
 * ─────────────────────────────────────────────────────────────────────────
 * 后端目前同时存在两套响应格式：
 *   1. 标准信封（40+ 模块走这个）：{ success, message, data, timestamp }
 *   2. 裸 payload（直接返回业务对象本身），主要集中在 /commerce/* 与 /trading/* 等模块：
 *        GET  /api/v1/trading/strategies       → { strategies, count, tier, ... }
 *        GET  /api/v1/commerce/payment/orders   → { orders, count }
 *        GET  /api/v1/commerce/balance          → 裸 Pydantic 模型
 *        GET  /api/v1/commerce/billing/history  → 裸数组
 *        POST /api/v1/auth/login                → { access_token, refresh_token, ... }
 *
 * 早期前端拦截器硬性要求响应体带 `success` 为真，把上述裸 payload 端点
 * 全部误判成「业务失败」，用户只会看到一句无用的「请求失败」。
 *
 * 本文件在响应拦截器里引入**单一归一化层**（normalizeResponse），让两种形态
 * 都能被同一套调用代码消费：
 *   - 对象且含布尔型 success → 按标准信封处理（success=false 仍然 reject 并抛出后端 message）
 *   - 其余（裸对象 / 数组 / 标量）→ 视为裸 payload，**不判失败**，包成
 *     { success: true, data: <payload>, message: 'OK', timestamp: <ISO> }，
 *     并把 payload 的业务字段同时提升到顶层，使 `res.data.xxx` 与 `res.data.data.xxx`
 *     两种访问写法都能取到值；标准信封接口的 res.data / res.data.data 语义保持不变。
 *
 * 【待办】这是临时兼容层：等后端把 /commerce/* 与 /trading/* 统一成标准信封后，
 * 应删除 normalizeResponse / unwrap 及其在拦截器里的调用，届时前端只保留
 * `res.data.data` 一种访问方式。
 */

import axios, {
  AxiosInstance,
  AxiosResponse,
  AxiosError,
  InternalAxiosRequestConfig,
} from 'axios';

// 标准响应格式（对应后端 StandardResponse）
export interface ApiResponse<T = unknown> {
  success: boolean;
  message: string;
  data?: T;
  timestamp: string;
}

/**
 * 归一化后的响应体。
 * 无论后端返回标准信封还是裸 payload，归一化后都保证
 * success / message / timestamp / data 四个字段存在；
 * 裸 payload 的业务字段另外被提升到顶层（索引签名），
 * 便于 `res.data.xxx` 与 `res.data.data.xxx` 两种访问写法都能取到值。
 */
export interface NormalizedApiResponse<T = unknown> {
  success: boolean;
  message: string;
  timestamp: string;
  data: T;
  [key: string]: unknown;
}

/**
 * 统一的后端错误对象：
 * - message 为可直接展示给用户的中文文案（已合并后端 detail / message）
 * - status 保留 HTTP 状态码，方便调用方按状态码分支处理
 */
export class ApiRequestError extends Error {
  readonly status?: number;
  readonly response?: AxiosError['response'];

  constructor(message: string, status?: number, response?: AxiosError['response']) {
    super(message);
    this.name = 'ApiRequestError';
    this.status = status;
    this.response = response;
    // tsconfig target=es5：继承内置 Error 后必须手动修正原型链，
    // 否则 `err instanceof Error` / `err instanceof ApiRequestError` 恒为 false
    Object.setPrototypeOf(this, ApiRequestError.prototype);
  }
}

type RequestParams = Record<string, unknown>;
type ErrorResponseData = { message?: string; detail?: unknown; [key: string]: unknown };
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type LooseApiPayload = any;

type RetryableRequestConfig = InternalAxiosRequestConfig & {
  _retry?: boolean;
};

type RefreshResponse = {
  access_token?: string;
  refresh_token?: string;
  data?: {
    access_token?: string;
    refresh_token?: string;
  };
};

type QueuedRequest = {
  resolve: (token: string) => void;
  reject: (error: unknown) => void;
};

/** HTTP 状态码 → 可直接展示给用户的中文文案 */
const STATUS_MESSAGES: Record<number, string> = {
  400: '请求参数有误',
  401: '登录状态已失效，请重新登录',
  403: '权限不足，无法执行该操作',
  404: '请求的资源不存在',
  405: '请求方法不被允许',
  408: '请求超时',
  409: '请求冲突，请刷新后重试',
  422: '请求参数校验失败',
  429: '请求过于频繁，请稍后重试',
  500: '服务器内部错误',
  502: '网关错误',
  503: '服务暂时不可用，请稍后重试',
  504: '网关超时',
};

/** 后端错误文案最大长度，避免把整段 HTML 塞进 UI */
const MAX_ERROR_MESSAGE_LENGTH = 300;

/** Token 刷新重试策略：最多重试 2 次，退避 1s / 2s */
const REFRESH_MAX_RETRIES = 2;
const REFRESH_BACKOFF_MS = [1000, 2000];

let isRefreshing = false;
let refreshQueue: QueuedRequest[] = [];

const isPlainObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** 是否是 axios 响应对象（而不是响应体本身） */
const isAxiosResponseLike = (value: unknown): boolean =>
  isPlainObject(value) &&
  'data' in value &&
  typeof (value as { status?: unknown }).status === 'number' &&
  'config' in value;

/**
 * 是否是后端标准信封（含布尔型 success 字段）。
 * 刻意只返回 boolean 而不是类型谓词：对 unknown 使用类型谓词会把 else 分支
 * 收窄成 never，导致后续裸 payload 分支无法通过类型检查。
 */
const hasBooleanSuccess = (value: unknown): boolean =>
  isPlainObject(value) && typeof (value as { success?: unknown }).success === 'boolean';

const getStoredToken = (key: string): string | null => {
  if (typeof window === 'undefined') {
    return null;
  }
  return window.localStorage.getItem(key);
};

const clearStoredAuth = (): void => {
  if (typeof window === 'undefined') {
    return;
  }
  window.localStorage.removeItem('access_token');
  window.localStorage.removeItem('refresh_token');
  window.localStorage.removeItem('user_info');
};

const redirectToLogin = (): void => {
  if (typeof window !== 'undefined' && window.location.pathname !== '/login') {
    window.location.assign('/login');
  }
};

const flushRefreshQueue = (error: unknown, token?: string): void => {
  refreshQueue.forEach(({ resolve, reject }) => {
    if (token) {
      resolve(token);
    } else {
      reject(error);
    }
  });
  refreshQueue = [];
};

const delay = (ms: number): Promise<void> =>
  new Promise<void>(resolve => {
    setTimeout(resolve, ms);
  });

/** 限流（429）判定：刷新失败时不能因此把用户踢下线 */
const isRateLimited = (error: unknown): boolean => {
  if (error instanceof ApiRequestError) {
    return error.status === 429;
  }
  if (axios.isAxiosError(error)) {
    return error.response?.status === 429;
  }
  return false;
};

/** 从任意错误对象里取出 HTTP 状态码 */
const extractErrorStatus = (error: unknown): number | undefined => {
  if (error instanceof ApiRequestError) {
    return error.status;
  }
  if (axios.isAxiosError(error)) {
    return error.response?.status;
  }
  return undefined;
};

/**
 * 刷新 access token。
 * 限流（429）与服务端错误（5xx）时按指数退避重试，最多 2 次（1s / 2s）。
 */
const refreshAccessToken = async (): Promise<string> => {
  const refreshToken = getStoredToken('refresh_token');
  if (!refreshToken) {
    throw new ApiRequestError('缺少 refresh token，请重新登录', 401);
  }

  let lastError: unknown = new ApiRequestError('刷新 token 失败', 401);

  for (let attempt = 0; attempt <= REFRESH_MAX_RETRIES; attempt += 1) {
    try {
      // Use the bare axios client so a failed refresh cannot recursively trigger
      // this response interceptor.
      const response = await axios.post<RefreshResponse>('/api/v1/auth/refresh', {
        refresh_token: refreshToken,
      });
      // 后端 /auth/refresh 返回的是裸 AuthResponse（无信封），同时也兼容带信封的形态
      const payload = response.data?.data || response.data;
      const accessToken = payload?.access_token;
      if (!accessToken) {
        throw new ApiRequestError('刷新 token 响应无效', 401);
      }

      localStorage.setItem('access_token', accessToken);
      if (payload.refresh_token) {
        localStorage.setItem('refresh_token', payload.refresh_token);
      }
      return accessToken;
    } catch (error) {
      lastError = error;
      const status = extractErrorStatus(error);
      // 仅对「限流 / 服务端错误 / 网络错误」退避重试；401、400 这类确定性失败立即抛出
      const retryable = status === undefined || status === 429 || status >= 500;
      if (!retryable || attempt === REFRESH_MAX_RETRIES) {
        break;
      }
      const backoff = REFRESH_BACKOFF_MS[attempt] ?? REFRESH_BACKOFF_MS[REFRESH_BACKOFF_MS.length - 1];
      await delay(backoff);
    }
  }

  throw lastError;
};

const apiLogger = {
  debug: (...args: unknown[]) => {
    if (process.env.NODE_ENV !== 'production') {
      globalThis.console.log(...args);
    }
  },
  warn: (...args: unknown[]) => {
    globalThis.console.warn(...args);
  },
  error: (...args: unknown[]) => {
    globalThis.console.error(...args);
  },
};

const stripHtml = (text: string): string => text.replace(/<[^>]*>/g, ' ');

/**
 * 从后端错误响应体里提取可读文案。
 * 覆盖三种常见形态：
 *   - FastAPI HTTPException：{ detail: "权限不足" }
 *   - FastAPI 校验错误：{ detail: [{ loc, msg, type }] }
 *   - 标准信封失败：{ success: false, message: "..." }
 */
export const extractBackendErrorMessage = (data: unknown): string | undefined => {
  if (typeof data === 'string') {
    const text = stripHtml(data).trim();
    return text ? text.slice(0, MAX_ERROR_MESSAGE_LENGTH) : undefined;
  }

  if (!isPlainObject(data)) {
    return undefined;
  }

  const detail = (data as ErrorResponseData).detail;
  if (typeof detail === 'string' && detail.trim()) {
    return detail.trim().slice(0, MAX_ERROR_MESSAGE_LENGTH);
  }
  if (Array.isArray(detail)) {
    const parts = detail
      .map(item => {
        if (typeof item === 'string') {
          return item.trim();
        }
        if (!isPlainObject(item)) {
          return '';
        }
        const msg = typeof item.msg === 'string' ? item.msg.trim() : '';
        const loc = Array.isArray(item.loc)
          ? item.loc.filter(part => part !== 'body').join('.')
          : '';
        if (msg && loc) {
          return `${loc}: ${msg}`;
        }
        return msg;
      })
      .filter(Boolean);
    if (parts.length) {
      return parts.join('；').slice(0, MAX_ERROR_MESSAGE_LENGTH);
    }
  }

  const message = (data as ErrorResponseData).message;
  if (typeof message === 'string' && message.trim()) {
    return message.trim().slice(0, MAX_ERROR_MESSAGE_LENGTH);
  }

  const errorField = (data as ErrorResponseData).error;
  if (typeof errorField === 'string' && errorField.trim()) {
    return errorField.trim().slice(0, MAX_ERROR_MESSAGE_LENGTH);
  }

  return undefined;
};

/**
 * 两段文案的最长公共子串长度占比（0~1），用于判断是否为「同一句话的微调」。
 * 用于错误文案去重：后端限流中间件的 detail 是「请求过于频繁，请稍后再试」，
 * 而状态码文案是「请求过于频繁，请稍后重试」，只差一个「再」字。
 */
const messageSimilarity = (a: string, b: string): number => {
  if (!a || !b) {
    return 0;
  }
  // 先去掉标点与空白，避免「，」这类差异拉低相似度
  const norm = (text: string): string => text.replace(/[\s，。、；：！？,.!?;:]/g, '');
  const s = norm(a);
  const t = norm(b);
  if (!s || !t) {
    return 0;
  }
  // 动态规划求最长公共子串
  let prev = new Array<number>(t.length + 1).fill(0);
  let best = 0;
  for (let i = 1; i <= s.length; i += 1) {
    const curr = new Array<number>(t.length + 1).fill(0);
    for (let j = 1; j <= t.length; j += 1) {
      if (s[i - 1] === t[j - 1]) {
        curr[j] = prev[j - 1] + 1;
        if (curr[j] > best) {
          best = curr[j];
        }
      }
    }
    prev = curr;
  }
  return best / Math.max(s.length, t.length);
};

/**
 * 组装最终抛给调用方的错误文案：状态码中文说明 + 后端 detail/message
 *
 * 去重：后端限流/鉴权中间件返回的 detail 往往就是状态码文案本身
 * （如 429 → "请求过于频繁，请稍后再试"），直接拼接会变成
 * 「请求过于频繁，请稍后重试：请求过于频繁，请稍后再试」，这里做一次收敛。
 */
export const buildErrorMessage = (status: number | undefined, backendMessage?: string): string => {
  const base = status !== undefined ? STATUS_MESSAGES[status] : undefined;
  const fallback = base ?? (status !== undefined ? `请求失败（HTTP ${status}）` : '请求失败');
  if (!backendMessage) {
    return fallback;
  }
  // 后端文案与状态码文案实质重复时只保留其一
  if (
    backendMessage === fallback ||
    fallback.includes(backendMessage) ||
    backendMessage.includes(fallback) ||
    messageSimilarity(fallback, backendMessage) >= 0.8
  ) {
    return fallback;
  }
  return `${fallback}：${backendMessage}`;
};

/**
 * 构造「业务 payload 视图」。
 *
 * 视图 = 裸 payload 的全部字段 + 一个指向自身的 `data` 别名（非枚举）。
 * 这样同一份业务数据可以同时通过三种写法取到：
 *   view.xxx / view.data.xxx / view.data.data.xxx
 *
 * `data` 用非枚举的 getter 实现（而不是普通自引用属性），
 * 这样 JSON.stringify 不会因为循环引用抛错。
 */
const createPayloadView = <T>(payload: Record<string, unknown>): T => {
  const view: Record<string, unknown> = { ...payload };
  // 裸 payload 自带 data 字段时保留原值（不覆盖后端语义）
  if (!('data' in view)) {
    Object.defineProperty(view, 'data', {
      get: () => view,
      enumerable: false,
      configurable: true,
    });
  }
  return view as unknown as T;
};

/**
 * 【临时兼容层】把任意形态的后端响应体归一化成统一信封。
 *
 * 规则：
 *   1. 对象且含布尔型 success → 标准信封，原样返回（仅在缺 timestamp 时补齐）
 *   2. 裸对象（无 success 字段）→ 不判失败，包成
 *      { success: true, data: payload, message: 'OK', timestamp: <ISO> }，
 *      并把 payload 的业务字段提升到顶层
 *   3. 数组 / 字符串 / 数字等非对象裸 payload → 同上包成 data
 *
 * 归一化后（normalized 相当于 axios 响应体 res.data）：
 *   - `res.data.xxx`            → 业务字段（裸 payload 会被提升到顶层）
 *   - `res.data.data.xxx`       → 业务字段（data 嵌套保留）
 *   - `res.data.data.data.xxx`  → 业务字段
 * 标准信封接口的 res.data / res.data.data 语义保持不变。
 */
export const normalizeResponse = <T = unknown>(
  response: unknown
): NormalizedApiResponse<T> => {
  // 既支持传响应体，也支持直接传 axios 响应对象
  const body = isAxiosResponseLike(response)
    ? (response as { data: unknown }).data
    : response;
  const timestamp = new Date().toISOString();

  // 规则 1：标准信封
  if (hasBooleanSuccess(body)) {
    const envelope = body as unknown as NormalizedApiResponse<T>;
    // 标准信封语义保持不变：data 仍是原来的业务数据，只在缺 timestamp 时补齐
    return {
      ...envelope,
      timestamp:
        typeof envelope.timestamp === 'string' ? envelope.timestamp : timestamp,
    };
  }

  // 规则 2：裸对象 payload
  if (isPlainObject(body)) {
    const view = createPayloadView<T>(body);
    return {
      ...(view as unknown as Record<string, unknown>),
      success: true,
      message: 'OK',
      timestamp,
      data: view,
    };
  }

  // 规则 3：数组 / 标量等非对象裸 payload
  return {
    success: true,
    message: 'OK',
    timestamp,
    data: body as T,
  };
};

/**
 * 【临时兼容层】从任意形态的响应中取出业务数据 T。
 *
 * 支持的输入：
 *   - axios 响应对象 → 取 .data 后继续归一化
 *   - 标准信封（含布尔 success）→ 取 .data
 *   - 归一化后的响应体 → 取 .data
 *   - 其他（裸 payload / 标量）→ 原样返回
 */
export const unwrap = <T = unknown>(response: unknown): T => {
  if (response === null || response === undefined) {
    return response as T;
  }

  if (hasBooleanSuccess(response)) {
    const envelope = response as unknown as { data?: T };
    return envelope.data as T;
  }

  if (isAxiosResponseLike(response)) {
    return unwrap<T>((response as { data: unknown }).data);
  }

  return response as T;
};

// 创建axios实例
// 使用相对路径，通过Next.js代理转发到后端
const createApiInstance = (): AxiosInstance => {
  const instance = axios.create({
    baseURL: '/api/v1', // 使用相对路径，通过Next.js rewrites代理
    timeout: 300000, // 增加到5分钟，用于长时间操作如数据同步
    headers: {
      'Content-Type': 'application/json',
    },
    // 配置参数序列化：FastAPI期望数组参数格式为 ?key=a&key=b，而不是 ?key[]=a&key[]=b
    paramsSerializer: (params: RequestParams) => {
      const searchParams = new URLSearchParams();
      for (const [key, value] of Object.entries(params)) {
        if (value === null || value === undefined) {
          continue;
        }
        if (Array.isArray(value)) {
          // 数组参数：每个值作为一个独立的 key=value
          value.forEach(item => {
            searchParams.append(key, String(item));
          });
        } else {
          searchParams.append(key, String(value));
        }
      }
      return searchParams.toString();
    },
  });

  // 请求拦截器
  instance.interceptors.request.use(
    config => {
      // 添加认证token（如果有）
      const token = getStoredToken('access_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }

      apiLogger.debug(
        `[API] 发起请求: ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`
      );
      apiLogger.debug('[API] 请求参数:', config.params);
      apiLogger.debug('[API] 请求头:', config.headers);
      return config;
    },
    error => {
      apiLogger.error('[API] 请求拦截器错误:', error);
      return Promise.reject(error);
    }
  );

  // 响应拦截器
  instance.interceptors.response.use(
    (response: AxiosResponse<unknown>) => {
      apiLogger.debug(
        `[API] 响应成功: ${response.config.method?.toUpperCase()} ${response.config.url}`
      );
      apiLogger.debug(`[API] 响应状态: ${response.status}`);
      apiLogger.debug('[API] 原始响应数据:', response.data);

      // 【临时兼容层】统一归一化：标准信封与裸 payload 都能正常往下走
      const normalized = normalizeResponse(response.data);
      response.data = normalized;

      // 检查业务逻辑错误（仅标准信封的 success=false 才算业务失败）
      if (!normalized.success) {
        const message = normalized.message || '请求失败';
        apiLogger.error('[API] 业务错误:', message);
        return Promise.reject(
          new ApiRequestError(message, response.status, response as AxiosError['response'])
        );
      }

      return response;
    },
    async (error: AxiosError) => {
      apiLogger.error('[API] 响应错误详情:', {
        message: error.message,
        code: error.code,
        config: {
          method: error.config?.method,
          url: error.config?.url,
          baseURL: error.config?.baseURL,
          fullURL: `${error.config?.baseURL}${error.config?.url}`,
        },
        response: {
          status: error.response?.status,
          statusText: error.response?.statusText,
          data: error.response?.data,
        },
      });

      // 网络错误 / 请求配置错误：没有 response
      if (!error.response) {
        const message = error.request
          ? '网络连接失败，请检查网络设置'
          : `请求配置错误：${error.message}`;
        apiLogger.error(message);
        return Promise.reject(new ApiRequestError(message));
      }

      const status = error.response.status;
      const backendMessage = extractBackendErrorMessage(error.response.data);

      const originalRequest = error.config as RetryableRequestConfig | undefined;
      const isRefreshRequest = originalRequest?.url?.includes('/auth/refresh');

      // 401：先尝试刷新 token 再重放原请求
      if (status === 401 && originalRequest && !originalRequest._retry && !isRefreshRequest) {
        originalRequest._retry = true;

        if (isRefreshing) {
          try {
            const token = await new Promise<string>((resolve, reject) => {
              refreshQueue.push({ resolve, reject });
            });
            originalRequest.headers.Authorization = `Bearer ${token}`;
            return instance(originalRequest);
          } catch (refreshError) {
            return Promise.reject(refreshError);
          }
        }

        isRefreshing = true;
        try {
          const token = await refreshAccessToken();
          flushRefreshQueue(undefined, token);
          originalRequest.headers.Authorization = `Bearer ${token}`;
          return instance(originalRequest);
        } catch (refreshError) {
          const refreshStatus = extractErrorStatus(refreshError);
          const refreshDetail = axios.isAxiosError(refreshError)
            ? extractBackendErrorMessage(refreshError.response?.data)
            : refreshError instanceof ApiRequestError
              ? refreshError.message
              : undefined;
          const refreshApiError = new ApiRequestError(
            buildErrorMessage(refreshStatus ?? status, refreshDetail),
            refreshStatus ?? status,
            error.response
          );
          flushRefreshQueue(refreshApiError);
          // 限流不是登录态问题：不清理 token、不跳登录，只提示稍后重试
          if (!isRateLimited(refreshError)) {
            clearStoredAuth();
            redirectToLogin();
          }
          return Promise.reject(refreshApiError);
        } finally {
          isRefreshing = false;
        }
      }

      // 429 限流：保留登录态，不清理 token、不跳登录
      if (status === 429) {
        apiLogger.error('请求过于频繁，请稍后重试');
        return Promise.reject(
          new ApiRequestError(buildErrorMessage(429, backendMessage), 429, error.response)
        );
      }

      if (status === 401) {
        apiLogger.error('未授权访问，请重新登录');
        // 清除token并跳转到登录页
        clearStoredAuth();
        redirectToLogin();
      }

      switch (status) {
        case 400:
          apiLogger.error('请求参数错误:', backendMessage || '请求参数错误');
          break;
        case 403:
          apiLogger.error('权限不足');
          break;
        case 404:
          apiLogger.error('请求的资源不存在:', `${error.config?.baseURL}${error.config?.url}`);
          break;
        case 500:
          apiLogger.error('服务器内部错误:', backendMessage || '服务器内部错误');
          break;
        default:
          apiLogger.error(`请求失败 (${status}):`, backendMessage || '未知错误');
      }

      // 抛出带中文文案与 status 的错误，替代原来只有一句「Request failed with status code XXX」
      return Promise.reject(
        new ApiRequestError(buildErrorMessage(status, backendMessage), status, error.response)
      );
    }
  );

  return instance;
};

// 创建API实例
export const api = createApiInstance();

// 通用请求方法
export const apiRequest = {
  get: <T = LooseApiPayload>(url: string, params?: RequestParams): Promise<T> => {
    return api
      .get<unknown>(url, { params })
      .then(res => {
        apiLogger.debug(`[API] GET ${url} 响应:`, res.data);
        // 【临时兼容层】unwrap 兼容标准信封与裸 payload 两种形态
        const payload = unwrap<T>(res.data);
        if (payload === undefined || payload === null) {
          apiLogger.warn(`[API] GET ${url} 返回的data字段为空`);
          return null as T;
        }
        return payload;
      })
      .catch((error: unknown) => {
        // 确保错误对象包含状态码信息
        if (error instanceof ApiRequestError) {
          throw error;
        }
        if (axios.isAxiosError(error) && error.response) {
          const enhancedError = new ApiRequestError(
            buildErrorMessage(
              error.response.status,
              extractBackendErrorMessage(error.response.data)
            ),
            error.response.status,
            error.response
          );
          throw enhancedError;
        }
        throw error;
      });
  },

  post: <T = LooseApiPayload>(url: string, data?: unknown): Promise<T> => {
    return api.post<unknown>(url, data).then(res => unwrap<T>(res.data));
  },

  put: <T = LooseApiPayload>(url: string, data?: unknown): Promise<T> => {
    return api.put<unknown>(url, data).then(res => unwrap<T>(res.data));
  },

  delete: <T = LooseApiPayload>(url: string): Promise<T> => {
    return api.delete<unknown>(url).then(res => unwrap<T>(res.data));
  },

  patch: <T = LooseApiPayload>(url: string, data?: unknown): Promise<T> => {
    return api.patch<unknown>(url, data).then(res => unwrap<T>(res.data));
  },
};

// 文件上传
export const uploadFile = async (
  url: string,
  file: File,
  onProgress?: (progress: number) => void
): Promise<unknown> => {
  const formData = new FormData();
  formData.append('file', file);

  return api
    .post(url, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: progressEvent => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    })
    .then(res => unwrap<unknown>(res.data));
};

// 健康检查
export const healthCheck = async (): Promise<boolean> => {
  try {
    await api.get('/health');
    return true;
  } catch (error) {
    return false;
  }
};
