/**
 * API服务层
 *
 * 封装所有与后端API的通信，包括：
 * - HTTP请求配置
 * - 错误处理
 * - 响应拦截
 * - 请求重试
 */

import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';

// 标准响应格式
export interface ApiResponse<T = unknown> {
  success: boolean;
  message: string;
  data?: T;
  timestamp: string;
}

type RequestParams = Record<string, unknown>;
type ErrorResponseData = { message?: string; [key: string]: unknown };
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type LooseApiPayload = any;
type EnhancedError = Error & {
  status?: number;
  response?: AxiosError['response'];
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

const extractErrorMessage = (data: unknown): string | undefined => {
  if (!data || typeof data !== 'object') {
    return undefined;
  }

  const candidate = data as ErrorResponseData;
  return typeof candidate.message === 'string' ? candidate.message : undefined;
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
      const token = localStorage.getItem('auth_token');
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
    (response: AxiosResponse<ApiResponse>) => {
      const { data } = response;

      apiLogger.debug(
        `[API] 响应成功: ${response.config.method?.toUpperCase()} ${response.config.url}`
      );
      apiLogger.debug(`[API] 响应状态: ${response.status}`);
      apiLogger.debug('[API] 响应数据:', data);

      // 检查业务逻辑错误
      if (!data.success) {
        apiLogger.error('[API] 业务错误:', data.message);
        apiLogger.error(data.message || '请求失败');
        return Promise.reject(new Error(data.message || '请求失败'));
      }

      return response;
    },
    (error: AxiosError) => {
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

      // 处理不同类型的错误
      if (error.response) {
        const status = error.response.status;
        const data = error.response.data;
        const errorMessage = extractErrorMessage(data);

        switch (status) {
          case 400:
            apiLogger.error('请求参数错误:', errorMessage || '请求参数错误');
            break;
          case 401:
            apiLogger.error('未授权访问，请重新登录');
            // 清除token并跳转到登录页
            localStorage.removeItem('auth_token');
            window.location.href = '/login';
            break;
          case 403:
            apiLogger.error('权限不足');
            break;
          case 404:
            apiLogger.error('请求的资源不存在:', `${error.config?.baseURL}${error.config?.url}`);
            break;
          case 429:
            apiLogger.error('请求过于频繁，请稍后再试');
            break;
          case 500:
            apiLogger.error('服务器内部错误:', errorMessage || '服务器内部错误');
            break;
          default:
            apiLogger.error(`请求失败 (${status}):`, errorMessage || '未知错误');
        }
      } else if (error.request) {
        apiLogger.error('网络连接失败，请检查网络设置');
      } else {
        apiLogger.error('请求配置错误:', error.message);
      }

      return Promise.reject(error);
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
      .get<ApiResponse<T>>(url, { params })
      .then(res => {
        apiLogger.debug(`[API] GET ${url} 响应:`, res.data);
        if (!res.data || !res.data.success) {
          apiLogger.error(`[API] GET ${url} 失败:`, res.data?.message || '未知错误');
          throw new Error(res.data?.message || '请求失败');
        }
        if (res.data.data === undefined || res.data.data === null) {
          apiLogger.warn(`[API] GET ${url} 返回的data字段为空`);
          return null as T;
        }
        return res.data.data;
      })
      .catch((error: unknown) => {
        // 确保错误对象包含状态码信息
        if (axios.isAxiosError(error) && error.response) {
          const enhancedError = new Error(
            extractErrorMessage(error.response.data) || error.message
          ) as EnhancedError;
          enhancedError.status = error.response.status;
          enhancedError.response = error.response;
          throw enhancedError;
        }
        throw error;
      });
  },

  post: <T = LooseApiPayload>(url: string, data?: unknown): Promise<T> => {
    return api.post<ApiResponse<T>>(url, data).then(res => {
      if (!res.data || !res.data.success) {
        throw new Error(res.data?.message || '请求失败');
      }
      return res.data.data as T;
    });
  },

  put: <T = LooseApiPayload>(url: string, data?: unknown): Promise<T> => {
    return api.put<ApiResponse<T>>(url, data).then(res => {
      if (!res.data || !res.data.success) {
        throw new Error(res.data?.message || '请求失败');
      }
      return res.data.data as T;
    });
  },

  delete: <T = LooseApiPayload>(url: string): Promise<T> => {
    return api.delete<ApiResponse<T>>(url).then(res => {
      if (!res.data || !res.data.success) {
        throw new Error(res.data?.message || '请求失败');
      }
      return res.data.data as T;
    });
  },

  patch: <T = LooseApiPayload>(url: string, data?: unknown): Promise<T> => {
    return api.patch<ApiResponse<T>>(url, data).then(res => {
      if (!res.data || !res.data.success) {
        throw new Error(res.data?.message || '请求失败');
      }
      return res.data.data as T;
    });
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
    .then(res => res.data.data);
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
