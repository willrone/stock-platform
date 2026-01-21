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
export interface ApiResponse<T = any> {
  success: boolean;
  message: string;
  data?: T;
  timestamp: string;
}

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
    paramsSerializer: (params) => {
      const searchParams = new URLSearchParams();
      for (const [key, value] of Object.entries(params)) {
        if (value === null || value === undefined) {
          continue;
        }
        if (Array.isArray(value)) {
          // 数组参数：每个值作为一个独立的 key=value
          value.forEach((item) => {
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
    (config) => {
      // 添加认证token（如果有）
      const token = localStorage.getItem('auth_token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      
      console.log(`[API] 发起请求: ${config.method?.toUpperCase()} ${config.baseURL}${config.url}`);
      console.log(`[API] 请求参数:`, config.params);
      console.log(`[API] 请求头:`, config.headers);
      return config;
    },
    (error) => {
      console.error('[API] 请求拦截器错误:', error);
      return Promise.reject(error);
    }
  );

  // 响应拦截器
  instance.interceptors.response.use(
    (response: AxiosResponse<ApiResponse>) => {
      const { data } = response;
      
      console.log(`[API] 响应成功: ${response.config.method?.toUpperCase()} ${response.config.url}`);
      console.log(`[API] 响应状态: ${response.status}`);
      console.log(`[API] 响应数据:`, data);
      
      // 检查业务逻辑错误
      if (!data.success) {
        console.error('[API] 业务错误:', data.message);
        console.error(data.message || '请求失败');
        return Promise.reject(new Error(data.message || '请求失败'));
      }
      
      return response;
    },
    (error: AxiosError) => {
      console.error('[API] 响应错误详情:', {
        message: error.message,
        code: error.code,
        config: {
          method: error.config?.method,
          url: error.config?.url,
          baseURL: error.config?.baseURL,
          fullURL: `${error.config?.baseURL}${error.config?.url}`
        },
        response: {
          status: error.response?.status,
          statusText: error.response?.statusText,
          data: error.response?.data
        }
      });
      
      // 处理不同类型的错误
      if (error.response) {
        const status = error.response.status;
        const data = error.response.data as any;
        
        switch (status) {
          case 400:
            console.error('请求参数错误:', data?.message || '请求参数错误');
            break;
          case 401:
            console.error('未授权访问，请重新登录');
            // 清除token并跳转到登录页
            localStorage.removeItem('auth_token');
            window.location.href = '/login';
            break;
          case 403:
            console.error('权限不足');
            break;
          case 404:
            console.error('请求的资源不存在:', `${error.config?.baseURL}${error.config?.url}`);
            break;
          case 429:
            console.error('请求过于频繁，请稍后再试');
            break;
          case 500:
            console.error('服务器内部错误:', data?.message || '服务器内部错误');
            break;
          default:
            console.error(`请求失败 (${status}):`, data?.message || '未知错误');
        }
      } else if (error.request) {
        console.error('网络连接失败，请检查网络设置');
      } else {
        console.error('请求配置错误:', error.message);
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
  get: <T = any>(url: string, params?: any): Promise<T> => {
    return api.get<ApiResponse<T>>(url, { params }).then(res => {
      console.log(`[API] GET ${url} 响应:`, res.data);
      if (!res.data || !res.data.success) {
        console.error(`[API] GET ${url} 失败:`, res.data?.message || '未知错误');
        throw new Error(res.data?.message || '请求失败');
      }
      if (res.data.data === undefined || res.data.data === null) {
        console.warn(`[API] GET ${url} 返回的data字段为空`);
        return null as T;
      }
      return res.data.data;
    }).catch((error: any) => {
      // 确保错误对象包含状态码信息
      if (error.response) {
        const enhancedError = new Error(error.response.data?.message || error.message);
        (enhancedError as any).status = error.response.status;
        (enhancedError as any).response = error.response;
        throw enhancedError;
      }
      throw error;
    });
  },
    
  post: <T = any>(url: string, data?: any): Promise<T> => {
    return api.post<ApiResponse<T>>(url, data).then(res => {
      if (!res.data || !res.data.success) {
        throw new Error(res.data?.message || '请求失败');
      }
      return res.data.data as T;
    });
  },
    
  put: <T = any>(url: string, data?: any): Promise<T> => {
    return api.put<ApiResponse<T>>(url, data).then(res => {
      if (!res.data || !res.data.success) {
        throw new Error(res.data?.message || '请求失败');
      }
      return res.data.data as T;
    });
  },
    
  delete: <T = any>(url: string): Promise<T> => {
    return api.delete<ApiResponse<T>>(url).then(res => {
      if (!res.data || !res.data.success) {
        throw new Error(res.data?.message || '请求失败');
      }
      return res.data.data as T;
    });
  },
    
  patch: <T = any>(url: string, data?: any): Promise<T> => {
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
): Promise<any> => {
  const formData = new FormData();
  formData.append('file', file);
  
  return api.post(url, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    onUploadProgress: (progressEvent) => {
      if (onProgress && progressEvent.total) {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress(progress);
      }
    },
  }).then(res => res.data.data);
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