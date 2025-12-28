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
import { message } from 'antd';

// 标准响应格式
export interface ApiResponse<T = any> {
  success: boolean;
  message: string;
  data?: T;
  timestamp: string;
}

// 创建axios实例
const createApiInstance = (): AxiosInstance => {
  const instance = axios.create({
    baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1',
    timeout: 30000,
    headers: {
      'Content-Type': 'application/json',
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
      
      console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
      return config;
    },
    (error) => {
      console.error('[API] 请求错误:', error);
      return Promise.reject(error);
    }
  );

  // 响应拦截器
  instance.interceptors.response.use(
    (response: AxiosResponse<ApiResponse>) => {
      const { data } = response;
      
      // 检查业务逻辑错误
      if (!data.success) {
        console.error('[API] 业务错误:', data.message);
        message.error(data.message || '请求失败');
        return Promise.reject(new Error(data.message || '请求失败'));
      }
      
      return response;
    },
    (error: AxiosError) => {
      console.error('[API] 响应错误:', error);
      
      // 处理不同类型的错误
      if (error.response) {
        const status = error.response.status;
        const data = error.response.data as any;
        
        switch (status) {
          case 400:
            message.error(data?.message || '请求参数错误');
            break;
          case 401:
            message.error('未授权访问，请重新登录');
            // 清除token并跳转到登录页
            localStorage.removeItem('auth_token');
            window.location.href = '/login';
            break;
          case 403:
            message.error('权限不足');
            break;
          case 404:
            message.error('请求的资源不存在');
            break;
          case 429:
            message.error('请求过于频繁，请稍后再试');
            break;
          case 500:
            message.error(data?.message || '服务器内部错误');
            break;
          default:
            message.error(`请求失败 (${status})`);
        }
      } else if (error.request) {
        message.error('网络连接失败，请检查网络设置');
      } else {
        message.error('请求配置错误');
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
  get: <T = any>(url: string, params?: any): Promise<T> =>
    api.get<ApiResponse<T>>(url, { params }).then(res => res.data.data!),
    
  post: <T = any>(url: string, data?: any): Promise<T> =>
    api.post<ApiResponse<T>>(url, data).then(res => res.data.data!),
    
  put: <T = any>(url: string, data?: any): Promise<T> =>
    api.put<ApiResponse<T>>(url, data).then(res => res.data.data!),
    
  delete: <T = any>(url: string): Promise<T> =>
    api.delete<ApiResponse<T>>(url).then(res => res.data.data!),
    
  patch: <T = any>(url: string, data?: any): Promise<T> =>
    api.patch<ApiResponse<T>>(url, data).then(res => res.data.data!),
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