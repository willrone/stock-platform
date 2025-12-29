/**
 * 全局应用状态管理
 * 
 * 使用Zustand管理应用的全局状态，包括：
 * - 用户信息
 * - 系统配置
 * - 全局加载状态
 * - 错误信息
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  avatar?: string;
}

interface SystemConfig {
  apiBaseUrl: string;
  wsUrl: string;
  theme: 'light' | 'dark';
  language: 'zh-CN' | 'en-US';
}

interface AppState {
  // 用户状态
  user: User | null;
  isAuthenticated: boolean;
  
  // 系统配置
  config: SystemConfig;
  
  // 全局状态
  loading: boolean;
  error: string | null;
  
  // Actions
  setUser: (user: User | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  updateConfig: (config: Partial<SystemConfig>) => void;
  clearError: () => void;
}

export const useAppStore = create<AppState>()(
  devtools(
    (set, get) => ({
      // 初始状态
      user: null,
      isAuthenticated: false,
      config: {
        apiBaseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1',
        wsUrl: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws',
        theme: 'light',
        language: 'zh-CN',
      },
      loading: false,
      error: null,

      // Actions
      setUser: (user) => set((state) => ({
        user,
        isAuthenticated: !!user,
      }), false, 'setUser'),

      setLoading: (loading) => set({ loading }, false, 'setLoading'),

      setError: (error) => set({ error }, false, 'setError'),

      updateConfig: (newConfig) => set((state) => ({
        config: { ...state.config, ...newConfig }
      }), false, 'updateConfig'),

      clearError: () => set({ error: null }, false, 'clearError'),
    }),
    {
      name: 'app-store',
    }
  )
);