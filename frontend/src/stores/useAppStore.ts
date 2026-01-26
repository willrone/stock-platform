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
        apiBaseUrl: '/api/v1', // 使用相对路径，通过Next.js代理转发
        wsUrl: (() => {
          // WebSocket不能通过HTTP代理，需要直接连接后端
          if (typeof window === 'undefined') {
            return process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';
          }
          const envUrl = process.env.NEXT_PUBLIC_WS_URL;
          if (envUrl) {
            return envUrl;
          }
          const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
          const hostname = window.location.hostname;
          const port = process.env.NEXT_PUBLIC_BACKEND_PORT || '8000';
          return `${protocol}//${hostname}:${port}/ws`;
        })(),
        theme: 'light',
        language: 'zh-CN',
      },
      loading: false,
      error: null,

      // Actions
      setUser: user =>
        set(
          state => ({
            user,
            isAuthenticated: !!user,
          }),
          false,
          'setUser'
        ),

      setLoading: loading => set({ loading }, false, 'setLoading'),

      setError: error => set({ error }, false, 'setError'),

      updateConfig: newConfig =>
        set(
          state => ({
            config: { ...state.config, ...newConfig },
          }),
          false,
          'updateConfig'
        ),

      clearError: () => set({ error: null }, false, 'clearError'),
    }),
    {
      name: 'app-store',
    }
  )
);
