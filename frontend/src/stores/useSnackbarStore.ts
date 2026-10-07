/**
 * 全局 Snackbar 状态管理
 *
 * 提供统一的 Toast/Snackbar 通知机制，用于：
 * - API 调用错误提示
 * - 操作成功反馈
 * - 系统通知
 */

import { create } from 'zustand';

export type SnackbarSeverity = 'success' | 'error' | 'warning' | 'info';

interface SnackbarMessage {
  id: string;
  message: string;
  severity: SnackbarSeverity;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface SnackbarState {
  messages: SnackbarMessage[];
  addMessage: (message: Omit<SnackbarMessage, 'id'>) => string;
  removeMessage: (id: string) => void;
  clearMessages: () => void;
  showSuccess: (message: string, duration?: number) => void;
  showError: (message: string, duration?: number) => void;
  showWarning: (message: string, duration?: number) => void;
  showInfo: (message: string, duration?: number) => void;
}

let messageCounter = 0;

const generateId = () => `snackbar-${++messageCounter}-${Date.now()}`;

export const useSnackbarStore = create<SnackbarState>((set, get) => ({
  messages: [],

  addMessage: (message) => {
    const id = generateId();
    set((state) => ({
      messages: [...state.messages, { ...message, id }],
    }));
    return id;
  },

  removeMessage: (id) => {
    set((state) => ({
      messages: state.messages.filter((m) => m.id !== id),
    }));
  },

  clearMessages: () => {
    set({ messages: [] });
  },

  showSuccess: (message, duration = 4000) => {
    get().addMessage({ message, severity: 'success', duration });
  },

  showError: (message, duration = 6000) => {
    get().addMessage({ message, severity: 'error', duration });
  },

  showWarning: (message, duration = 5000) => {
    get().addMessage({ message, severity: 'warning', duration });
  },

  showInfo: (message, duration = 4000) => {
    get().addMessage({ message, severity: 'info', duration });
  },
}));
