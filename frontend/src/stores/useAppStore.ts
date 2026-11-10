'use client';

/**
 * 全局应用状态管理
 *
 * 使用Zustand管理应用的全局状态，包括：
 * - 用户信息（持久化，刷新后不丢失）
 * - 系统配置 / 主题语言偏好（持久化）
 * - 全局加载状态与错误（不持久化）
 *
 * 持久化说明（2026-10-27）：
 * - 使用 zustand/middleware 的 persist 中间件，key = 'stock-platform-app-store'
 * - 只持久化 user + 主题/语言偏好；loading / error 等运行时状态不落盘
 * - 带 version + migrate：旧版本或结构异常的 localStorage 数据不会导致启动崩溃
 * - 恢复时优先用 localStorage.user_info（后端登录态原样）映射出 store 的 user，
 *   保证 is_admin / subscription_tier 等后端字段在刷新后不丢
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import type { PersistStorage, StateStorage, StorageValue } from 'zustand/middleware';
import { DEFAULT_BACKEND_WS_URL } from '@/config/backend';

export interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  avatar?: string;
  /** 2026-10-25：管理端入口（充值订单确认到账）按此字段显示 */
  is_admin?: boolean;
  subscription_tier?: string;
}

/**
 * 允许写入 store 的用户形态：
 * - 登录页写入的是规范化后的 User（name / role）
 * - 注册页写入的是后端原始用户（username / subscription_tier）
 * 两种形态都要能正确归一化。
 */
export type UserInput = Partial<User> & {
  username?: string;
};

export interface SystemConfig {
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

  // 全局状态（不持久化）
  loading: boolean;
  error: string | null;

  // Actions
  setUser: (user: UserInput | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  updateConfig: (config: Partial<SystemConfig>) => void;
  clearError: () => void;
}

/** 实际落盘的字段：用户 + 主题/语言偏好 */
interface PersistedAppState {
  user: User | null;
  theme: SystemConfig['theme'];
  language: SystemConfig['language'];
}

export const APP_STORE_KEY = 'stock-platform-app-store';
export const APP_STORE_VERSION = 2;

/** 后端登录态在 localStorage 中的 key（由登录/注册页写入） */
export const USER_INFO_STORAGE_KEY = 'user_info';

/**
 * 把「后端原始用户」或「已规范化用户」统一映射成 store 的 User 结构。
 * 后端字段：id / email / username / subscription_tier / is_admin
 */
export function normalizeUser(input: UserInput | null | undefined): User | null {
  if (!input) {
    return null;
  }

  const email = input.email ?? '';
  const name = input.name ?? input.username ?? email;
  const id = input.id ?? '';

  // 三者全空说明没有有效用户
  if (!id && !email && !name) {
    return null;
  }

  const normalized: User = {
    id,
    name,
    email,
    role: input.role ?? input.subscription_tier ?? 'free',
    is_admin: Boolean(input.is_admin),
  };

  if (input.avatar) {
    normalized.avatar = input.avatar;
  }
  if (input.subscription_tier) {
    normalized.subscription_tier = input.subscription_tier;
  }

  return normalized;
}

/** 读取 localStorage.user_info（后端登录态原样），映射为 store 的 user */
export function readStoredUser(): User | null {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    const raw = window.localStorage.getItem(USER_INFO_STORAGE_KEY);
    if (!raw) {
      return null;
    }
    return normalizeUser(JSON.parse(raw) as UserInput);
  } catch {
    // 脏数据不应阻断启动
    return null;
  }
}

/**
 * 自定义 storage：
 * - SSR 环境下完全惰性（不访问 window）
 * - JSON 解析失败时返回 null，而不是抛错中断启动
 * - 明确声明为**同步**实现（StateStorage 的 getItem 类型允许返回 Promise，
 *   但这里的实现是同步的；标成同步才能在 getItem 里直接用 string 做 JSON.parse）
 */
const safeStateStorage = {
  getItem: (name: string): string | null => {
    if (typeof window === 'undefined') {
      return null;
    }
    try {
      return window.localStorage.getItem(name);
    } catch {
      return null;
    }
  },
  setItem: (name: string, value: string): void => {
    if (typeof window === 'undefined') {
      return;
    }
    try {
      window.localStorage.setItem(name, value);
    } catch {
      // 超出配额等场景静默失败
    }
  },
  removeItem: (name: string): void => {
    if (typeof window === 'undefined') {
      return;
    }
    try {
      window.localStorage.removeItem(name);
    } catch {
      // ignore
    }
  },
} satisfies StateStorage;

const appStorage: PersistStorage<PersistedAppState> = {
  getItem: (name: string): StorageValue<PersistedAppState> | null => {
    const raw = safeStateStorage.getItem(name);
    if (!raw) {
      return null;
    }
    try {
      const parsed = JSON.parse(raw) as StorageValue<PersistedAppState>;
      if (!parsed || typeof parsed !== 'object' || !('state' in parsed)) {
        return null;
      }
      return parsed;
    } catch {
      return null;
    }
  },
  setItem: (name: string, value: StorageValue<PersistedAppState>) => {
    safeStateStorage.setItem(name, JSON.stringify(value));
  },
  removeItem: (name: string) => {
    safeStateStorage.removeItem(name);
  },
};

const defaultWsUrl = (): string => {
  if (typeof window === 'undefined') {
    return process.env.NEXT_PUBLIC_WS_URL || DEFAULT_BACKEND_WS_URL;
  }
  const envUrl = process.env.NEXT_PUBLIC_WS_URL;
  if (envUrl) {
    return envUrl;
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const hostname = window.location.hostname;
  const port = process.env.NEXT_PUBLIC_BACKEND_PORT || '8000';
  return `${protocol}//${hostname}:${port}/ws`;
};

export const useAppStore = create<AppState>()(
  devtools(
    persist(
      set => ({
        // 初始状态
        user: null,
        isAuthenticated: false,
        config: {
          apiBaseUrl: '/api/v1', // 使用相对路径，通过Next.js代理转发
          wsUrl: defaultWsUrl(),
          theme: 'light',
          language: 'zh-CN',
        },
        loading: false,
        error: null,

        // Actions
        setUser: user => {
          const normalized = normalizeUser(user);
          set(
            () => ({
              user: normalized,
              isAuthenticated: Boolean(normalized),
            }),
            false,
            'setUser'
          );
        },

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
        name: APP_STORE_KEY,
        version: APP_STORE_VERSION,
        storage: appStorage,
        // 只落盘 user + 主题/语言偏好
        partialize: (state): PersistedAppState => ({
          user: state.user,
          theme: state.config.theme,
          language: state.config.language,
        }),
        merge: (persistedState, currentState): AppState => {
          const persisted = (persistedState ?? {}) as Partial<PersistedAppState>;
          const user = normalizeUser(persisted.user);

          return {
            ...currentState,
            user,
            isAuthenticated: Boolean(user),
            config: {
              ...currentState.config,
              theme: persisted.theme === 'dark' ? 'dark' : 'light',
              language: persisted.language === 'en-US' ? 'en-US' : 'zh-CN',
            },
          };
        },
        /**
         * 版本迁移：旧结构（无 version / 结构不符 / user 字段异常）统一降级为安全结构，
         * 绝不抛错，保证老 localStorage 不会让应用白屏。
         *
         * 类型说明（zustand 4.x 的类型缺陷）：`PersistOptions.migrate` 的签名被硬编码为
         * `=> S | Promise<S>`（S = 完整 store state），但配合 `partialize` 时 migrate
         * 实际只应返回**落盘子集**（PersistedAppState）。运行时 zustand 会把 migrate 的
         * 结果交给 merge 拼回完整 state，所以这里返回落盘形态是正确的；
         * 仅类型层面需要一次断言。
         */
        migrate: persistedState => {
          const raw = (persistedState ?? {}) as Record<string, unknown>;
          const rawUser = raw.user as UserInput | undefined | null;

          const migrated: PersistedAppState = {
            user: normalizeUser(rawUser),
            theme: raw.theme === 'dark' ? 'dark' : 'light',
            language: raw.language === 'en-US' ? 'en-US' : 'zh-CN',
          };

          return migrated as unknown as AppState;
        },
        /**
         * 恢复完成后的兜底：用 localStorage.user_info（后端登录态）补齐 user，
         * 这样 is_admin / subscription_tier 在刷新后依然可见。
         */
        onRehydrateStorage: () => state => {
          if (!state) {
            return;
          }
          const stored = readStoredUser();
          if (stored) {
            state.setUser(stored);
          }
        },
      }
    ),
    {
      name: 'app-store',
    }
  )
);
