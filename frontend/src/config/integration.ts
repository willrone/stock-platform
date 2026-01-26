/**
 * 前后端集成配置
 *
 * 统一管理前后端集成相关的配置，包括：
 * - API端点配置
 * - WebSocket配置
 * - 错误处理配置
 * - 重试策略配置
 */

// API配置
export const API_CONFIG = {
  // 基础URL - 使用相对路径，通过Next.js代理转发
  BASE_URL: '/api/v1',

  // WebSocket URL - WebSocket不能通过HTTP代理，需要直接连接后端
  // 从环境变量读取，如果没有则从当前页面hostname推断
  WS_URL: (() => {
    if (typeof window === 'undefined') {
      // 服务端渲染时使用环境变量
      return process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';
    }
    // 客户端：使用环境变量或从当前页面推断
    const envUrl = process.env.NEXT_PUBLIC_WS_URL;
    if (envUrl) {
      return envUrl;
    }

    // 从当前页面的hostname和协议推断WebSocket地址
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const hostname = window.location.hostname;
    // 从环境变量获取后端端口，或使用默认8000
    const port = process.env.NEXT_PUBLIC_BACKEND_PORT || '8000';
    return `${protocol}//${hostname}:${port}/ws`;
  })(),

  // 请求超时时间（毫秒）
  TIMEOUT: 30000,

  // 重试配置
  RETRY: {
    MAX_ATTEMPTS: 3,
    DELAY: 1000,
    BACKOFF_FACTOR: 2,
  },

  // 分页配置
  PAGINATION: {
    DEFAULT_PAGE_SIZE: 20,
    MAX_PAGE_SIZE: 100,
  },
};

// WebSocket配置
export const WS_CONFIG = {
  // 重连配置
  RECONNECT: {
    MAX_ATTEMPTS: 5,
    INITIAL_DELAY: 1000,
    MAX_DELAY: 30000,
    BACKOFF_FACTOR: 2,
  },

  // 心跳配置
  HEARTBEAT: {
    INTERVAL: 30000, // 30秒
    TIMEOUT: 5000, // 5秒
  },

  // 消息队列配置
  MESSAGE_QUEUE: {
    MAX_SIZE: 100,
    FLUSH_INTERVAL: 1000,
  },
};

// 错误处理配置
export const ERROR_CONFIG = {
  // 错误类型映射
  ERROR_TYPES: {
    NETWORK_ERROR: 'network_error',
    TIMEOUT_ERROR: 'timeout_error',
    VALIDATION_ERROR: 'validation_error',
    AUTHENTICATION_ERROR: 'auth_error',
    AUTHORIZATION_ERROR: 'permission_error',
    SERVER_ERROR: 'server_error',
    UNKNOWN_ERROR: 'unknown_error',
  },

  // 错误消息映射
  ERROR_MESSAGES: {
    network_error: '网络连接失败，请检查网络设置',
    timeout_error: '请求超时，请稍后重试',
    validation_error: '输入数据格式错误',
    auth_error: '身份验证失败，请重新登录',
    permission_error: '权限不足，无法执行此操作',
    server_error: '服务器内部错误，请稍后重试',
    unknown_error: '未知错误，请联系技术支持',
  },

  // 自动重试的错误类型
  RETRYABLE_ERRORS: ['network_error', 'timeout_error', 'server_error'],

  // 需要用户干预的错误类型
  USER_ACTION_REQUIRED: ['auth_error', 'permission_error', 'validation_error'],
};

// 缓存配置
export const CACHE_CONFIG = {
  // 缓存键前缀
  KEY_PREFIX: 'stock_prediction_',

  // 缓存过期时间（毫秒）
  TTL: {
    STOCK_DATA: 5 * 60 * 1000, // 5分钟
    TASK_LIST: 30 * 1000, // 30秒
    TASK_DETAIL: 10 * 1000, // 10秒
    MODEL_LIST: 10 * 60 * 1000, // 10分钟
    SYSTEM_STATUS: 60 * 1000, // 1分钟
  },

  // 最大缓存大小
  MAX_SIZE: 100,
};

// 通知配置
export const NOTIFICATION_CONFIG = {
  // 通知类型
  TYPES: {
    SUCCESS: 'success',
    INFO: 'info',
    WARNING: 'warning',
    ERROR: 'error',
  },

  // 默认显示时间（毫秒）
  DURATION: {
    SUCCESS: 3000,
    INFO: 4000,
    WARNING: 5000,
    ERROR: 0, // 不自动关闭
  },

  // 最大通知数量
  MAX_NOTIFICATIONS: 5,

  // 位置配置
  POSITION: 'top-right' as const,
};

// 实时更新配置
export const REALTIME_CONFIG = {
  // 任务状态轮询间隔（毫秒）
  TASK_POLLING_INTERVAL: 2000,

  // 系统状态轮询间隔（毫秒）
  SYSTEM_STATUS_INTERVAL: 10000,

  // 数据刷新间隔（毫秒）
  DATA_REFRESH_INTERVAL: 30000,

  // 自动刷新开关
  AUTO_REFRESH: {
    TASK_LIST: true,
    TASK_DETAIL: true,
    SYSTEM_STATUS: true,
    DATA_STATUS: false, // 手动刷新
  },
};

// 性能配置
export const PERFORMANCE_CONFIG = {
  // 虚拟滚动配置
  VIRTUAL_SCROLL: {
    ITEM_HEIGHT: 60,
    BUFFER_SIZE: 10,
    THRESHOLD: 100, // 超过100项启用虚拟滚动
  },

  // 防抖延迟（毫秒）
  DEBOUNCE_DELAY: {
    SEARCH: 300,
    RESIZE: 100,
    SCROLL: 50,
  },

  // 节流间隔（毫秒）
  THROTTLE_INTERVAL: {
    API_CALL: 100,
    UI_UPDATE: 16, // 60fps
  },

  // 懒加载配置
  LAZY_LOAD: {
    ROOT_MARGIN: '50px',
    THRESHOLD: 0.1,
  },
};

// 开发配置
export const DEV_CONFIG = {
  // 是否启用调试模式
  DEBUG: process.env.NODE_ENV === 'development',

  // 是否启用性能监控
  PERFORMANCE_MONITORING: process.env.NODE_ENV === 'development',

  // 是否启用详细日志
  VERBOSE_LOGGING: process.env.NODE_ENV === 'development',

  // 模拟延迟（毫秒）
  MOCK_DELAY: 500,

  // 是否使用模拟数据
  USE_MOCK_DATA: false,
};

// 导出所有配置
export const INTEGRATION_CONFIG = {
  API: API_CONFIG,
  WS: WS_CONFIG,
  ERROR: ERROR_CONFIG,
  CACHE: CACHE_CONFIG,
  NOTIFICATION: NOTIFICATION_CONFIG,
  REALTIME: REALTIME_CONFIG,
  PERFORMANCE: PERFORMANCE_CONFIG,
  DEV: DEV_CONFIG,
} as const;

// 类型定义
export type IntegrationConfig = typeof INTEGRATION_CONFIG;
export type ApiConfig = typeof API_CONFIG;
export type WebSocketConfig = typeof WS_CONFIG;
export type ErrorConfig = typeof ERROR_CONFIG;
