/**
 * WebSocket服务
 *
 * 处理实时数据通信，包括：
 * - 任务状态实时更新
 * - 系统状态监控
 * - 实时数据推送
 * - 连接管理和重连
 */

// WebSocket事件类型
type TaskCreatedData = { task_id: string; task_name: string };
type TaskProgressData = { task_id: string; progress: number; status: string };
type TaskCompletedData = { task_id: string; results: unknown };
type TaskFailedData = { task_id: string; error: string };
type SystemStatusData = Record<string, unknown>;
type SystemAlertData = { level: 'info' | 'warning' | 'error'; message: string };
type DataUpdatedData = { stock_code: string; timestamp: string };
type PredictionResultData = { prediction_id: string; results: unknown };

export interface WebSocketEvents {
  // 任务相关事件
  'task:created': (data: TaskCreatedData) => void;
  'task:progress': (data: TaskProgressData) => void;
  'task:completed': (data: TaskCompletedData) => void;
  'task:failed': (data: TaskFailedData) => void;

  // 系统状态事件
  'system:status': (data: SystemStatusData) => void;
  'system:alert': (data: SystemAlertData) => void;

  // 数据更新事件
  'data:updated': (data: DataUpdatedData) => void;
  'prediction:result': (data: PredictionResultData) => void;
}

type ServerMessage = Record<string, unknown> & { type?: string };
type GenericEventHandler = (data: unknown) => void;

const wsLogger = {
  debug: (...args: unknown[]) => {
    if (process.env.NODE_ENV !== 'production') {
      globalThis.console.log(...args);
    }
  },
  info: (...args: unknown[]) => {
    if (process.env.NODE_ENV !== 'production') {
      globalThis.console.info(...args);
    }
  },
  warn: (...args: unknown[]) => {
    globalThis.console.warn(...args);
  },
  error: (...args: unknown[]) => {
    globalThis.console.error(...args);
  },
};

// WebSocket管理类
export class WebSocketService {
  private socket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private eventHandlers = new Map<keyof WebSocketEvents, Set<GenericEventHandler>>();

  constructor() {
    this.connect();
  }

  /**
   * 建立WebSocket连接
   * WebSocket不能通过HTTP代理，需要直接连接到后端服务器
   */
  private connect(): void {
    let wsUrl: string;

    // 优先使用环境变量配置
    if (process.env.NEXT_PUBLIC_WS_URL) {
      // 确保 URL 以 /ws 结尾（环境变量可能只配了 host:port）
      const base = process.env.NEXT_PUBLIC_WS_URL.replace(/\/ws\/?$/, '');
      wsUrl = `${base}/ws`;
    } else if (typeof window !== 'undefined') {
      // 客户端：根据当前页面地址推断后端WebSocket地址
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const hostname = window.location.hostname;
      // 从环境变量获取后端端口，或使用默认8000
      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || '8000';
      wsUrl = `${protocol}//${hostname}:${port}/ws`;
    } else {
      // 服务端：使用默认值
      wsUrl = 'ws://localhost:8000/ws';
    }

    wsLogger.debug('[WebSocket] 连接到:', wsUrl);

    try {
      this.socket = new WebSocket(wsUrl);
      this.setupEventListeners();
    } catch (error) {
      wsLogger.error('[WebSocket] 连接创建失败:', error);
      this.handleReconnect();
    }
  }

  /**
   * 设置事件监听器
   */
  private setupEventListeners(): void {
    if (!this.socket) {
      return;
    }

    // 连接成功
    this.socket.onopen = () => {
      wsLogger.debug('[WebSocket] 连接成功');
      this.reconnectAttempts = 0;
      wsLogger.info('实时连接已建立');
    };

    // 连接断开
    this.socket.onclose = event => {
      wsLogger.debug('[WebSocket] 连接断开:', event.code, event.reason);
      wsLogger.info('实时连接已断开');

      // 自动重连（除非是正常关闭）
      if (event.code !== 1000) {
        this.handleReconnect();
      }
    };

    // 连接错误
    this.socket.onerror = error => {
      wsLogger.error('[WebSocket] 连接错误:', error);
      this.handleReconnect();
    };

    // 接收消息
    this.socket.onmessage = event => {
      try {
        const data = JSON.parse(event.data) as ServerMessage;
        this.handleMessage(data);
      } catch (error) {
        wsLogger.error('[WebSocket] 消息解析失败:', error);
      }
    };
  }

  /**
   * 处理接收到的消息
   */
  private handleMessage(data: ServerMessage): void {
    const { type } = data;
    if (typeof type !== 'string') {
      wsLogger.warn('[WebSocket] 消息缺少有效 type:', data);
      return;
    }

    switch (type) {
      case 'connection':
        wsLogger.debug('[WebSocket] 连接确认:', data.message);
        break;

      case 'task:created':
        wsLogger.debug('[WebSocket] 任务创建:', data);
        this.emit('task:created', data as TaskCreatedData);
        break;

      case 'task:progress':
        wsLogger.debug('[WebSocket] 任务进度:', data);
        this.emit('task:progress', data as TaskProgressData);
        break;

      case 'task:completed':
        wsLogger.debug('[WebSocket] 任务完成:', data);
        this.emit('task:completed', data as TaskCompletedData);
        wsLogger.info(`任务 ${String(data.task_id)} 已完成`);
        break;

      case 'task:failed':
        wsLogger.debug('[WebSocket] 任务失败:', data);
        this.emit('task:failed', data as TaskFailedData);
        wsLogger.error(`任务 ${String(data.task_id)} 执行失败: ${String(data.error)}`);
        break;

      case 'system:status':
        this.emit('system:status', data as SystemStatusData);
        break;

      case 'system:alert':
        this.emit('system:alert', data as SystemAlertData);

        // 显示系统警告
        switch ((data as SystemAlertData).level) {
          case 'info':
            wsLogger.info((data as SystemAlertData).message);
            break;
          case 'warning':
            wsLogger.warn((data as SystemAlertData).message);
            break;
          case 'error':
            wsLogger.error((data as SystemAlertData).message);
            break;
        }
        break;

      case 'data:updated':
        this.emit('data:updated', data as DataUpdatedData);
        break;

      case 'prediction:result':
        this.emit('prediction:result', data as PredictionResultData);
        break;

      case 'subscription':
      case 'unsubscription':
        wsLogger.debug('[WebSocket] 订阅状态:', data.message);
        break;

      case 'pong':
        // 心跳响应
        break;

      case 'error':
        wsLogger.error('[WebSocket] 服务器错误:', data.message);
        break;

      default:
        wsLogger.warn('[WebSocket] 未知消息类型:', type, data);
    }
  }

  /**
   * 处理重连逻辑
   */
  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      wsLogger.error('[WebSocket] 重连次数已达上限');
      wsLogger.error('无法建立实时连接，请刷新页面重试');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    wsLogger.debug(`[WebSocket] ${delay}ms 后尝试第 ${this.reconnectAttempts} 次重连`);

    setTimeout(() => {
      this.connect();
    }, delay);
  }

  /**
   * 订阅事件
   */
  public on<K extends keyof WebSocketEvents>(event: K, handler: WebSocketEvents[K]): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.add(handler as GenericEventHandler);
      return;
    }
    this.eventHandlers.set(event, new Set<GenericEventHandler>([handler as GenericEventHandler]));
  }

  /**
   * 取消订阅事件
   */
  public off<K extends keyof WebSocketEvents>(event: K, handler: WebSocketEvents[K]): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.delete(handler as GenericEventHandler);
      if (handlers.size === 0) {
        this.eventHandlers.delete(event);
      }
    }
  }

  /**
   * 触发事件
   */
  private emit<K extends keyof WebSocketEvents>(
    event: K,
    data: Parameters<WebSocketEvents[K]>[0]
  ): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          wsLogger.error(`[WebSocket] 事件处理器错误 (${event}):`, error);
        }
      });
    }
  }

  /**
   * 发送消息到服务器
   */
  public send(event: string, data?: Record<string, unknown>): void {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      const message = {
        type: event,
        ...data,
      };
      this.socket.send(JSON.stringify(message));
    } else {
      wsLogger.warn('[WebSocket] 连接未建立，无法发送消息');
    }
  }

  /**
   * 订阅任务更新
   */
  public subscribeToTask(taskId: string): void {
    this.send('subscribe:task', { task_id: taskId });
  }

  /**
   * 取消订阅任务更新
   */
  public unsubscribeFromTask(taskId: string): void {
    this.send('unsubscribe:task', { task_id: taskId });
  }

  /**
   * 订阅系统状态
   */
  public subscribeToSystemStatus(): void {
    this.send('subscribe:system');
  }

  /**
   * 取消订阅系统状态
   */
  public unsubscribeFromSystemStatus(): void {
    this.send('unsubscribe:system');
  }

  /**
   * 检查连接状态
   */
  public isConnected(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  /**
   * 手动重连
   */
  public reconnect(): void {
    if (this.socket) {
      this.socket.close();
    }
    this.reconnectAttempts = 0;
    this.connect();
  }

  /**
   * 断开连接
   */
  public disconnect(): void {
    if (this.socket) {
      this.socket.close(1000, 'Client disconnect');
      this.socket = null;
    }
    this.eventHandlers.clear();
  }
}

// 创建全局WebSocket实例
export const wsService = new WebSocketService();
