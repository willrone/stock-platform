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
export interface WebSocketEvents {
  // 任务相关事件
  'task:created': (data: { task_id: string; task_name: string }) => void;
  'task:progress': (data: { task_id: string; progress: number; status: string }) => void;
  'task:completed': (data: { task_id: string; results: any }) => void;
  'task:failed': (data: { task_id: string; error: string }) => void;

  // 系统状态事件
  'system:status': (data: any) => void;
  'system:alert': (data: { level: 'info' | 'warning' | 'error'; message: string }) => void;

  // 数据更新事件
  'data:updated': (data: { stock_code: string; timestamp: string }) => void;
  'prediction:result': (data: { prediction_id: string; results: any }) => void;
}

// WebSocket管理类
export class WebSocketService {
  private socket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private eventHandlers: Map<string, Function[]> = new Map();

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
      wsUrl = process.env.NEXT_PUBLIC_WS_URL;
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

    console.log('[WebSocket] 连接到:', wsUrl);

    try {
      this.socket = new WebSocket(wsUrl);
      this.setupEventListeners();
    } catch (error) {
      console.error('[WebSocket] 连接创建失败:', error);
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
      console.log('[WebSocket] 连接成功');
      this.reconnectAttempts = 0;
      console.log('实时连接已建立');
    };

    // 连接断开
    this.socket.onclose = event => {
      console.log('[WebSocket] 连接断开:', event.code, event.reason);
      console.log('实时连接已断开');

      // 自动重连（除非是正常关闭）
      if (event.code !== 1000) {
        this.handleReconnect();
      }
    };

    // 连接错误
    this.socket.onerror = error => {
      console.error('[WebSocket] 连接错误:', error);
      this.handleReconnect();
    };

    // 接收消息
    this.socket.onmessage = event => {
      try {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      } catch (error) {
        console.error('[WebSocket] 消息解析失败:', error);
      }
    };
  }

  /**
   * 处理接收到的消息
   */
  private handleMessage(data: any): void {
    const { type } = data;

    switch (type) {
      case 'connection':
        console.log('[WebSocket] 连接确认:', data.message);
        break;

      case 'task:created':
        console.log('[WebSocket] 任务创建:', data);
        this.emit('task:created', data);
        break;

      case 'task:progress':
        console.log('[WebSocket] 任务进度:', data);
        this.emit('task:progress', data);
        break;

      case 'task:completed':
        console.log('[WebSocket] 任务完成:', data);
        this.emit('task:completed', data);
        console.log(`任务 ${data.task_id} 已完成`);
        break;

      case 'task:failed':
        console.log('[WebSocket] 任务失败:', data);
        this.emit('task:failed', data);
        console.error(`任务 ${data.task_id} 执行失败: ${data.error}`);
        break;

      case 'system:status':
        this.emit('system:status', data);
        break;

      case 'system:alert':
        this.emit('system:alert', data);

        // 显示系统警告
        switch (data.level) {
          case 'info':
            console.info(data.message);
            break;
          case 'warning':
            console.warn(data.message);
            break;
          case 'error':
            console.error(data.message);
            break;
        }
        break;

      case 'data:updated':
        this.emit('data:updated', data);
        break;

      case 'prediction:result':
        this.emit('prediction:result', data);
        break;

      case 'subscription':
      case 'unsubscription':
        console.log('[WebSocket] 订阅状态:', data.message);
        break;

      case 'pong':
        // 心跳响应
        break;

      case 'error':
        console.error('[WebSocket] 服务器错误:', data.message);
        break;

      default:
        console.warn('[WebSocket] 未知消息类型:', type, data);
    }
  }

  /**
   * 处理重连逻辑
   */
  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WebSocket] 重连次数已达上限');
      console.error('无法建立实时连接，请刷新页面重试');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    console.log(`[WebSocket] ${delay}ms 后尝试第 ${this.reconnectAttempts} 次重连`);

    setTimeout(() => {
      this.connect();
    }, delay);
  }

  /**
   * 订阅事件
   */
  public on<K extends keyof WebSocketEvents>(event: K, handler: WebSocketEvents[K]): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event)!.push(handler);
  }

  /**
   * 取消订阅事件
   */
  public off<K extends keyof WebSocketEvents>(event: K, handler: WebSocketEvents[K]): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      const index = handlers.indexOf(handler);
      if (index > -1) {
        handlers.splice(index, 1);
      }
    }
  }

  /**
   * 触发事件
   */
  private emit(event: string, data: any): void {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach(handler => {
        try {
          handler(data);
        } catch (error) {
          console.error(`[WebSocket] 事件处理器错误 (${event}):`, error);
        }
      });
    }
  }

  /**
   * 发送消息到服务器
   */
  public send(event: string, data?: any): void {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      const message = {
        type: event,
        ...data,
      };
      this.socket.send(JSON.stringify(message));
    } else {
      console.warn('[WebSocket] 连接未建立，无法发送消息');
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
