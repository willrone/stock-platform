/**
 * WebSocket服务
 * 
 * 处理实时数据通信，包括：
 * - 任务状态实时更新
 * - 系统状态监控
 * - 实时数据推送
 * - 连接管理和重连
 */

import { io, Socket } from 'socket.io-client';
import { message } from 'antd';

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
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private eventHandlers: Map<string, Function[]> = new Map();

  constructor() {
    this.connect();
  }

  /**
   * 建立WebSocket连接
   */
  private connect(): void {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
    
    this.socket = io(wsUrl, {
      transports: ['websocket'],
      timeout: 10000,
      forceNew: true,
    });

    this.setupEventListeners();
  }

  /**
   * 设置事件监听器
   */
  private setupEventListeners(): void {
    if (!this.socket) return;

    // 连接成功
    this.socket.on('connect', () => {
      console.log('[WebSocket] 连接成功');
      this.reconnectAttempts = 0;
      message.success('实时连接已建立');
    });

    // 连接断开
    this.socket.on('disconnect', (reason) => {
      console.log('[WebSocket] 连接断开:', reason);
      message.warning('实时连接已断开');
      
      // 自动重连
      if (reason === 'io server disconnect') {
        // 服务器主动断开，不重连
        return;
      }
      
      this.handleReconnect();
    });

    // 连接错误
    this.socket.on('connect_error', (error) => {
      console.error('[WebSocket] 连接错误:', error);
      this.handleReconnect();
    });

    // 任务相关事件
    this.socket.on('task:created', (data) => {
      console.log('[WebSocket] 任务创建:', data);
      this.emit('task:created', data);
    });

    this.socket.on('task:progress', (data) => {
      console.log('[WebSocket] 任务进度:', data);
      this.emit('task:progress', data);
    });

    this.socket.on('task:completed', (data) => {
      console.log('[WebSocket] 任务完成:', data);
      this.emit('task:completed', data);
      message.success(`任务 ${data.task_id} 已完成`);
    });

    this.socket.on('task:failed', (data) => {
      console.log('[WebSocket] 任务失败:', data);
      this.emit('task:failed', data);
      message.error(`任务 ${data.task_id} 执行失败: ${data.error}`);
    });

    // 系统状态事件
    this.socket.on('system:status', (data) => {
      this.emit('system:status', data);
    });

    this.socket.on('system:alert', (data) => {
      this.emit('system:alert', data);
      
      // 显示系统警告
      switch (data.level) {
        case 'info':
          message.info(data.message);
          break;
        case 'warning':
          message.warning(data.message);
          break;
        case 'error':
          message.error(data.message);
          break;
      }
    });

    // 数据更新事件
    this.socket.on('data:updated', (data) => {
      this.emit('data:updated', data);
    });

    this.socket.on('prediction:result', (data) => {
      this.emit('prediction:result', data);
    });
  }

  /**
   * 处理重连逻辑
   */
  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WebSocket] 重连次数已达上限');
      message.error('无法建立实时连接，请刷新页面重试');
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
  public on<K extends keyof WebSocketEvents>(
    event: K,
    handler: WebSocketEvents[K]
  ): void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event)!.push(handler);
  }

  /**
   * 取消订阅事件
   */
  public off<K extends keyof WebSocketEvents>(
    event: K,
    handler: WebSocketEvents[K]
  ): void {
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
    if (this.socket && this.socket.connected) {
      this.socket.emit(event, data);
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
    return this.socket?.connected || false;
  }

  /**
   * 手动重连
   */
  public reconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
    }
    this.reconnectAttempts = 0;
    this.connect();
  }

  /**
   * 断开连接
   */
  public disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.eventHandlers.clear();
  }
}

// 创建全局WebSocket实例
export const wsService = new WebSocketService();