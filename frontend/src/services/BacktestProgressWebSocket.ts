/**
 * 回测进度WebSocket客户端
 *
 * 管理与后端的WebSocket连接，用于接收实时回测进度更新
 */

export interface BacktestProgressStage {
  name: string;
  description: string;
  progress: number;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  start_time?: string;
  end_time?: string;
  details: Record<string, any>;
}

export interface BacktestProgressData {
  type: string;
  task_id: string;
  backtest_id: string;
  overall_progress: number;
  current_stage: string;
  processed_days: number;
  total_days: number;
  current_date?: string;
  processing_speed: number;
  estimated_completion?: string;
  elapsed_time?: string;
  portfolio_value: number;
  signals_generated: number;
  trades_executed: number;
  warnings_count: number;
  error_message?: string;
  stages: BacktestProgressStage[];
  timestamp: string;
}

export interface BacktestErrorData {
  type: 'backtest_error';
  task_id: string;
  error_message: string;
  timestamp: string;
}

export interface BacktestCompletionData {
  type: 'backtest_completed';
  task_id: string;
  results: Record<string, any>;
  timestamp: string;
}

export interface BacktestCancellationData {
  type: 'backtest_cancelled';
  task_id: string;
  reason: string;
  timestamp: string;
}

export type BacktestWebSocketMessage =
  | BacktestProgressData
  | BacktestErrorData
  | BacktestCompletionData
  | BacktestCancellationData
  | { type: 'connection_established' | 'pong' | 'error' | 'no_progress_data'; [key: string]: any };

export type ProgressCallback = (data: BacktestProgressData) => void;
export type ErrorCallback = (error: BacktestErrorData) => void;
export type CompletionCallback = (completion: BacktestCompletionData) => void;
export type CancellationCallback = (cancellation: BacktestCancellationData) => void;
export type ConnectionCallback = (connected: boolean) => void;

export interface BacktestWebSocketCallbacks {
  onProgress?: ProgressCallback;
  onError?: ErrorCallback;
  onCompletion?: CompletionCallback;
  onCancellation?: CancellationCallback;
  onConnection?: ConnectionCallback;
}

export class BacktestProgressWebSocket {
  private ws: WebSocket | null = null;
  private taskId: string;
  private callbacks: BacktestWebSocketCallbacks = {};
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000; // 1秒
  private isConnecting = false;
  private isManuallyDisconnected = false;
  private heartbeatInterval: NodeJS.Timeout | null = null;

  constructor(
    taskId: string,
    private wsUrl?: string
  ) {
    this.taskId = taskId;

    // 确定WebSocket基础URL
    let baseUrl: string;

    if (wsUrl) {
      baseUrl = wsUrl;
    } else if (process.env.NEXT_PUBLIC_WS_URL) {
      baseUrl = process.env.NEXT_PUBLIC_WS_URL;
    } else if (typeof window !== 'undefined') {
      // 客户端：根据当前页面地址推断后端WebSocket地址
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const hostname = window.location.hostname;
      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || '8000';
      baseUrl = `${protocol}//${hostname}:${port}`;
    } else {
      // 服务端：使用默认值
      baseUrl = 'ws://localhost:8000';
    }

    // 清理URL末尾的斜杠，并确保不包含多余的/ws前缀
    baseUrl = baseUrl.replace(/\/+$/, ''); // 移除末尾斜杠
    // 如果URL末尾是/ws，移除它（避免重复）
    if (baseUrl.endsWith('/ws')) {
      baseUrl = baseUrl.slice(0, -3);
    }
    this.wsUrl = baseUrl;
  }

  /**
   * 设置回调函数
   */
  setCallbacks(callbacks: BacktestWebSocketCallbacks): void {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }

  /**
   * 连接WebSocket
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      if (this.isConnecting) {
        return;
      }

      this.isConnecting = true;
      this.isManuallyDisconnected = false;

      try {
        const wsEndpoint = `${this.wsUrl}/api/v1/backtest/ws/${this.taskId}`;
        this.ws = new WebSocket(wsEndpoint);

        this.ws.onopen = () => {
          console.log(`回测进度WebSocket连接已建立: ${this.taskId}`);
          this.isConnecting = false;
          this.reconnectAttempts = 0;

          // 启动心跳
          this.startHeartbeat();

          // 请求当前进度
          this.requestCurrentProgress();

          // 通知连接状态
          this.callbacks.onConnection?.(true);

          resolve();
        };

        this.ws.onmessage = event => {
          try {
            const data: BacktestWebSocketMessage = JSON.parse(event.data);
            this.handleMessage(data);
          } catch (error) {
            console.error('解析回测WebSocket消息失败:', error);
          }
        };

        this.ws.onclose = event => {
          console.log(`回测进度WebSocket连接已关闭: ${this.taskId}`, event.code, event.reason);
          this.isConnecting = false;
          this.ws = null;

          // 停止心跳
          this.stopHeartbeat();

          // 通知连接状态
          this.callbacks.onConnection?.(false);

          // 自动重连（如果不是手动断开）
          if (!this.isManuallyDisconnected && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = error => {
          console.error(`回测进度WebSocket连接错误: ${this.taskId}`, error);
          this.isConnecting = false;
          this.callbacks.onConnection?.(false);
          reject(error);
        };
      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  /**
   * 断开WebSocket连接
   */
  disconnect(): void {
    this.isManuallyDisconnected = true;
    this.stopHeartbeat();

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.callbacks.onConnection?.(false);
  }

  /**
   * 请求当前进度
   */
  requestCurrentProgress(): void {
    this.sendMessage({
      type: 'get_current_progress',
    });
  }

  /**
   * 取消回测
   */
  cancelBacktest(reason: string = '用户取消'): void {
    this.sendMessage({
      type: 'cancel_backtest',
      reason: reason,
    });
  }

  /**
   * 获取连接状态
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * 获取任务ID
   */
  getTaskId(): string {
    return this.taskId;
  }

  /**
   * 发送消息
   */
  private sendMessage(message: Record<string, any>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket未连接，无法发送消息:', message);
    }
  }

  /**
   * 处理接收到的消息
   */
  private handleMessage(data: BacktestWebSocketMessage): void {
    switch (data.type) {
      case 'connection_established':
        console.log('回测WebSocket连接建立确认:', data);
        break;

      case 'progress_update':
        this.callbacks.onProgress?.(data as BacktestProgressData);
        break;

      case 'backtest_error':
        this.callbacks.onError?.(data as BacktestErrorData);
        break;

      case 'backtest_completed':
        this.callbacks.onCompletion?.(data as BacktestCompletionData);
        break;

      case 'backtest_cancelled':
        this.callbacks.onCancellation?.(data as BacktestCancellationData);
        break;

      case 'pong':
        // 心跳响应，无需处理
        break;

      case 'no_progress_data':
        console.log('当前没有进度数据');
        break;

      case 'error':
        console.error('WebSocket错误消息:', data);
        break;

      default:
        console.log('未知的WebSocket消息类型:', data);
    }
  }

  /**
   * 启动心跳
   */
  private startHeartbeat(): void {
    this.stopHeartbeat();

    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.sendMessage({ type: 'ping' });
      }
    }, 30000); // 30秒心跳
  }

  /**
   * 停止心跳
   */
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  /**
   * 安排重连
   */
  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1); // 指数退避

    console.log(`${delay}ms后尝试重连回测WebSocket (第${this.reconnectAttempts}次)`);

    setTimeout(() => {
      if (!this.isManuallyDisconnected) {
        this.connect().catch(error => {
          console.error('回测WebSocket重连失败:', error);
        });
      }
    }, delay);
  }
}

/**
 * 回测进度WebSocket管理器
 * 管理多个任务的WebSocket连接
 */
export class BacktestProgressWebSocketManager {
  private connections: Map<string, BacktestProgressWebSocket> = new Map();

  /**
   * 创建或获取WebSocket连接
   */
  getConnection(taskId: string): BacktestProgressWebSocket {
    if (!this.connections.has(taskId)) {
      const connection = new BacktestProgressWebSocket(taskId);
      this.connections.set(taskId, connection);
    }
    return this.connections.get(taskId)!;
  }

  /**
   * 连接指定任务的WebSocket
   */
  async connect(
    taskId: string,
    callbacks?: BacktestWebSocketCallbacks
  ): Promise<BacktestProgressWebSocket> {
    const connection = this.getConnection(taskId);

    if (callbacks) {
      connection.setCallbacks(callbacks);
    }

    await connection.connect();
    return connection;
  }

  /**
   * 断开指定任务的WebSocket
   */
  disconnect(taskId: string): void {
    const connection = this.connections.get(taskId);
    if (connection) {
      connection.disconnect();
      this.connections.delete(taskId);
    }
  }

  /**
   * 断开所有WebSocket连接
   */
  disconnectAll(): void {
    this.connections.forEach((connection, taskId) => {
      connection.disconnect();
    });
    this.connections.clear();
  }

  /**
   * 获取活跃连接数
   */
  getActiveConnectionCount(): number {
    let count = 0;
    this.connections.forEach(conn => {
      if (conn.isConnected()) {
        count++;
      }
    });
    return count;
  }

  /**
   * 获取所有连接的任务ID
   */
  getConnectedTaskIds(): string[] {
    const connectedIds: string[] = [];
    this.connections.forEach((conn, taskId) => {
      if (conn.isConnected()) {
        connectedIds.push(taskId);
      }
    });
    return connectedIds;
  }
}

// 全局单例实例
let globalWebSocketManager: BacktestProgressWebSocketManager | null = null;

/**
 * 获取全局WebSocket管理器实例
 */
export function getBacktestProgressWebSocketManager(): BacktestProgressWebSocketManager {
  if (!globalWebSocketManager) {
    globalWebSocketManager = new BacktestProgressWebSocketManager();
  }
  return globalWebSocketManager;
}

/**
 * 清理全局WebSocket管理器实例
 */
export function cleanupBacktestProgressWebSocket(): void {
  if (globalWebSocketManager) {
    globalWebSocketManager.disconnectAll();
    globalWebSocketManager = null;
  }
}
