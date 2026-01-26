/**
 * 训练进度WebSocket客户端
 *
 * 管理与后端的WebSocket连接，用于接收实时训练进度更新
 */

export interface TrainingProgressData {
  type: string;
  model_id: string;
  progress: number;
  stage: string;
  message?: string;
  metrics?: Record<string, any>;
  timestamp: string;
}

export type ProgressCallback = (data: TrainingProgressData) => void;

export class TrainingProgressWebSocket {
  private ws: WebSocket | null = null;
  private subscribers: Map<string, ProgressCallback[]> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000; // 1秒
  private isConnecting = false;

  constructor(private wsUrl?: string) {
    // 确定WebSocket URL
    if (wsUrl) {
      this.wsUrl = wsUrl;
    } else if (process.env.NEXT_PUBLIC_WS_URL) {
      this.wsUrl = process.env.NEXT_PUBLIC_WS_URL;
    } else if (typeof window !== 'undefined') {
      // 客户端：根据当前页面地址推断后端WebSocket地址
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const hostname = window.location.hostname;
      const port = process.env.NEXT_PUBLIC_BACKEND_PORT || '8000';
      this.wsUrl = `${protocol}//${hostname}:${port}/ws`;
    } else {
      // 服务端：使用默认值
      this.wsUrl = 'ws://localhost:8000/ws';
    }
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

      try {
        this.ws = new WebSocket(this.wsUrl!);

        this.ws.onopen = () => {
          console.log('训练进度WebSocket连接已建立');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          resolve();
        };

        this.ws.onmessage = event => {
          try {
            const data: TrainingProgressData = JSON.parse(event.data);
            this.handleMessage(data);
          } catch (error) {
            console.error('解析WebSocket消息失败:', error);
          }
        };

        this.ws.onclose = event => {
          console.log('训练进度WebSocket连接已关闭', event.code, event.reason);
          this.isConnecting = false;
          this.ws = null;

          // 自动重连
          if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = error => {
          console.error('训练进度WebSocket连接错误:', error);
          this.isConnecting = false;
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
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.subscribers.clear();
  }

  /**
   * 订阅特定模型的训练进度
   */
  subscribeToModel(modelId: string, callback: ProgressCallback): () => void {
    if (!this.subscribers.has(modelId)) {
      this.subscribers.set(modelId, []);
    }

    this.subscribers.get(modelId)!.push(callback);

    // 返回取消订阅函数
    return () => {
      const callbacks = this.subscribers.get(modelId);
      if (callbacks) {
        const index = callbacks.indexOf(callback);
        if (index > -1) {
          callbacks.splice(index, 1);
        }

        // 如果没有更多订阅者，删除该模型的订阅
        if (callbacks.length === 0) {
          this.subscribers.delete(modelId);
        }
      }
    };
  }

  /**
   * 订阅所有训练进度更新
   */
  subscribeToAll(callback: ProgressCallback): () => void {
    return this.subscribeToModel('*', callback);
  }

  /**
   * 获取连接状态
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * 获取当前订阅的模型数量
   */
  getSubscriptionCount(): number {
    return this.subscribers.size;
  }

  /**
   * 处理接收到的消息
   */
  private handleMessage(data: TrainingProgressData): void {
    // 通知特定模型的订阅者
    const modelCallbacks = this.subscribers.get(data.model_id);
    if (modelCallbacks) {
      modelCallbacks.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('执行训练进度回调失败:', error);
        }
      });
    }

    // 通知全局订阅者
    const globalCallbacks = this.subscribers.get('*');
    if (globalCallbacks) {
      globalCallbacks.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('执行全局训练进度回调失败:', error);
        }
      });
    }
  }

  /**
   * 安排重连
   */
  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1); // 指数退避

    console.log(`${delay}ms后尝试重连 (第${this.reconnectAttempts}次)`);

    setTimeout(() => {
      this.connect().catch(error => {
        console.error('重连失败:', error);
      });
    }, delay);
  }
}

// 全局单例实例
let globalWebSocketInstance: TrainingProgressWebSocket | null = null;

/**
 * 获取全局WebSocket实例
 */
export function getTrainingProgressWebSocket(): TrainingProgressWebSocket {
  if (!globalWebSocketInstance) {
    globalWebSocketInstance = new TrainingProgressWebSocket();
  }
  return globalWebSocketInstance;
}

/**
 * 清理全局WebSocket实例
 */
export function cleanupTrainingProgressWebSocket(): void {
  if (globalWebSocketInstance) {
    globalWebSocketInstance.disconnect();
    globalWebSocketInstance = null;
  }
}
