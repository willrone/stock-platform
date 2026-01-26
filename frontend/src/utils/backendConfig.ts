/**
 * 后端服务器配置工具
 *
 * 统一管理后端服务器地址配置，用于WebSocket等需要直接连接后端的场景
 */

/**
 * 获取后端WebSocket地址
 * WebSocket不能通过HTTP代理，需要直接连接到后端服务器
 */
export function getBackendWebSocketUrl(): string {
  // 优先使用环境变量
  if (process.env.NEXT_PUBLIC_WS_URL) {
    return process.env.NEXT_PUBLIC_WS_URL;
  }

  // 客户端：根据当前页面地址推断
  if (typeof window !== 'undefined') {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const hostname = window.location.hostname;
    const port = process.env.NEXT_PUBLIC_BACKEND_PORT || '8000';
    return `${protocol}//${hostname}:${port}/ws`;
  }

  // 服务端：使用默认值
  return 'ws://localhost:8000/ws';
}

/**
 * 获取后端HTTP地址（仅用于信息显示，不用于实际请求）
 * 实际API请求应使用相对路径 /api/v1，通过Next.js代理转发
 */
export function getBackendHttpUrl(): string {
  // 优先使用环境变量
  if (process.env.NEXT_PUBLIC_BACKEND_HOST) {
    const host = process.env.NEXT_PUBLIC_BACKEND_HOST;
    return host.startsWith('http') ? host : `http://${host}`;
  }

  // 客户端：根据当前页面地址推断
  if (typeof window !== 'undefined') {
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    const port = process.env.NEXT_PUBLIC_BACKEND_PORT || '8000';
    return `${protocol}//${hostname}:${port}`;
  }

  // 服务端：使用默认值
  return 'http://localhost:8000';
}
