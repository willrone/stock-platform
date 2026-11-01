/**
 * 漏斗埋点客户端。
 *
 * 事件注册表与后端 `app/services/funnel/funnel_service.py` 一一对应
 * —— **同一个事件名只能有一个来源**，否则报告会重复计数：
 *
 *   后端记录：signup / login / quota_exhausted
 *       服务端才知道的事实（注册成功、登录成功、配额打满），
 *       放后端才不会被前端报错、网络失败吃掉。
 *   前端记录（本文件）：first_screen / view_signal / view_pricing /
 *       click_upgrade / wallet_open
 *
 * 铁律：
 *   - 埋点**永远不能影响业务**。所有异常一律吞掉，调用方不需要 await。
 *   - 失败不重试、不排队、不缓存到内存 —— 丢一条埋点远好过拖慢首屏。
 *   - 同一事件在同一标签页会话内只上报一次（React StrictMode 会双跑 effect）。
 */

export type FunnelEventName =
  | 'first_screen'
  | 'view_signal'
  | 'view_pricing'
  | 'click_upgrade'
  | 'wallet_open';

const ANON_KEY = 'funnel_anon_id';
const SESSION_KEY = 'funnel_session_id';
const FIRED_PREFIX = 'funnel_fired:';

function randomId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `a${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
}

function safeGet(store: Storage | undefined, key: string): string {
  if (!store) return '';
  try {
    return store.getItem(key) ?? '';
  } catch {
    return '';
  }
}

function safeSet(store: Storage | undefined, key: string, value: string): void {
  if (!store) return;
  try {
    store.setItem(key, value);
  } catch {
    /* 隐私模式下 localStorage 可能不可写，忽略 */
  }
}

/** 匿名设备标识（注册前也用它记录行为，注册后由后端归并到 user_id） */
export function getAnonId(): string {
  if (typeof window === 'undefined') return '';
  let id = safeGet(window.localStorage, ANON_KEY);
  if (!id) {
    id = randomId();
    safeSet(window.localStorage, ANON_KEY, id);
  }
  return id;
}

/** 标签页会话标识 */
export function getSessionId(): string {
  if (typeof window === 'undefined') return '';
  let id = safeGet(window.sessionStorage, SESSION_KEY);
  if (!id) {
    id = randomId();
    safeSet(window.sessionStorage, SESSION_KEY, id);
  }
  return id;
}

/**
 * 上报一条埋点事件（fire-and-forget）。
 * @param event 事件名，取值见 FunnelEventName
 * @param props 附加属性（可选，序列化后 ≤2KB）
 */
export function trackEvent(
  event: FunnelEventName,
  props?: Record<string, unknown>
): void {
  if (typeof window === 'undefined') return;
  try {
    const token = safeGet(window.localStorage, 'access_token');
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (token) headers.Authorization = `Bearer ${token}`;

    void fetch('/api/v1/funnel/track', {
      method: 'POST',
      headers,
      body: JSON.stringify({
        event,
        anon_id: getAnonId(),
        session_id: getSessionId(),
        props,
      }),
      keepalive: true,
      credentials: 'same-origin',
    }).catch(() => undefined);
  } catch {
    /* 埋点失败不影响业务 */
  }
}

/**
 * 同一标签页会话内只上报一次（用于页面挂载类事件）。
 */
export function trackEventOnce(
  event: FunnelEventName,
  props?: Record<string, unknown>
): void {
  if (typeof window === 'undefined') return;
  const key = `${FIRED_PREFIX}${event}`;
  if (safeGet(window.sessionStorage, key)) return;
  safeSet(window.sessionStorage, key, '1');
  trackEvent(event, props);
}
