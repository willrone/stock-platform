/**
 * 漏斗埋点客户端单测。
 *
 * 关注三件事：
 *   1. anon_id / session_id 的生成与持久化（注册前必须一直稳定，否则归并会断）
 *   2. trackEventOnce 的"每会话一次"语义（React StrictMode 会双跑 effect）
 *   3. trackEvent 永不抛错、永远带 auth 头（登录态）
 */

import { getAnonId, getSessionId, trackEvent, trackEventOnce } from '@/services/funnel';

describe('漏斗埋点客户端', () => {
  const originalFetch = global.fetch;
  let fetchMock: jest.Mock;

  beforeEach(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
    fetchMock = jest.fn().mockResolvedValue({ ok: true, json: async () => ({ ok: true }) });
    (global as unknown as { fetch: unknown }).fetch = fetchMock;
  });

  afterEach(() => {
    (global as unknown as { fetch: unknown }).fetch = originalFetch;
  });

  it('anon_id 生成一次后持久复用', () => {
    const first = getAnonId();
    expect(first).toBeTruthy();
    expect(getAnonId()).toBe(first);
    expect(window.localStorage.getItem('funnel_anon_id')).toBe(first);
  });

  it('session_id 与 anon_id 是两套标识', () => {
    expect(getSessionId()).not.toBe(getAnonId());
    expect(window.sessionStorage.getItem('funnel_session_id')).toBe(getSessionId());
  });

  it('trackEvent 发 POST /api/v1/funnel/track 且带事件名与标识', () => {
    trackEvent('view_signal', { source: 'today-signal' });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/v1/funnel/track');
    expect(init.method).toBe('POST');
    const body = JSON.parse(init.body);
    expect(body.event).toBe('view_signal');
    expect(body.anon_id).toBe(getAnonId());
    expect(body.session_id).toBe(getSessionId());
    expect(body.props).toEqual({ source: 'today-signal' });
  });

  it('登录后带上 Authorization 头', () => {
    window.localStorage.setItem('access_token', 'tok-123');
    trackEvent('view_pricing');
    const [, init] = fetchMock.mock.calls[0];
    expect(init.headers.Authorization).toBe('Bearer tok-123');
  });

  it('trackEventOnce 同会话只发一次', () => {
    trackEventOnce('first_screen');
    trackEventOnce('first_screen');
    trackEventOnce('first_screen');
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('trackEvent 失败不抛错（埋点不能拖垮业务）', () => {
    fetchMock.mockRejectedValueOnce(new Error('network down'));
    expect(() => trackEvent('click_upgrade', { plan: 'pro' })).not.toThrow();
  });
});
