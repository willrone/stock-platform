/**
 * 后端地址默认值 —— 单一真源
 *
 * ## 为什么是 18082 而不是 8000
 *
 * 本仓库后端开发端口由 `scripts/env.sh` 约定为 **18082**（8000 是历史遗留，
 * 早已没有服务在监听）。此前以下 6 处各自硬编码 `'8000'` 作为兜底：
 *
 * | 文件 | 位置 |
 * |---|---|
 * | `src/utils/backendConfig.ts` | WebSocket / HTTP 兜底 |
 * | `src/config/integration.ts` | `WS_URL` 兜底 |
 * | `src/stores/useAppStore.ts` | WebSocket 兜底 |
 * | `src/services/websocket.ts` | `wsUrl` 兜底 |
 * | `src/services/BacktestProgressWebSocket.ts` | `baseUrl` 兜底 |
 * | `src/services/TrainingProgressWebSocket.ts` | `wsUrl` 兜底 |
 * | `src/app/api/health/route.ts` | `backend_host` 兜底 |
 *
 * 任何**没有加载 `.env.local`** 的场景（`next build`、CI、容器、他人机器）
 * 都会落到这些兜底 → 指向一个没人监听的端口 → 连接失败。
 * 本文件把这些兜底收敛到一处，改端口只改这里。
 *
 * ## 优先级
 *
 * 环境变量 > 本文件默认值。生产部署必须显式设置
 * `NEXT_PUBLIC_BACKEND_HOST` / `NEXT_PUBLIC_WS_URL` / `NEXT_PUBLIC_BACKEND_PORT`。
 */

/** 默认后端主机 */
export const DEFAULT_BACKEND_HOST = '127.0.0.1';

/**
 * 默认后端端口 —— 必须与仓库根 `scripts/env.sh` 的 `BACKEND_PORT` 一致。
 * 改这里之前先改 env.sh，否则两边会再次漂移。
 */
export const DEFAULT_BACKEND_PORT = '18082';

/** 默认后端 HTTP 基址 */
export const DEFAULT_BACKEND_HTTP_URL = `http://${DEFAULT_BACKEND_HOST}:${DEFAULT_BACKEND_PORT}`;

/** 默认后端 WebSocket 基址 */
export const DEFAULT_BACKEND_WS_URL = `ws://${DEFAULT_BACKEND_HOST}:${DEFAULT_BACKEND_PORT}/ws`;
