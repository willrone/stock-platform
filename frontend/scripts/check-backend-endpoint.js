#!/usr/bin/env node
/**
 * 前端启动守卫 —— 防止「连上了一个源码已删的孤儿后端实例」
 *
 * ## 它拦的是什么
 *
 * 2026-11-03 实测事故：某个会话手工起前端时，shell 里带着
 * `NEXT_PUBLIC_API_URL=http://127.0.0.1:8890`（那个端口上跑的是一个
 * **源码已删、进程还活着、而且免鉴权**的旧后端实例）。
 * Next.js 的 dotenv **不覆盖已存在的 process.env** → `.env.local` 里正确的
 * `http://127.0.0.1:18082` 被无视 → 前端全程连着那个孤儿实例，
 * 读到的数据全是旧结构，而我以为自己在验证新代码。
 *
 * `scripts/dev-frontend.sh` 走 `source .env.local` 能覆盖外部 env，所以**走脚本是安全的**；
 * 危险的是绕过脚本直接 `npm run dev` —— 这正是本守卫要覆盖的场景。
 *
 * ## 为什么不能靠 health 响应区分身份
 *
 * 孤儿实例**就是本项目的旧代码**，它的 `/api/v1/health` 返回和现在**一模一样**
 * （`{"status":"healthy","version":"1.0.0"}`）→ 响应内容无法用来分辨新旧。
 * 唯一可靠的判据是**端口是否等于仓库约定值**。
 *
 * ## 判定规则
 *
 * | 情况 | 结果 |
 * |---|---|
 * | `process.env` 没设 `NEXT_PUBLIC_API_URL` | 通过（Next 会自己加载 `.env.local`） |
 * | 端口与 `.env.local` 一致 | 通过 |
 * | 端口与 `.env.local` 不一致 | **退出码 1**，打印诊断 |
 * | `.env.local` 不存在且 `process.env` 也没设 | 退出码 1（提示跑 setup） |
 *
 * 退出码 1 是刻意的：这类漂移**静默通过比直接报错危险得多** ——
 * 前端会正常启动、页面正常渲染、数据却来自一个不该在的进程。
 */

const fs = require('fs');
const path = require('path');

const FRONTEND_DIR = path.resolve(__dirname, '..');
const ENV_FILE = path.join(FRONTEND_DIR, '.env.local');

/** 仓库约定的后端端口（与 backend/scripts/env.sh、src/config/backend.ts 一致） */
const CANONICAL_PORT = '18082';

function parseEnvFile(file) {
  const out = {};
  if (!fs.existsSync(file)) return out;
  for (const raw of fs.readFileSync(file, 'utf8').split('\n')) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const eq = line.indexOf('=');
    if (eq === -1) continue;
    const key = line.slice(0, eq).trim();
    let val = line.slice(eq + 1).trim();
    if (
      (val.startsWith('"') && val.endsWith('"')) ||
      (val.startsWith("'") && val.endsWith("'"))
    ) {
      val = val.slice(1, -1);
    }
    out[key] = val;
  }
  return out;
}

function portOf(url) {
  if (!url) return null;
  const m = String(url).match(/^(?:https?|wss?):\/\/(?:[^:/@]+@)?[^:/@]*:(\d+)/);
  return m ? m[1] : null;
}

function fail(msg) {
  process.stderr.write(
    '\n' +
      '╔══════════════════════════════════════════════════════════════╗\n' +
      '║  前端启动守卫：后端地址与仓库约定不一致，已中止启动            ║\n' +
      '╚══════════════════════════════════════════════════════════════╝\n' +
      msg +
      '\n\n' +
      '  修复方式（任选其一）：\n' +
      '    1) 清掉外部遗留的环境变量，让 Next 自己读 .env.local：\n' +
      '         unset NEXT_PUBLIC_API_URL NEXT_PUBLIC_WS_URL\n' +
      '    2) 用仓库脚本启动（它会 source .env.local 覆盖外部值）：\n' +
      '         ./scripts/dev-frontend.sh\n' +
      '    3) 确实要用别的端口，就同时改三处保持一致：\n' +
      '         frontend/.env.local\n' +
      '         frontend/src/config/backend.ts (DEFAULT_BACKEND_PORT)\n' +
      '         scripts/env.sh (DEFAULT_DEV_BACKEND_PORT)\n' +
      '\n'
  );
  process.exit(1);
}

function main() {
  const fileEnv = parseEnvFile(ENV_FILE);
  const fileApi = fileEnv.NEXT_PUBLIC_API_URL;
  const filePort = portOf(fileApi);

  const procApi = process.env.NEXT_PUBLIC_API_URL;
  const procPort = portOf(procApi);

  // WebSocket 地址独立检查 —— 它不经过 Next 代理，直连后端，
  // 漂移了会静默连到孤儿实例（前端页面照样正常，实时数据却是旧的）。
  const fileWs = fileEnv.NEXT_PUBLIC_WS_URL;
  const fileWsPort = portOf(fileWs);
  const procWs = process.env.NEXT_PUBLIC_WS_URL;
  const procWsPort = portOf(procWs);

  if (procWs && fileWsPort && procWsPort !== fileWsPort) {
    fail(
      `  环境变量 NEXT_PUBLIC_WS_URL = ${procWs}   (端口 ${procWsPort})\n` +
        `  frontend/.env.local      = ${fileWs}   (端口 ${fileWsPort})\n\n` +
        '  WebSocket 不走 Next 代理，是浏览器直连后端的 ——\n' +
        '  地址漂移时前端页面照样正常渲染，实时数据却来自别的进程。\n'
    );
  }

  if (!procApi) {
    // 没设外部值 → Next 会自己加载 .env.local → 无漂移风险
    if (!fileApi) {
      fail('  原因：frontend/.env.local 不存在，且环境变量 NEXT_PUBLIC_API_URL 也未设置。\n');
    }
    return; // 通过
  }

  if (!filePort) {
    fail(
      `  原因：环境变量 NEXT_PUBLIC_API_URL=${procApi} 已设置，\n` +
        '        但 frontend/.env.local 里没有可解析的端口，无法确认是否一致。\n'
    );
  }

  if (procPort !== filePort) {
    fail(
      `  环境变量 NEXT_PUBLIC_API_URL = ${procApi}   (端口 ${procPort})\n` +
        `  frontend/.env.local      = ${fileApi}   (端口 ${filePort})\n\n` +
        '  Next.js 的 dotenv 不会覆盖已存在的 process.env，\n' +
        '  所以你手工设的地址会赢过 .env.local —— 前端会连到它。\n' +
        `  仓库约定的后端端口是 ${CANONICAL_PORT}。\n`
    );
  }

  // 端口一致 → 通过
}

main();
