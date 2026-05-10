#!/usr/bin/env node

const http = require('http');
const WebSocket = require('ws');

const baseUrl = process.env.SMOKE_BASE_URL || 'http://127.0.0.1:13000';
const cdpBaseUrl = process.env.SMOKE_CDP_URL || 'http://127.0.0.1:18800';
const delayMs = Number(process.env.SMOKE_DELAY_MS || '2500');
const paths = [
  '/',
  '/dashboard',
  '/tasks',
  '/tasks/create',
  '/backtest',
  '/predictions',
  '/models',
  '/data',
  '/monitoring',
  '/optimization',
  '/signals',
  '/settings',
];

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function httpJson(url, options = {}) {
  return new Promise((resolve, reject) => {
    const req = http.request(url, { method: options.method || 'GET' }, res => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', chunk => {
        body += chunk;
      });
      res.on('end', () => {
        if ((res.statusCode || 0) >= 400) {
          reject(new Error(`${options.method || 'GET'} ${url} failed: ${res.statusCode} ${body}`));
          return;
        }
        try {
          resolve(JSON.parse(body));
        } catch (error) {
          reject(error);
        }
      });
    });
    req.setTimeout(10000, () => req.destroy(new Error(`Timeout: ${url}`)));
    req.on('error', reject);
    req.end();
  });
}

function cdpCall(ws, method, params = {}) {
  cdpCall.nextId = (cdpCall.nextId || 0) + 1;
  const id = cdpCall.nextId;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      ws.off('message', onMessage);
      reject(new Error(`CDP timeout: ${method}`));
    }, 30000);
    const onMessage = raw => {
      const msg = JSON.parse(raw.toString());
      if (msg.id !== id) return;
      clearTimeout(timer);
      ws.off('message', onMessage);
      if (msg.error) reject(new Error(`${method}: ${msg.error.message}`));
      else resolve(msg.result || {});
    };
    ws.on('message', onMessage);
  });
}

function waitForEvent(ws, predicate, timeoutMs = 30000) {
  return new Promise(resolve => {
    const timer = setTimeout(() => {
      ws.off('message', onMessage);
      resolve(null);
    }, timeoutMs);
    const onMessage = raw => {
      const msg = JSON.parse(raw.toString());
      if (!predicate(msg)) return;
      clearTimeout(timer);
      ws.off('message', onMessage);
      resolve(msg);
    };
    ws.on('message', onMessage);
  });
}

function isFatalBrowserLog(text) {
  return [
    'ReactServerComponentsError',
    'Unhandled Runtime Error',
    'Application error',
    'Cannot find module',
    'Failed to compile',
    'Minified React error',
    'Hydration failed',
  ].some(marker => text.includes(marker));
}

async function inspectPath(path) {
  const target = await httpJson(`${cdpBaseUrl}/json/new?${encodeURIComponent(`${baseUrl}${path}`)}`, {
    method: 'PUT',
  });
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.once('open', resolve);
    ws.once('error', reject);
  });

  const errors = [];
  ws.on('message', raw => {
    const msg = JSON.parse(raw.toString());
    if (msg.method === 'Runtime.exceptionThrown') {
      const detail = msg.params.exceptionDetails;
      const description = detail.exception?.description || detail.exception?.value || detail.text || 'unknown';
      const location = `${detail.url || ''}:${detail.lineNumber ?? ''}:${detail.columnNumber ?? ''}`;
      errors.push(`exception: ${description} ${location}`.trim());
    }
    if (msg.method === 'Log.entryAdded') {
      const entry = msg.params.entry;
      if (entry.level === 'error' && isFatalBrowserLog(entry.text || '')) {
        errors.push(`log: ${entry.text}`);
      }
    }
    if (msg.method === 'Console.messageAdded') {
      const message = msg.params.message;
      if (message.level === 'error' && isFatalBrowserLog(message.text || '')) {
        errors.push(`console: ${message.text}`);
      }
    }
  });

  await cdpCall(ws, 'Runtime.enable');
  await cdpCall(ws, 'Log.enable');
  await cdpCall(ws, 'Page.enable');
  await waitForEvent(ws, msg => msg.method === 'Page.loadEventFired', 45000);
  await cdpCall(ws, 'Runtime.evaluate', {
    expression: `new Promise(resolve => setTimeout(resolve, ${delayMs}))`,
    awaitPromise: true,
  });
  const evalResult = await cdpCall(ws, 'Runtime.evaluate', {
    expression: `({
      title: document.title,
      bodyText: document.body ? document.body.innerText.slice(0, 800) : '',
      hasNextError: Boolean(document.querySelector('nextjs-portal')) || document.body.innerText.includes('Unhandled Runtime Error') || document.body.innerText.includes('Application error'),
      url: location.href
    })`,
    returnByValue: true,
  });
  const value = evalResult.result?.value || {};
  if (value.hasNextError) {
    errors.push('page contains Next.js/runtime error overlay');
  }
  if (value.title !== '股票预测平台') {
    errors.push(`unexpected title: ${value.title}`);
  }

  await httpJson(`${cdpBaseUrl}/json/close/${target.id}`).catch(() => null);
  ws.close();
  return { path, errors, title: value.title, bodyText: value.bodyText || '' };
}

(async () => {
  const failures = [];
  for (const path of paths) {
    try {
      await sleep(delayMs);
      const result = await inspectPath(path);
      if (result.errors.length > 0) {
        console.log(`FAIL ${path}: ${result.errors.join('; ')}`);
        failures.push(result);
      } else {
        console.log(`OK ${path}`);
      }
    } catch (error) {
      console.log(`FAIL ${path}: ${error.message}`);
      failures.push({ path, errors: [error.message] });
    }
  }

  if (failures.length > 0) {
    console.error('Browser smoke failed:', JSON.stringify(failures, null, 2));
    process.exit(1);
  }
})();
