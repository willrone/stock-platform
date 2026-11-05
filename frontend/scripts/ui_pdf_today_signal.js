#!/usr/bin/env node
/*
 * 「今日信号」UI 目测脚本（本机 Chrome 的 Page.captureScreenshot 会超时，
 * GPU 进程 exit 15，连 CLI --screenshot 和 about:blank 都一样 —— 实测踩过。
 * 所以走 printToPDF：PDF 保留背景色/字色，再用 macOS 自带 sips 转 PNG 看。）
 *
 * 流程：/login 建立同源上下文并写入后端签发的 access_token
 *      → /today-signal → 选策略 → 读 DOM 事实 → 打印 PDF
 * mode=card 时只保留「可下单清单」卡片，方便逐行看排版。
 *
 * 用法: node scripts/ui_pdf_today_signal.js <out.pdf> [策略下拉索引] [card|full]
 */
const fs = require('fs');
const http = require('http');
const WebSocket = require('ws');

const CDP = process.env.SMOKE_CDP_URL || 'http://127.0.0.1:18800';
const BASE = 'http://127.0.0.1:13000';
const TOKEN_FILE = '/Users/rone/.hermes/cache/scratch/ui_token.txt';
const OUT = process.argv[2] || '/Users/rone/.hermes/cache/scratch/ui_shot.pdf';
const SEL_INDEX = Number(process.argv[3] || -1);
const MODE = process.argv[4] === 'full' ? 'full' : 'card';
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

function httpJson(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let b = '';
      res.setEncoding('utf8');
      res.on('data', (c) => (b += c));
      res.on('end', () => resolve(JSON.parse(b)));
    }).on('error', reject);
  });
}

const PROBE = (onlyCard) => `(() => {
  const c = document.querySelector('[data-testid="orderable-list-card"]');
  if (!c) return JSON.stringify({ card: false });
  if (${onlyCard}) {
    // 从卡片往上走，每层隐藏不相关兄弟节点，只留这一支
    let node = c;
    while (node && node !== document.body) {
      const parent = node.parentElement;
      if (!parent) break;
      Array.from(parent.children).forEach((sib) => {
        if (sib !== node) sib.style.display = 'none';
      });
      node = parent;
    }
    document.querySelectorAll('header,nav,aside,.MuiDrawer-root,.MuiDrawer-paper').forEach((e) => {
      if (!e.contains(c)) e.style.display = 'none';
    });
    c.style.maxWidth = 'none';
    c.style.boxShadow = 'none';
    c.style.border = '1px solid #D0D5DD';
    window.scrollTo(0, 0);
  }
  // PDF 默认不打印背景色，必须显式 exact，否则红色行 tint 会消失
  const st = document.createElement('style');
  st.textContent = '*{ -webkit-print-color-adjust:exact !important; print-color-adjust:exact !important }';
  document.head.appendChild(st);
  const trs = Array.from(c.querySelectorAll('tbody tr'));
  const styleOf = (t) => {
    const cs = getComputedStyle(t);
    return cs.backgroundColor + '|' + cs.color;
  };
  const isBlocked = (t) => /买不进|卖不出|无行情|未收盘/.test(t.lastElementChild.innerText);
  const blocked = trs.filter(isBlocked);
  const ok = trs.filter((t) => !isBlocked(t));
  const rowH = trs.length ? Math.round(trs[0].getBoundingClientRect().height) : 0;
  return JSON.stringify({
    card: true,
    rows: trs.length,
    blocked: blocked.length,
    rowH: rowH,
    chips: Array.from(c.querySelectorAll('.MuiChip-label')).map((x) => x.innerText).join(' | '),
    header_cols: Array.from(c.querySelectorAll('thead th')).map((t) => t.innerText).join(' | '),
    blocked_style: blocked.length ? styleOf(blocked[0]) : '',
    ok_style: ok.length ? styleOf(ok[0]) : '',
    blocked_reason: blocked.length ? blocked[0].lastElementChild.innerText : ''
  });
})()`;

(async () => {
  const token = fs.readFileSync(TOKEN_FILE, 'utf8').trim();
  // 只认本项目页面的 target（之前抓到过 chrome://settings/help，后面全打空）
  const targets = (await httpJson(CDP + '/json/list')).filter((t) => t.type === 'page');
  let page = targets.find((t) => (t.url || '').startsWith(BASE));
  if (!page) {
    page = await new Promise((res2, rej2) => {
      const r2 = http.request(CDP + '/json/new?about:blank', { method: 'PUT' }, (res) => {
        let b = '';
        res.setEncoding('utf8');
        res.on('data', (c) => (b += c));
        res.on('end', () => res2(JSON.parse(b)));
      });
      r2.on('error', rej2);
      r2.end();
    });
    await sleep(1200);
  }
  const ws = new WebSocket(page.webSocketDebuggerUrl, { perMessageDeflate: false });
  let id = 0;
  const pending = new Map();
  const send = (method, params = {}, timeoutMs = 30000) =>
    new Promise((resolve, reject) => {
      const mid = ++id;
      const timer = setTimeout(() => {
        pending.delete(mid);
        reject(new Error(method + ' 超时 ' + timeoutMs + 'ms'));
      }, timeoutMs);
      pending.set(mid, {
        resolve: (v) => {
          clearTimeout(timer);
          resolve(v);
        },
        reject: (e) => {
          clearTimeout(timer);
          reject(e);
        },
      });
      ws.send(JSON.stringify({ id: mid, method, params }));
    });
  await new Promise((res, rej) => {
    ws.on('open', res);
    ws.on('error', rej);
  });
  ws.on('message', (raw) => {
    const msg = JSON.parse(raw.toString());
    if (msg.id && pending.has(msg.id)) {
      const p = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) p.reject(new Error(msg.error.message));
      else p.resolve(msg.result);
    }
  });

  const evalJs = async (expr) => {
    const r = await send('Runtime.evaluate', { expression: expr, returnByValue: true });
    if (r.exceptionDetails) throw new Error('JS: ' + JSON.stringify(r.exceptionDetails).slice(0, 200));
    return r.result.value;
  };

  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
  await send('Page.navigate', { url: BASE + '/login' });
  await sleep(2500);
  await evalJs('localStorage.setItem("access_token", ' + JSON.stringify(token) + '); 1');
  await send('Page.navigate', { url: BASE + '/today-signal' });
  await sleep(9000);
  if (SEL_INDEX >= 0) {
    await evalJs('(()=>{const e=document.querySelector(".MuiSelect-select");'
      + '["mousedown","mouseup"].forEach(t=>e.dispatchEvent(new MouseEvent(t,{bubbles:true})));})()');
    await sleep(1200);
    await evalJs('(()=>{const li=Array.from(document.querySelectorAll(\'li[role="option"]\'))[' + SEL_INDEX + '];'
      + '["mousedown","mouseup","click"].forEach(t=>li.dispatchEvent(new MouseEvent(t,{bubbles:true})));})()');
    await sleep(9000);
  }

  const info = JSON.parse(await evalJs(PROBE(MODE === 'card')));
  log('DOM:', JSON.stringify(info));
  await sleep(800);

  const pdf = await send('Page.printToPDF', {
    printBackground: true,
    preferCSSPageSize: false,
    paperWidth: 13.3,
    paperHeight: MODE === 'card' ? 11.5 : 20,
    marginTop: 0.25,
    marginBottom: 0.25,
    marginLeft: 0.25,
    marginRight: 0.25,
    scale: 0.9,
  }, 45000);
  fs.writeFileSync(OUT, Buffer.from(pdf.data, 'base64'));
  log('saved:', OUT, fs.statSync(OUT).size, 'bytes');
  ws.close();
  process.exit(0);
})().catch((e) => {
  console.error('FAIL:', e.message);
  process.exit(1);
});
