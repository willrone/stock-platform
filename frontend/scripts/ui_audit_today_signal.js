#!/usr/bin/env node
/*
 * 「今日信号」可下单清单 —— 排版量化审计（不是目测，是量）：
 *   - 横向溢出：任何单元格 scrollWidth > clientWidth 即文字被截
 *   - 卡片是否超出可视容器宽度
 *   - 拦单行/正常行的文字对比度（WCAG 相对亮度法）
 *   - 行高、字号、chip 字号、档位 chip 是否还在、升级按钮是否已删
 * 用法: node scripts/ui_audit_today_signal.js [策略下拉索引]
 */
const fs = require('fs');
const http = require('http');
const WebSocket = require('ws');

const CDP = process.env.SMOKE_CDP_URL || 'http://127.0.0.1:18800';
const BASE = 'http://127.0.0.1:13000';
const TOKEN_FILE = '/Users/rone/.hermes/cache/scratch/ui_token.txt';
// 第二个参数：策略下拉里要选的那一项的文字（默认选有拦单的那条线，才能看到红行）
const SEL_TEXT = process.argv[2] || '';
const SEL_INDEX = Number(process.argv[3] || -1);
const SABOTAGE = process.argv.includes('--sabotage');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (...a) => console.log(new Date().toISOString().slice(11, 19), ...a);

function httpJson(url, method = 'GET') {
  return new Promise((resolve, reject) => {
    const req = http.request(url, { method }, (res) => {
      let b = '';
      res.setEncoding('utf8');
      res.on('data', (c) => (b += c));
      res.on('end', () => resolve(JSON.parse(b)));
    });
    req.on('error', reject);
    req.end();
  });
}

const AUDIT = `(() => {
  const c = document.querySelector('[data-testid="orderable-list-card"]');
  if (!c) return JSON.stringify({ card: false });
  const clamp = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  const lum = (rgb) => 0.2126 * clamp(rgb[0]) + 0.7152 * clamp(rgb[1]) + 0.0722 * clamp(rgb[2]);
  // 解析 rgb()/rgba()：不用正则字面量（在页面里被转义坑过一次），改成剥字符+切分
  const toRgba = (s) => {
    const parts = String(s)
      .replace(/[^0-9.,]/g, ' ')
      .split(/[, ]+/)
      .filter((x) => x.length)
      .map(Number);
    if (parts.length < 3) return [255, 255, 255, 1];
    return [parts[0], parts[1], parts[2], parts.length > 3 ? parts[3] : 1];
  };
  // rgba over 白底混合，才能算真实对比度
  const overWhite = (s) => {
    const p = toRgba(s);
    const a = p[3];
    return [
      Math.round(p[0] * a + 255 * (1 - a)),
      Math.round(p[1] * a + 255 * (1 - a)),
      Math.round(p[2] * a + 255 * (1 - a)),
    ];
  };
  const ratio = (fg, bg) => {
    const a = lum(fg) + 0.05;
    const b = lum(bg) + 0.05;
    return Number((Math.max(a, b) / Math.min(a, b)).toFixed(2));
  };
  const trs = Array.from(c.querySelectorAll('tbody tr'));
  const clipped = [];
  c.querySelectorAll('td,th').forEach((el) => {
    if (el.scrollWidth > el.clientWidth + 1) {
      clipped.push(el.innerText.replace(/\\n/g, '/').slice(0, 26) + ' (' + el.scrollWidth + '>' + el.clientWidth + ')');
    }
  });
  const blocked = trs.find((t) => /买不进|卖不出|无行情|未收盘/.test(t.lastElementChild.innerText));
  const ok = trs.find((t) => !/买不进|卖不出|无行情|未收盘/.test(t.lastElementChild.innerText));
  // 量文字本身的 color（td 自己的 color 会被 MUI 继承盖掉 —— 上一版量错了对象）
  const textEl = (cell) => {
    let el = cell;
    while (el && el.children.length) el = el.children[0];
    return el || cell;
  };
  const bRow = blocked ? getComputedStyle(blocked) : null;
  const bReason = blocked ? textEl(blocked.lastElementChild) : null;
  const bTintBg = blocked ? overWhite(bRow.backgroundColor) : null;
  const blockedOut = blocked
    ? {
        rowBg: bRow.backgroundColor,
        tintOverWhite: 'rgb(' + bTintBg.join(',') + ')',
        reasonText: blocked.lastElementChild.innerText.trim(),
        reasonColor: getComputedStyle(bReason).color,
        reasonFontPx: getComputedStyle(bReason).fontSize,
        contrastReasonOnTint: ratio(toRgba(getComputedStyle(bReason).color), bTintBg),
        sideColor: getComputedStyle(blocked.children[0]).color,
        contrastSideOnTint: ratio(toRgba(getComputedStyle(blocked.children[0]).color), bTintBg),
        codeColor: getComputedStyle(textEl(blocked.children[1])).color,
        contrastCodeOnTint: ratio(toRgba(getComputedStyle(textEl(blocked.children[1])).color), bTintBg),
      }
    : null;
  const okOut = ok
    ? {
        rowBg: getComputedStyle(ok).backgroundColor,
        sideColor: getComputedStyle(ok.children[0]).color,
        contrastSideOnWhite: ratio(toRgba(getComputedStyle(ok.children[0]).color), [255, 255, 255]),
        codeColor: getComputedStyle(textEl(ok.children[1])).color,
        contrastCodeOnWhite: ratio(toRgba(getComputedStyle(textEl(ok.children[1])).color), [255, 255, 255]),
        captionColor: getComputedStyle(textEl(ok.children[2])).color,
        contrastCaptionOnWhite: ratio(toRgba(getComputedStyle(textEl(ok.children[2])).color), [255, 255, 255]),
      }
    : null;
  const cardRect = c.getBoundingClientRect();
  const container = c.parentElement ? c.parentElement.getBoundingClientRect() : null;
  const chips = Array.from(c.querySelectorAll('.MuiChip-label')).map((x) => ({
    text: x.innerText,
    color: getComputedStyle(x).color,
    bg: getComputedStyle(x).backgroundColor,
    fontPx: getComputedStyle(x).fontSize,
  }));
  const tierChip = Array.from(document.querySelectorAll('.MuiChip-label'))
    .map((x) => x.innerText).filter((t) => /免费|专业|标准/.test(t));
  const upgradeBtn = Array.from(document.querySelectorAll('button,a'))
    .map((b) => (b.innerText || '').trim())
    .filter((t) => /升级|解锁|订阅/.test(t));
  const firstCell = trs.length ? getComputedStyle(trs[0].firstElementChild) : null;
  return JSON.stringify({
    card: true,
    rows: trs.length,
    cardWidth: Math.round(cardRect.width),
    containerWidth: container ? Math.round(container.width) : null,
    cardOverflowsContainer: container ? cardRect.width > container.width + 1 : null,
    horizontalScroll: c.querySelector('.MuiTableContainer-root')
      ? (() => { const e = c.querySelector('.MuiTableContainer-root'); return e.scrollWidth > e.clientWidth + 1; })()
      : null,
    clippedCells: clipped,
    rowHeightPx: trs.length ? Math.round(trs[0].getBoundingClientRect().height) : null,
    bodyFontPx: firstCell ? firstCell.fontSize : null,
    headerFontPx: (() => { const th = c.querySelector('thead th'); return th ? getComputedStyle(th).fontSize : null; })(),
    blocked: blockedOut,
    ok: okOut,
    chips,
    tierChipOnPage: tierChip,
    upgradeButtonsOnPage: upgradeBtn,
  });
})()`;

(async () => {
  const token = fs.readFileSync(TOKEN_FILE, 'utf8').trim();
  // 只认本项目页面的 target；之前这里抓到过 chrome://settings/help，导致后面全打空
  const targets = (await httpJson(CDP + '/json/list')).filter((t) => t.type === 'page');
  let page = targets.find((t) => (t.url || '').startsWith(BASE));
  if (!page) {
    page = await httpJson(CDP + '/json/new?about:blank', 'PUT');
    await sleep(1200);
  }
  if (!page || !page.webSocketDebuggerUrl) throw new Error('没有可用 page target');
  const ws = new WebSocket(page.webSocketDebuggerUrl, { perMessageDeflate: false });
  let id = 0;
  const pending = new Map();
  const send = (method, params = {}, timeoutMs = 25000) =>
    new Promise((resolve, reject) => {
      const mid = ++id;
      const timer = setTimeout(() => {
        pending.delete(mid);
        reject(new Error(method + ' 超时'));
      }, timeoutMs);
      pending.set(mid, { resolve: (v) => { clearTimeout(timer); resolve(v); }, reject: (e) => { clearTimeout(timer); reject(e); } });
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
    if (r.exceptionDetails) throw new Error('JS: ' + (r.exceptionDetails.exception?.description || r.exceptionDetails.text).slice(0, 300));
    return r.result.value;
  };
  // 页面是异步加载的，固定 sleep 会撞空节点 —— 一律轮询
  const waitFor = async (expr, timeoutMs = 30000, label = expr) => {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      try {
        const v = await evalJs(expr);
        if (v) return v;
      } catch (e) {
        /* 节点还没渲染，继续等 */
      }
      await sleep(700);
    }
    throw new Error('等不到: ' + label);
  };

  const key = async (k, vk) => {
    for (const type of ['keyDown', 'keyUp']) {
      await send('Input.dispatchKeyEvent', {
        type, key: k, code: k, windowsVirtualKeyCode: vk, nativeVirtualKeyCode: vk,
      });
    }
  };
  // MUI Select 的键盘导航：focus → Enter 打开 → ArrowDown ×n → Enter 选中
  const pickOption = async (needle) => {
    await evalJs('(()=>{const e=document.querySelector(".MuiSelect-select"); if(!e) return false; e.focus(); return document.activeElement===e;})()');
    await key('Enter', 13);
    await sleep(600);
    let n = Number(await evalJs('document.querySelectorAll(\'li[role="option"]\').length'));
    if (!n) {
      await key('Space', 32);
      await sleep(600);
      n = Number(await evalJs('document.querySelectorAll(\'li[role="option"]\').length'));
    }
    if (!n) throw new Error('下拉打不开（键盘也试过）');
    const texts = await evalJs('JSON.stringify(Array.from(document.querySelectorAll(\'li[role="option"]\')).map(x=>x.innerText.replace(/\\n/g," ")))');
    const idx = JSON.parse(texts).findIndex((t) => t.includes(needle));
    if (idx < 0) throw new Error('选项里没有: ' + needle + ' 现有: ' + texts.slice(0, 300));
    for (let i = 0; i < idx; i++) await key('ArrowDown', 40);
    await key('Enter', 13);
    await sleep(800);
    return { n, idx, chosen: await evalJs('document.querySelector(".MuiSelect-select").innerText.replace(/\\n/g," ")') };
  };

  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 900, deviceScaleFactor: 1, mobile: false });
  await send('Page.navigate', { url: BASE + '/login' });
  await sleep(2000);
  await evalJs('localStorage.setItem("access_token", ' + JSON.stringify(token) + '); 1');
  await send('Page.navigate', { url: BASE + '/today-signal' });
  await waitFor('!!localStorage.getItem("access_token") && !!document.querySelector(".MuiSelect-select")', 30000, '策略下拉框');
  if (SEL_TEXT) {
    const picked = await pickOption(SEL_TEXT);
    log('选中:', JSON.stringify(picked));
  }
  await waitFor('!!document.querySelector(\'[data-testid="orderable-list-card"] tbody tr\')', 30000, '清单表格行');
  await sleep(1500);
  if (SABOTAGE) {
    // 故意把方向列改回旧主题色（实测对比度 2.91），验证门禁真的会红，而不是永远绿
    await evalJs('(()=>{const s=document.createElement("style");'
      + 's.textContent="[data-testid=\\"orderable-list-card\\"] tbody td:first-child{color:#16A34A !important}";'
      + 'document.head.appendChild(s);return true;})()');
    log('已注入 sabotage 样式（方向列回到 #16A34A）');
  }
  const a = JSON.parse(await evalJs(AUDIT));
  console.log(JSON.stringify(a, null, 2));

  // 门禁：不是打印看看就算过，不及格要 exit 1
  const fails = [];
  if (!a.card) fails.push('可下单清单卡片没渲染出来');
  if (a.clippedCells && a.clippedCells.length) fails.push(`单元格文字被截: ${a.clippedCells.join(' / ')}`);
  if (a.cardOverflowsContainer) fails.push('卡片宽度超出容器');
  const need = 4.5;
  for (const [k, v] of Object.entries({
    方向_白底: a.ok?.contrastSideOnWhite,
    代码_白底: a.ok?.contrastCodeOnWhite,
    次要文字_白底: a.ok?.contrastCaptionOnWhite,
    方向_拦单底: a.blocked?.contrastSideOnTint,
    代码_拦单底: a.blocked?.contrastCodeOnTint,
    不可成交理由_拦单底: a.blocked?.contrastReasonOnTint,
  })) {
    if (v === null || v === undefined) continue; // 该场景今天没数据，不硬判
    if (v < need) fails.push(`对比度不及格 ${k}=${v} (<${need})`);
  }
  if (a.upgradeButtonsOnPage && a.upgradeButtonsOnPage.length) {
    fails.push(`页面上又出现了升级引导按钮: ${a.upgradeButtonsOnPage.join(' / ')}（自用口径不允许）`);
  }
  ws.close();
  if (fails.length) {
    console.log('\n门禁不通过:');
    fails.forEach((f) => console.log('  ✗ ' + f));
    process.exit(1);
  }
  console.log('\n门禁通过：无截断、无溢出、对比度均 ≥4.5、无升级引导按钮');
  process.exit(0);
})().catch((e) => {
  console.error('FAIL:', e.message);
  process.exit(1);
});
