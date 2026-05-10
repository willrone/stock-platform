#!/usr/bin/env node

const http = require('http');

const baseUrl = process.env.SMOKE_BASE_URL || 'http://127.0.0.1:13000';
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

const fatalMarkers = [
  'ReactServerComponentsError',
  'Unhandled Runtime Error',
  'Application error',
  'Internal Server Error',
  'Cannot find module',
  'Failed to compile',
  'Hydration failed',
];

function fetchPage(url) {
  return new Promise((resolve, reject) => {
    const request = http.get(url, response => {
      let body = '';
      response.setEncoding('utf8');
      response.on('data', chunk => {
        body += chunk;
      });
      response.on('end', () => {
        resolve({ statusCode: response.statusCode || 0, body });
      });
    });

    request.setTimeout(30000, () => request.destroy(new Error(`Timeout fetching ${url}`)));
    request.on('error', reject);
  });
}

(async () => {
  const failures = [];

  for (const path of paths) {
    const url = `${baseUrl}${path}`;
    try {
      const { statusCode, body } = await fetchPage(url);
      const marker = fatalMarkers.find(item => body.includes(item));
      const ok = statusCode >= 200 && statusCode < 400 && !marker;
      console.log(`${ok ? 'OK' : 'FAIL'} ${path} ${statusCode}${marker ? ` marker=${marker}` : ''}`);
      if (!ok) {
        failures.push({ path, statusCode, marker });
      }
    } catch (error) {
      console.log(`FAIL ${path} ${error.message}`);
      failures.push({ path, error: error.message });
    }
  }

  if (failures.length > 0) {
    console.error('Page smoke failed:', JSON.stringify(failures, null, 2));
    process.exit(1);
  }
})();
