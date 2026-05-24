const assert = require('node:assert/strict');
const test = require('node:test');

const {
  createLocalSmokeChecks,
  runLocalSmoke,
  summarizeResults,
} = require('../../scripts/smoke-local');

test('createLocalSmokeChecks uses local development defaults and supports port overrides', () => {
  const checks = createLocalSmokeChecks({
    STOCK_PLATFORM_FRONTEND_PORT: '13001',
    STOCK_PLATFORM_BACKEND_PORT: '18083',
    STOCK_PLATFORM_DATA_API_PORT: '5003',
    STOCK_PLATFORM_METRICS_PORT: '19091',
  });

  const byName = Object.fromEntries(checks.map(check => [check.name, check]));

  assert.equal(byName['frontend /'].url, 'http://127.0.0.1:13001/');
  assert.equal(byName['frontend /data'].url, 'http://127.0.0.1:13001/data');
  assert.equal(byName['frontend /monitoring'].url, 'http://127.0.0.1:13001/monitoring');
  assert.equal(byName['backend health'].url, 'http://127.0.0.1:18083/api/v1/health');
  assert.equal(byName['backend data status'].url, 'http://127.0.0.1:18083/api/v1/data/status');
  assert.equal(byName['data-api health'].url, 'http://127.0.0.1:5003/api/data/health');
  assert.equal(byName['metrics'].url, 'http://127.0.0.1:19091/metrics');
  assert.equal(byName['metrics'].fallbackUrl, 'http://127.0.0.1:18083/metrics');
});

test('runLocalSmoke reports endpoint, payload, marker, and fallback results', async () => {
  const responses = new Map([
    ['http://127.0.0.1:13000/', { statusCode: 200, body: '<html>股票预测平台</html>' }],
    ['http://127.0.0.1:13000/data', { statusCode: 200, body: '<html>数据管理</html>' }],
    [
      'http://127.0.0.1:13000/monitoring',
      { statusCode: 200, body: '<html>Application error</html>' },
    ],
    [
      'http://127.0.0.1:18082/api/v1/health',
      { statusCode: 200, body: JSON.stringify({ success: true, data: { status: 'healthy' } }) },
    ],
    [
      'http://127.0.0.1:18082/api/v1/data/status',
      { statusCode: 200, body: JSON.stringify({ success: true, data: { is_connected: false } }) },
    ],
    [
      'http://127.0.0.1:18082/api/v1/monitoring/health',
      { statusCode: 200, body: JSON.stringify({ success: true, data: { overall_healthy: true } }) },
    ],
    [
      'http://127.0.0.1:5002/api/data/health',
      { statusCode: 200, body: JSON.stringify({ status: 'healthy', storage_available: true }) },
    ],
    ['http://127.0.0.1:19090/metrics', new Error('ECONNREFUSED')],
    ['http://127.0.0.1:18082/metrics', { statusCode: 200, body: '# HELP requests_total\n' }],
  ]);

  const results = await runLocalSmoke({
    request: async url => {
      const response = responses.get(url);
      if (response instanceof Error) {
        throw response;
      }
      if (!response) {
        throw new Error(`unexpected URL: ${url}`);
      }
      return response;
    },
  });

  const byName = Object.fromEntries(results.map(result => [result.name, result]));

  assert.equal(byName['frontend /'].ok, true);
  assert.equal(byName['frontend /monitoring'].ok, false);
  assert.match(byName['frontend /monitoring'].message, /Application error/);
  assert.equal(byName['backend data status'].ok, false);
  assert.match(byName['backend data status'].message, /is_connected/);
  assert.equal(byName.metrics.ok, true);
  assert.equal(byName.metrics.usedFallback, true);

  const summary = summarizeResults(results);
  assert.equal(summary.ok, false);
  assert.equal(summary.failed, 2);
});
