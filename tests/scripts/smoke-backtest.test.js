const assert = require('node:assert/strict');
const test = require('node:test');

const {
  buildSmokeBacktestPayload,
  createBacktestSmokeChecks,
  runBacktestSmoke,
  validateBacktestResponse,
  validateStrategiesResponse,
  validateStrategyConfigsResponse,
} = require('../../scripts/smoke-backtest.js');

test('createBacktestSmokeChecks targets required backend endpoints', () => {
  const checks = createBacktestSmokeChecks({ STOCK_PLATFORM_BACKEND_PORT: '18082' });

  assert.deepEqual(
    checks.map(check => `${check.method} ${new URL(check.url).pathname}`),
    ['GET /api/v1/backtest/strategies', 'GET /api/v1/strategy-configs', 'POST /api/v1/backtest'],
  );
});

test('validateStrategiesResponse requires the baseline strategy keys', () => {
  const required = ['moving_average', 'rsi', 'macd', 'model_topk_dropout'];
  const response = {
    success: true,
    data: required.map(key => ({ key })),
  };

  assert.deepEqual(validateStrategiesResponse(response), { ok: true });

  const missing = { success: true, data: [{ key: 'moving_average' }] };
  assert.equal(validateStrategiesResponse(missing).ok, false);
  assert.match(validateStrategiesResponse(missing).message, /missing strategies/);
});

test('validateStrategyConfigsResponse accepts the standard config list response', () => {
  assert.deepEqual(
    validateStrategyConfigsResponse({
      success: true,
      data: { configs: [], total_count: 0 },
    }),
    { ok: true },
  );
});

test('validateBacktestResponse enforces required response fields and warns on zero trades', () => {
  const response = {
    success: true,
    data: {
      portfolio: { initial_cash: 100000, final_value: 100000 },
      trading_stats: { total_trades: 0 },
      risk_metrics: { max_drawdown: 0 },
      dates: ['2024-01-02', '2024-01-03'],
    },
  };

  const validation = validateBacktestResponse(response);

  assert.equal(validation.ok, true);
  assert.deepEqual(validation.warnings, ['短样本仅验证链路，不评价策略收益']);
});

test('buildSmokeBacktestPayload uses a short local moving-average sample', () => {
  const payload = buildSmokeBacktestPayload({
    SMOKE_BACKTEST_STOCK_CODES: '000001,600000',
    SMOKE_BACKTEST_START_DATE: '2024-01-02',
    SMOKE_BACKTEST_END_DATE: '2024-02-01',
  });

  assert.equal(payload.strategy_name, 'moving_average');
  assert.deepEqual(payload.stock_codes, ['000001', '600000']);
  assert.equal(payload.strategy_config.signal_threshold, 0.005);
  assert.equal(payload.initial_cash, 100000);
});

test('runBacktestSmoke executes checks and preserves warnings', async () => {
  const fakeRequest = async check => {
    if (check.name === 'strategies') {
      return {
        statusCode: 200,
        json: {
          success: true,
          data: ['moving_average', 'rsi', 'macd', 'model_topk_dropout'].map(key => ({ key })),
        },
      };
    }
    if (check.name === 'strategy-configs') {
      return { statusCode: 200, json: { success: true, data: { configs: [], total_count: 0 } } };
    }
    return {
      statusCode: 200,
      json: {
        success: true,
        data: {
          portfolio: { initial_cash: 100000, final_value: 100000 },
          trading_stats: { total_trades: 0 },
          risk_metrics: { max_drawdown: 0 },
          dates: ['2024-01-02'],
        },
      },
    };
  };

  const results = await runBacktestSmoke({ request: fakeRequest });

  assert.equal(results.every(result => result.ok), true);
  assert.deepEqual(results.at(-1).warnings, ['短样本仅验证链路，不评价策略收益']);
});
