#!/usr/bin/env node

const http = require('node:http');
const https = require('node:https');

const DEFAULT_TIMEOUT_MS = Number(process.env.SMOKE_BACKTEST_TIMEOUT_MS || 30000);
const ZERO_TRADE_WARNING = '短样本仅验证链路，不评价策略收益';

function envValue(env, name, fallback) {
  const value = env[name];
  return value && String(value).trim() ? String(value).trim() : fallback;
}

function buildSmokeBacktestPayload(env = process.env) {
  const stockCodes = envValue(env, 'SMOKE_BACKTEST_STOCK_CODES', '000001.SZ')
    .split(',')
    .map(item => item.trim())
    .filter(Boolean);

  return {
    strategy_name: 'moving_average',
    stock_codes: stockCodes,
    start_date: envValue(env, 'SMOKE_BACKTEST_START_DATE', '2024-01-02'),
    end_date: envValue(env, 'SMOKE_BACKTEST_END_DATE', '2024-02-01'),
    initial_cash: Number(envValue(env, 'SMOKE_BACKTEST_INITIAL_CASH', '100000')),
    strategy_config: {
      short_window: Number(envValue(env, 'SMOKE_BACKTEST_SHORT_WINDOW', '5')),
      long_window: Number(envValue(env, 'SMOKE_BACKTEST_LONG_WINDOW', '20')),
      signal_threshold: Number(envValue(env, 'SMOKE_BACKTEST_SIGNAL_THRESHOLD', '0.005')),
      commission_rate: Number(envValue(env, 'SMOKE_BACKTEST_COMMISSION_RATE', '0.0003')),
      slippage_rate: Number(envValue(env, 'SMOKE_BACKTEST_SLIPPAGE_RATE', '0.0001')),
    },
  };
}

function createBacktestSmokeChecks(env = process.env) {
  const backendPort = envValue(env, 'STOCK_PLATFORM_BACKEND_PORT', '18082');
  const backendBase = envValue(env, 'STOCK_PLATFORM_BACKEND_URL', `http://127.0.0.1:${backendPort}`)
    .replace(/\/$/, '');

  return [
    {
      name: 'strategies',
      method: 'GET',
      url: `${backendBase}/api/v1/backtest/strategies`,
      validateJson: validateStrategiesResponse,
    },
    {
      name: 'strategy-configs',
      method: 'GET',
      url: `${backendBase}/api/v1/strategy-configs`,
      validateJson: validateStrategyConfigsResponse,
    },
    {
      name: 'backtest',
      method: 'POST',
      url: `${backendBase}/api/v1/backtest`,
      body: buildSmokeBacktestPayload(env),
      validateJson: validateBacktestResponse,
    },
  ];
}

function parseJsonBody(body) {
  try {
    return { ok: true, json: JSON.parse(body) };
  } catch (error) {
    return { ok: false, message: `invalid JSON: ${error.message}` };
  }
}

function requestCheck(check, timeoutMs = DEFAULT_TIMEOUT_MS) {
  const url = new URL(check.url);
  const client = url.protocol === 'https:' ? https : http;
  const body = check.body ? JSON.stringify(check.body) : undefined;

  return new Promise((resolve, reject) => {
    const request = client.request(
      url,
      {
        method: check.method || 'GET',
        headers: body
          ? { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) }
          : undefined,
      },
      response => {
        let responseBody = '';
        response.setEncoding('utf8');
        response.on('data', chunk => {
          responseBody += chunk;
        });
        response.on('end', () => {
          const parsed = parseJsonBody(responseBody);
          resolve({
            statusCode: response.statusCode || 0,
            headers: response.headers,
            body: responseBody,
            json: parsed.ok ? parsed.json : undefined,
            parseError: parsed.ok ? undefined : parsed.message,
          });
        });
      },
    );

    request.setTimeout(timeoutMs, () => request.destroy(new Error(`Timeout after ${timeoutMs}ms`)));
    request.on('error', reject);
    if (body) {
      request.write(body);
    }
    request.end();
  });
}

function validateStrategiesResponse(json) {
  if (json?.success !== true || !Array.isArray(json?.data)) {
    return { ok: false, message: 'expected success=true and data array' };
  }

  const keys = new Set(json.data.map(item => (typeof item === 'string' ? item : item?.key)).filter(Boolean));
  const required = ['moving_average', 'rsi', 'macd', 'model_topk_dropout'];
  const missing = required.filter(key => !keys.has(key));
  if (missing.length > 0) {
    return { ok: false, message: `missing strategies: ${missing.join(', ')}` };
  }
  return { ok: true };
}

function validateStrategyConfigsResponse(json) {
  if (json?.success !== true) {
    return { ok: false, message: 'expected success=true' };
  }
  if (!json.data || !Array.isArray(json.data.configs) || typeof json.data.total_count !== 'number') {
    return { ok: false, message: 'expected data.configs array and data.total_count number' };
  }
  return { ok: true };
}

function hasPath(object, path) {
  let current = object;
  for (const key of path.split('.')) {
    if (current == null || !Object.prototype.hasOwnProperty.call(current, key)) {
      return false;
    }
    current = current[key];
  }
  return true;
}

function validateBacktestResponse(json) {
  if (json?.success !== true || !json?.data) {
    return { ok: false, message: 'expected success=true and data object' };
  }

  const requiredPaths = [
    'portfolio.initial_cash',
    'portfolio.final_value',
    'trading_stats.total_trades',
    'risk_metrics.max_drawdown',
    'dates',
  ];
  const missing = requiredPaths.filter(path => !hasPath(json.data, path));
  if (missing.length > 0) {
    return { ok: false, message: `missing backtest fields: ${missing.join(', ')}` };
  }
  if (!Array.isArray(json.data.dates)) {
    return { ok: false, message: 'dates must be an array' };
  }

  const warnings = [];
  if (Number(json.data.trading_stats.total_trades || 0) === 0) {
    warnings.push(ZERO_TRADE_WARNING);
  }
  return { ok: true, warnings };
}

async function executeCheck(check, request = requestCheck) {
  try {
    const response = await request(check);
    if (response.statusCode < 200 || response.statusCode >= 300) {
      return { ...check, ok: false, message: `HTTP ${response.statusCode}` };
    }
    if (response.parseError) {
      return { ...check, ok: false, message: response.parseError };
    }
    const validation = check.validateJson(response.json);
    return {
      ...check,
      ok: validation.ok,
      message: validation.ok ? `HTTP ${response.statusCode}` : validation.message,
      warnings: validation.warnings || [],
      statusCode: response.statusCode,
    };
  } catch (error) {
    return { ...check, ok: false, message: error.message, warnings: [] };
  }
}

async function runBacktestSmoke(options = {}) {
  const checks = options.checks || createBacktestSmokeChecks(options.env || process.env);
  const request = options.request || requestCheck;
  const results = [];
  for (const check of checks) {
    results.push(await executeCheck(check, request));
  }
  return results;
}

function summarizeResults(results) {
  const failed = results.filter(result => !result.ok).length;
  return {
    ok: failed === 0,
    passed: results.length - failed,
    failed,
    total: results.length,
  };
}

function printResults(results) {
  for (const result of results) {
    const status = result.ok ? 'OK' : 'FAIL';
    console.log(`${status} ${result.name} - ${result.message}`);
    for (const warning of result.warnings || []) {
      console.warn(`WARN ${result.name} - ${warning}`);
    }
  }
  const summary = summarizeResults(results);
  console.log(`\nSummary: ${summary.passed}/${summary.total} passed`);
  return summary;
}

async function main() {
  const results = await runBacktestSmoke();
  const summary = printResults(results);
  process.exit(summary.ok ? 0 : 1);
}

if (require.main === module) {
  main().catch(error => {
    console.error(`FAIL smoke-backtest crashed: ${error.stack || error.message}`);
    process.exit(1);
  });
}

module.exports = {
  ZERO_TRADE_WARNING,
  buildSmokeBacktestPayload,
  createBacktestSmokeChecks,
  executeCheck,
  requestCheck,
  runBacktestSmoke,
  summarizeResults,
  validateBacktestResponse,
  validateStrategiesResponse,
  validateStrategyConfigsResponse,
};
