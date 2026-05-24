#!/usr/bin/env node

const http = require('node:http');
const https = require('node:https');

const DEFAULT_TIMEOUT_MS = Number(process.env.SMOKE_TIMEOUT_MS || 8000);
const fatalMarkers = [
  'ReactServerComponentsError',
  'Unhandled Runtime Error',
  'Application error',
  'Internal Server Error',
  'Cannot find module',
  'Failed to compile',
  'Hydration failed',
];

function envPort(env, name, fallback) {
  const value = env[name];
  return value && String(value).trim() ? String(value).trim() : fallback;
}

function createLocalSmokeChecks(env = process.env) {
  const frontendPort = envPort(env, 'STOCK_PLATFORM_FRONTEND_PORT', '13000');
  const backendPort = envPort(env, 'STOCK_PLATFORM_BACKEND_PORT', '18082');
  const dataApiPort = envPort(env, 'STOCK_PLATFORM_DATA_API_PORT', '5002');
  const metricsPort = envPort(env, 'STOCK_PLATFORM_METRICS_PORT', '19090');

  const frontendBase = `http://127.0.0.1:${frontendPort}`;
  const backendBase = `http://127.0.0.1:${backendPort}`;
  const dataApiBase = `http://127.0.0.1:${dataApiPort}`;
  const metricsBase = `http://127.0.0.1:${metricsPort}`;

  return [
    {
      name: 'frontend /',
      url: `${frontendBase}/`,
      expectText: '股票预测平台',
    },
    {
      name: 'frontend /data',
      url: `${frontendBase}/data`,
      expectText: '数据管理',
    },
    {
      name: 'frontend /monitoring',
      url: `${frontendBase}/monitoring`,
      expectText: '系统监控',
    },
    {
      name: 'backend health',
      url: `${backendBase}/api/v1/health`,
      expectJson: json => json?.success === true && json?.data?.status === 'healthy',
      expectDescription: 'success=true and data.status=healthy',
    },
    {
      name: 'backend data status',
      url: `${backendBase}/api/v1/data/status`,
      expectJson: json => json?.success === true && json?.data?.is_connected === true,
      expectDescription: 'success=true and data.is_connected=true',
    },
    {
      name: 'backend monitoring health',
      url: `${backendBase}/api/v1/monitoring/health`,
      expectJson: json => json?.success === true && json?.data?.overall_healthy === true,
      expectDescription: 'success=true and data.overall_healthy=true',
    },
    {
      name: 'data-api health',
      url: `${dataApiBase}/api/data/health`,
      expectJson: json => json?.status === 'healthy' && json?.storage_available === true,
      expectDescription: 'status=healthy and storage_available=true',
    },
    {
      name: 'metrics',
      url: `${metricsBase}/metrics`,
      fallbackUrl: `${backendBase}/metrics`,
      expectText: '# HELP',
      optionalPrimary: true,
    },
  ];
}

function requestUrl(url, timeoutMs = DEFAULT_TIMEOUT_MS) {
  const client = url.startsWith('https:') ? https : http;

  return new Promise((resolve, reject) => {
    const request = client.get(url, response => {
      let body = '';
      response.setEncoding('utf8');
      response.on('data', chunk => {
        body += chunk;
      });
      response.on('end', () => {
        resolve({
          statusCode: response.statusCode || 0,
          headers: response.headers,
          body,
        });
      });
    });

    request.setTimeout(timeoutMs, () => request.destroy(new Error(`Timeout after ${timeoutMs}ms`)));
    request.on('error', reject);
  });
}

function validateResponse(check, response) {
  const statusCode = response.statusCode || 0;
  if (statusCode < 200 || statusCode >= 400) {
    return { ok: false, message: `HTTP ${statusCode}` };
  }

  const marker = fatalMarkers.find(item => response.body.includes(item));
  if (marker) {
    return { ok: false, message: `fatal marker found: ${marker}` };
  }

  if (check.expectText && !response.body.includes(check.expectText)) {
    return { ok: false, message: `missing text: ${check.expectText}` };
  }

  if (check.expectJson) {
    let json;
    try {
      json = JSON.parse(response.body);
    } catch (error) {
      return { ok: false, message: `invalid JSON: ${error.message}` };
    }

    if (!check.expectJson(json)) {
      return { ok: false, message: `JSON expectation failed: ${check.expectDescription}` };
    }
  }

  return { ok: true, message: `HTTP ${statusCode}` };
}

async function executeCheck(check, request = requestUrl) {
  try {
    const response = await request(check.url);
    const validation = validateResponse(check, response);
    if (validation.ok || !check.fallbackUrl) {
      return { ...check, ...validation, statusCode: response.statusCode, usedFallback: false };
    }
    return executeFallback(check, request, `${check.url} failed: ${validation.message}`);
  } catch (error) {
    if (check.fallbackUrl) {
      return executeFallback(check, request, `${check.url} failed: ${error.message}`);
    }
    return { ...check, ok: false, message: error.message, usedFallback: false };
  }
}

async function executeFallback(check, request, primaryFailure) {
  const fallbackCheck = { ...check, url: check.fallbackUrl, fallbackUrl: undefined };
  try {
    const response = await request(check.fallbackUrl);
    const validation = validateResponse(fallbackCheck, response);
    return {
      ...check,
      ...validation,
      statusCode: response.statusCode,
      usedFallback: true,
      primaryFailure,
      message: validation.ok ? `fallback OK (${primaryFailure})` : `${validation.message}; ${primaryFailure}`,
    };
  } catch (error) {
    return {
      ...check,
      ok: false,
      message: `${check.fallbackUrl} failed: ${error.message}; ${primaryFailure}`,
      usedFallback: true,
      primaryFailure,
    };
  }
}

async function runLocalSmoke(options = {}) {
  const checks = options.checks || createLocalSmokeChecks(options.env || process.env);
  const request = options.request || requestUrl;
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
    const fallback = result.usedFallback ? ' fallback=backend' : '';
    console.log(`${status} ${result.name}${fallback} - ${result.message}`);
  }

  const summary = summarizeResults(results);
  console.log(`\nSummary: ${summary.passed}/${summary.total} passed`);
  return summary;
}

async function main() {
  const results = await runLocalSmoke();
  const summary = printResults(results);
  process.exit(summary.ok ? 0 : 1);
}

if (require.main === module) {
  main().catch(error => {
    console.error(`FAIL smoke-local crashed: ${error.stack || error.message}`);
    process.exit(1);
  });
}

module.exports = {
  createLocalSmokeChecks,
  fatalMarkers,
  requestUrl,
  runLocalSmoke,
  summarizeResults,
  validateResponse,
};
