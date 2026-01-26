/**
 * 前端指标收集API端点
 */

import { NextRequest, NextResponse } from 'next/server';

// 简单的内存指标存储（生产环境应使用Redis等）
const metrics = {
  page_views: 0,
  api_calls: 0,
  errors: 0,
  last_reset: Date.now(),
};

export async function GET(_request: NextRequest) {
  try {
    const prometheusMetrics = generatePrometheusMetrics();

    return new Response(prometheusMetrics, {
      headers: {
        'Content-Type': 'text/plain; version=0.0.4; charset=utf-8',
      },
    });
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        message: '指标收集失败',
        error: error instanceof Error ? error.message : '未知错误',
      },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { type, value = 1 } = body;

    // 更新指标
    switch (type) {
      case 'page_view':
        metrics.page_views += value;
        break;
      case 'api_call':
        metrics.api_calls += value;
        break;
      case 'error':
        metrics.errors += value;
        break;
      default:
        return NextResponse.json(
          {
            success: false,
            message: '未知的指标类型',
          },
          { status: 400 }
        );
    }

    return NextResponse.json({
      success: true,
      message: '指标更新成功',
      data: metrics,
    });
  } catch (error) {
    return NextResponse.json(
      {
        success: false,
        message: '指标更新失败',
        error: error instanceof Error ? error.message : '未知错误',
      },
      { status: 500 }
    );
  }
}

function generatePrometheusMetrics(): string {
  const now = Date.now();
  const uptime = (now - metrics.last_reset) / 1000;

  return `# HELP frontend_page_views_total 页面访问总数
# TYPE frontend_page_views_total counter
frontend_page_views_total ${metrics.page_views}

# HELP frontend_api_calls_total API调用总数
# TYPE frontend_api_calls_total counter
frontend_api_calls_total ${metrics.api_calls}

# HELP frontend_errors_total 错误总数
# TYPE frontend_errors_total counter
frontend_errors_total ${metrics.errors}

# HELP frontend_uptime_seconds 运行时间（秒）
# TYPE frontend_uptime_seconds gauge
frontend_uptime_seconds ${uptime}

# HELP frontend_memory_usage_bytes 内存使用量（字节）
# TYPE frontend_memory_usage_bytes gauge
frontend_memory_usage_bytes ${process.memoryUsage().heapUsed}

# HELP frontend_info 前端应用信息
# TYPE frontend_info info
frontend_info{version="1.0.0",service="stock-prediction-frontend",environment="${
    process.env.NODE_ENV || 'development'
  }"} 1
`;
}
