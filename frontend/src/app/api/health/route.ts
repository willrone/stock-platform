/**
 * 前端健康检查API端点
 */

import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  try {
    const healthData = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      service: 'stock-prediction-frontend',
      version: '1.0.0',
      uptime: process.uptime(),
      environment: process.env.NODE_ENV || 'development',
      memory: {
        used: process.memoryUsage().heapUsed,
        total: process.memoryUsage().heapTotal,
        external: process.memoryUsage().external,
        rss: process.memoryUsage().rss
      },
      backend_connection: {
        url: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
        status: 'unknown' // 这里可以添加对后端的连接检查
      }
    };

    return NextResponse.json({
      success: true,
      message: '前端服务运行正常',
      data: healthData
    });
  } catch (error) {
    return NextResponse.json({
      success: false,
      message: '健康检查失败',
      error: error instanceof Error ? error.message : '未知错误'
    }, { status: 500 });
  }
}