/**
 * 性能指标卡片组件
 * 
 * 显示系统性能指标，包括：
 * - 平均响应时间
 * - 请求总数
 * - 错误统计
 * - 服务负载情况
 */

'use client';

import React, { useEffect, useState } from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Progress,
  Button,
  Tabs,
  Tab,
} from '@heroui/react';
import {
  BarChart3,
  TrendingUp,
  Zap,
  AlertCircle,
  RefreshCw,
  Clock,
  Activity,
} from 'lucide-react';
import { DataService } from '../../services/dataService';

interface PerformanceData {
  services?: Record<string, any>;
  summary?: {
    total_services: number;
    avg_response_time: number;
    total_requests: number;
    total_errors: number;
  };
}

export function PerformanceMetricsCard() {
  const [performanceData, setPerformanceData] = useState<PerformanceData | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedTab, setSelectedTab] = useState('overview');

  const loadPerformanceData = async () => {
    try {
      setLoading(true);
      const data = await DataService.getPerformanceMetrics();
      setPerformanceData(data);
    } catch (error) {
      console.error('获取性能指标失败:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPerformanceData();
    
    // 每60秒自动刷新
    const interval = setInterval(loadPerformanceData, 60000);
    return () => clearInterval(interval);
  }, []);

  const formatNumber = (num: number) => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  };

  const getErrorRate = () => {
    if (!performanceData?.summary) return 0;
    const { total_requests, total_errors } = performanceData.summary;
    return total_requests > 0 ? (total_errors / total_requests) * 100 : 0;
  };

  const getResponseTimeColor = (responseTime: number) => {
    if (responseTime < 100) return 'success';
    if (responseTime < 500) return 'warning';
    return 'danger';
  };

  if (loading && !performanceData) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-5 h-5" />
            <h3 className="text-lg font-semibold">性能指标</h3>
          </div>
        </CardHeader>
        <CardBody>
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="w-6 h-6 animate-spin text-primary" />
            <span className="ml-2 text-default-500">加载性能数据中...</span>
          </div>
        </CardBody>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between w-full">
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-5 h-5" />
            <h3 className="text-lg font-semibold">性能指标</h3>
          </div>
          <Button
            isIconOnly
            variant="light"
            size="sm"
            onPress={loadPerformanceData}
            isLoading={loading}
          >
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>
      </CardHeader>
      <CardBody>
        <Tabs
          selectedKey={selectedTab}
          onSelectionChange={(key) => setSelectedTab(key as string)}
          variant="underlined"
          classNames={{
            tabList: "gap-6 w-full relative rounded-none p-0 border-b border-divider",
            cursor: "w-full bg-primary",
            tab: "max-w-fit px-0 h-12",
          }}
        >
          <Tab key="overview" title="概览">
            <div className="space-y-6 pt-4">
              {/* 关键指标 */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-primary-50 rounded-lg">
                  <div className="flex items-center justify-center mb-2">
                    <Clock className="w-5 h-5 text-primary" />
                  </div>
                  <p className="text-2xl font-bold text-primary">
                    {performanceData?.summary?.avg_response_time?.toFixed(0) || 0}ms
                  </p>
                  <p className="text-sm text-default-500">平均响应时间</p>
                </div>
                
                <div className="text-center p-4 bg-success-50 rounded-lg">
                  <div className="flex items-center justify-center mb-2">
                    <TrendingUp className="w-5 h-5 text-success" />
                  </div>
                  <p className="text-2xl font-bold text-success">
                    {formatNumber(performanceData?.summary?.total_requests || 0)}
                  </p>
                  <p className="text-sm text-default-500">总请求数</p>
                </div>
                
                <div className="text-center p-4 bg-warning-50 rounded-lg">
                  <div className="flex items-center justify-center mb-2">
                    <AlertCircle className="w-5 h-5 text-warning" />
                  </div>
                  <p className="text-2xl font-bold text-warning">
                    {formatNumber(performanceData?.summary?.total_errors || 0)}
                  </p>
                  <p className="text-sm text-default-500">错误总数</p>
                </div>
                
                <div className="text-center p-4 bg-secondary-50 rounded-lg">
                  <div className="flex items-center justify-center mb-2">
                    <Activity className="w-5 h-5 text-secondary" />
                  </div>
                  <p className="text-2xl font-bold text-secondary">
                    {performanceData?.summary?.total_services || 0}
                  </p>
                  <p className="text-sm text-default-500">活跃服务</p>
                </div>
              </div>

              {/* 错误率 */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">错误率</span>
                  <span className="text-sm text-default-500">
                    {getErrorRate().toFixed(2)}%
                  </span>
                </div>
                <Progress
                  value={getErrorRate()}
                  color={getErrorRate() > 5 ? 'danger' : getErrorRate() > 1 ? 'warning' : 'success'}
                  className="w-full"
                />
              </div>

              {/* 响应时间状态 */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">响应时间状态</span>
                  <span className="text-sm text-default-500">
                    {performanceData?.summary?.avg_response_time || 0}ms
                  </span>
                </div>
                <Progress
                  value={Math.min((performanceData?.summary?.avg_response_time || 0) / 10, 100)}
                  color={getResponseTimeColor(performanceData?.summary?.avg_response_time || 0)}
                  className="w-full"
                />
              </div>
            </div>
          </Tab>
          
          <Tab key="services" title="服务详情">
            <div className="space-y-4 pt-4">
              {performanceData?.services && Object.entries(performanceData.services).map(([serviceName, metrics]) => (
                <div key={serviceName} className="p-4 bg-default-50 rounded-lg">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="font-medium">{serviceName}</h4>
                    <div className="flex items-center space-x-2">
                      <Zap className="w-4 h-4 text-primary" />
                      <span className="text-sm font-medium">
                        {metrics.avg_response_time?.toFixed(0) || 0}ms
                      </span>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="text-default-500">请求数</p>
                      <p className="font-medium">{formatNumber(metrics.request_count || 0)}</p>
                    </div>
                    <div>
                      <p className="text-default-500">错误数</p>
                      <p className="font-medium text-danger">{metrics.error_count || 0}</p>
                    </div>
                    <div>
                      <p className="text-default-500">成功率</p>
                      <p className="font-medium text-success">
                        {metrics.request_count > 0 
                          ? (((metrics.request_count - metrics.error_count) / metrics.request_count) * 100).toFixed(1)
                          : 100
                        }%
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Tab>
        </Tabs>
      </CardBody>
    </Card>
  );
}