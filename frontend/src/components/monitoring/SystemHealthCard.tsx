/**
 * 系统健康状态卡片组件
 * 
 * 显示各个服务的健康状态，包括：
 * - 整体健康状态
 * - 各服务详细状态
 * - 响应时间监控
 * - 错误信息展示
 */

'use client';

import React, { useEffect, useState } from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Chip,
  Progress,
  Tooltip,
  Button,
} from '@heroui/react';
import {
  Activity,
  CheckCircle,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Zap,
  Clock,
} from 'lucide-react';
import { DataService } from '../../services/dataService';

interface ServiceHealth {
  healthy: boolean;
  response_time_ms: number;
  last_check: string;
  error_message: string | null;
}

interface SystemHealthData {
  overall_healthy: boolean;
  services: Record<string, ServiceHealth>;
  check_time: string;
}

export function SystemHealthCard() {
  const [healthData, setHealthData] = useState<SystemHealthData | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const loadHealthData = async () => {
    try {
      setLoading(true);
      const data = await DataService.getSystemHealth();
      setHealthData(data);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('获取系统健康状态失败:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHealthData();
    
    // 每30秒自动刷新
    const interval = setInterval(loadHealthData, 30000);
    return () => clearInterval(interval);
  }, []);

  const getServiceIcon = (service: ServiceHealth) => {
    if (service.healthy) {
      return <CheckCircle className="w-4 h-4 text-success" />;
    } else {
      return <XCircle className="w-4 h-4 text-danger" />;
    }
  };

  const getServiceStatus = (service: ServiceHealth) => {
    if (service.healthy) {
      return { color: 'success' as const, text: '正常' };
    } else {
      return { color: 'danger' as const, text: '异常' };
    }
  };

  const getResponseTimeColor = (responseTime: number) => {
    if (responseTime < 100) return 'text-success';
    if (responseTime < 500) return 'text-warning';
    return 'text-danger';
  };

  const formatServiceName = (serviceName: string) => {
    const nameMap: Record<string, string> = {
      'data_service': '数据服务',
      'indicators_service': '指标服务',
      'parquet_manager': '文件管理',
      'sync_engine': '同步引擎',
    };
    return nameMap[serviceName] || serviceName;
  };

  if (loading && !healthData) {
    return (
      <Card>
        <CardHeader>
          <div className="flex items-center space-x-2">
            <Activity className="w-5 h-5" />
            <h3 className="text-lg font-semibold">系统健康状态</h3>
          </div>
        </CardHeader>
        <CardBody>
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="w-6 h-6 animate-spin text-primary" />
            <span className="ml-2 text-default-500">检查系统状态中...</span>
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
            <Activity className="w-5 h-5" />
            <h3 className="text-lg font-semibold">系统健康状态</h3>
          </div>
          <div className="flex items-center space-x-2">
            <Chip
              color={healthData?.overall_healthy ? 'success' : 'danger'}
              variant="flat"
              size="sm"
              startContent={
                healthData?.overall_healthy ? (
                  <CheckCircle className="w-3 h-3" />
                ) : (
                  <AlertTriangle className="w-3 h-3" />
                )
              }
            >
              {healthData?.overall_healthy ? '系统正常' : '系统异常'}
            </Chip>
            <Button
              isIconOnly
              variant="light"
              size="sm"
              onPress={loadHealthData}
              isLoading={loading}
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardBody className="space-y-4">
        {healthData?.services && Object.entries(healthData.services).map(([serviceName, service]) => (
          <div key={serviceName} className="flex items-center justify-between p-3 bg-default-50 rounded-lg">
            <div className="flex items-center space-x-3">
              {getServiceIcon(service)}
              <div>
                <p className="font-medium">{formatServiceName(serviceName)}</p>
                <p className="text-xs text-default-500">
                  最后检查: {new Date(service.last_check).toLocaleTimeString()}
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <div className="text-right">
                <div className="flex items-center space-x-1">
                  <Clock className="w-3 h-3 text-default-400" />
                  <span className={`text-sm font-medium ${getResponseTimeColor(service.response_time_ms)}`}>
                    {service.response_time_ms}ms
                  </span>
                </div>
                {service.error_message && (
                  <Tooltip content={service.error_message}>
                    <p className="text-xs text-danger cursor-help">
                      错误详情
                    </p>
                  </Tooltip>
                )}
              </div>
              
              <Chip
                color={getServiceStatus(service).color}
                variant="flat"
                size="sm"
              >
                {getServiceStatus(service).text}
              </Chip>
            </div>
          </div>
        ))}
        
        <div className="pt-2 border-t border-default-200">
          <div className="flex items-center justify-between text-sm text-default-500">
            <span>最后更新: {lastUpdate.toLocaleTimeString()}</span>
            <span>自动刷新: 30秒</span>
          </div>
        </div>
      </CardBody>
    </Card>
  );
}