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
  CardContent,
  CardHeader,
  Chip,
  Tooltip,
  Button,
  Box,
  Typography,
  IconButton,
  CircularProgress,
} from '@mui/material';
import { Activity, CheckCircle, XCircle, AlertTriangle, RefreshCw, Clock } from 'lucide-react';
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
      return <CheckCircle size={16} color="#2e7d32" />;
    } else {
      return <XCircle size={16} color="#d32f2f" />;
    }
  };

  const getServiceStatus = (service: ServiceHealth) => {
    if (service.healthy) {
      return { color: 'success' as const, text: '正常' };
    } else {
      return { color: 'error' as const, text: '异常' };
    }
  };

  const getResponseTimeColor = (responseTime: number): string => {
    if (responseTime < 100) {
      return 'success.main';
    }
    if (responseTime < 500) {
      return 'warning.main';
    }
    return 'error.main';
  };

  const formatServiceName = (serviceName: string) => {
    const nameMap: Record<string, string> = {
      data_service: '数据服务',
      indicators_service: '指标服务',
      parquet_manager: '文件管理',
      sync_engine: '同步引擎',
    };
    return nameMap[serviceName] || serviceName;
  };

  if (loading && !healthData) {
    return (
      <Card>
        <CardHeader avatar={<Activity size={24} />} title="系统健康状态" />
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', py: 4 }}>
            <CircularProgress size={24} sx={{ mr: 1 }} />
            <Typography variant="body2" color="text.secondary">
              检查系统状态中...
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader
        avatar={<Activity size={24} />}
        title="系统健康状态"
        action={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip
              label={healthData?.overall_healthy ? '系统正常' : '系统异常'}
              color={healthData?.overall_healthy ? 'success' : 'error'}
              size="small"
              icon={
                healthData?.overall_healthy ? (
                  <CheckCircle size={12} />
                ) : (
                  <AlertTriangle size={12} />
                )
              }
            />
            <IconButton size="small" onClick={loadHealthData} disabled={loading}>
              <RefreshCw size={16} />
            </IconButton>
          </Box>
        }
      />
      <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {healthData?.services &&
          Object.entries(healthData.services).map(([serviceName, service]) => (
            <Box
              key={serviceName}
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                p: 1.5,
                bgcolor: 'grey.50',
                borderRadius: 1,
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                {getServiceIcon(service)}
                <Box>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {formatServiceName(serviceName)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    最后检查: {new Date(service.last_check).toLocaleTimeString()}
                  </Typography>
                </Box>
              </Box>

              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                <Box sx={{ textAlign: 'right' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Clock size={12} color="#666" />
                    <Typography
                      variant="caption"
                      sx={{
                        fontWeight: 500,
                        color: getResponseTimeColor(service.response_time_ms),
                      }}
                    >
                      {service.response_time_ms}ms
                    </Typography>
                  </Box>
                  {service.error_message && (
                    <Tooltip title={service.error_message}>
                      <Typography variant="caption" color="error" sx={{ cursor: 'help' }}>
                        错误详情
                      </Typography>
                    </Tooltip>
                  )}
                </Box>

                <Chip
                  label={getServiceStatus(service).text}
                  color={getServiceStatus(service).color}
                  size="small"
                />
              </Box>
            </Box>
          ))}

        <Box sx={{ pt: 2, borderTop: 1, borderColor: 'divider' }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Typography variant="caption" color="text.secondary">
              最后更新: {lastUpdate.toLocaleTimeString()}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              自动刷新: 30秒
            </Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
}
