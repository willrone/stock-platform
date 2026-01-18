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
  CardContent,
  CardHeader,
  LinearProgress,
  Button,
  Tabs,
  Tab,
  Box,
  Typography,
  IconButton,
  CircularProgress,
} from '@mui/material';
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

  const getResponseTimeColor = (responseTime: number): "success" | "warning" | "error" => {
    if (responseTime < 100) return 'success';
    if (responseTime < 500) return 'warning';
    return 'error';
  };

  if (loading && !performanceData) {
    return (
      <Card>
        <CardHeader
          avatar={<BarChart3 size={24} />}
          title="性能指标"
        />
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', py: 4 }}>
            <CircularProgress size={24} sx={{ mr: 1 }} />
            <Typography variant="body2" color="text.secondary">
              加载性能数据中...
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader
        avatar={<BarChart3 size={24} />}
        title="性能指标"
        action={
          <IconButton
            size="small"
            onClick={loadPerformanceData}
            disabled={loading}
          >
            <RefreshCw size={16} />
          </IconButton>
        }
      />
      <CardContent>
        <Tabs
          value={selectedTab}
          onChange={(e, newValue) => setSelectedTab(newValue)}
        >
          <Tab label="概览" value="overview" />
          <Tab label="服务详情" value="services" />
        </Tabs>

        <Box sx={{ mt: 2 }}>
          {selectedTab === 'overview' && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {/* 关键指标 */}
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(4, 1fr)' }, gap: 2 }}>
                <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'primary.light', borderRadius: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
                    <Clock size={20} color="#1976d2" />
                  </Box>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'primary.main' }}>
                    {performanceData?.summary?.avg_response_time?.toFixed(0) || 0}ms
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    平均响应时间
                  </Typography>
                </Box>
                
                <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'success.light', borderRadius: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
                    <TrendingUp size={20} color="#2e7d32" />
                  </Box>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'success.main' }}>
                    {formatNumber(performanceData?.summary?.total_requests || 0)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    总请求数
                  </Typography>
                </Box>
                
                <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'warning.light', borderRadius: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
                    <AlertCircle size={20} color="#ed6c02" />
                  </Box>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'warning.main' }}>
                    {formatNumber(performanceData?.summary?.total_errors || 0)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    错误总数
                  </Typography>
                </Box>
                
                <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'secondary.light', borderRadius: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
                    <Activity size={20} color="#9c27b0" />
                  </Box>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'secondary.main' }}>
                    {performanceData?.summary?.total_services || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    活跃服务
                  </Typography>
                </Box>
              </Box>

              {/* 错误率 */}
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    错误率
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {getErrorRate().toFixed(2)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={getErrorRate()}
                  color={getErrorRate() > 5 ? 'error' : getErrorRate() > 1 ? 'warning' : 'success'}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              {/* 响应时间状态 */}
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    响应时间状态
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {performanceData?.summary?.avg_response_time || 0}ms
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={Math.min((performanceData?.summary?.avg_response_time || 0) / 10, 100)}
                  color={getResponseTimeColor(performanceData?.summary?.avg_response_time || 0)}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>
            </Box>
          )}
          
          {selectedTab === 'services' && (
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {performanceData?.services && Object.entries(performanceData.services).map(([serviceName, metrics]) => (
                <Box key={serviceName} sx={{ p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="body2" sx={{ fontWeight: 500 }}>
                      {serviceName}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                      <Zap size={16} color="#1976d2" />
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {metrics.avg_response_time?.toFixed(0) || 0}ms
                      </Typography>
                    </Box>
                  </Box>
                  
                  <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 2 }}>
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        请求数
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500 }}>
                        {formatNumber(metrics.request_count || 0)}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        错误数
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500, color: 'error.main' }}>
                        {metrics.error_count || 0}
                      </Typography>
                    </Box>
                    <Box>
                      <Typography variant="caption" color="text.secondary">
                        成功率
                      </Typography>
                      <Typography variant="body2" sx={{ fontWeight: 500, color: 'success.main' }}>
                        {metrics.request_count > 0 
                          ? (((metrics.request_count - metrics.error_count) / metrics.request_count) * 100).toFixed(1)
                          : 100
                        }%
                      </Typography>
                    </Box>
                  </Box>
                </Box>
              ))}
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
}
