/**
 * 系统监控页面
 *
 * 提供全面的系统监控功能，包括：
 * - 系统健康状态
 * - 性能指标监控
 * - 错误统计分析
 * - 数据质量检查
 * - 异常检测
 */

'use client';

import React, { useEffect, useState } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Tabs,
  Tab,
  Chip,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Select,
  MenuItem,
  Box,
  Typography,
  IconButton,
  FormControl,
  InputLabel,
} from '@mui/material';
import { Activity, AlertTriangle, Shield, RefreshCw, Database, Server } from 'lucide-react';
import { DataService } from '../../services/dataService';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import { SystemHealthCard } from '../../components/monitoring/SystemHealthCard';
import { PerformanceMetricsCard } from '../../components/monitoring/PerformanceMetricsCard';
import { MobileErrorCard } from '../../components/mobile/MobileErrorCard';

interface ErrorStatistics {
  time_range_hours: number;
  total_error_types: number;
  total_errors: number;
  error_statistics: Array<{
    error_type: string;
    count: number;
    last_occurrence: string;
    sample_message: string;
  }>;
}

interface Anomalies {
  total_anomalies: number;
  by_severity: {
    high: number;
    medium: number;
    low: number;
  };
  anomalies: Array<{
    type: string;
    severity: string;
    description: string;
    detected_at: string;
    affected_component: string;
  }>;
  detection_time: string;
}

export default function MonitoringPage() {
  const [loading, setLoading] = useState(true);
  const [errorStats, setErrorStats] = useState<ErrorStatistics | null>(null);
  const [anomalies, setAnomalies] = useState<Anomalies | null>(null);
  const [dataQuality, setDataQuality] = useState<any>(null);
  const [selectedTab, setSelectedTab] = useState('overview');
  const [timeRange, setTimeRange] = useState('24');

  const loadMonitoringData = async () => {
    try {
      setLoading(true);
      const [errorsData, anomaliesData, qualityData] = await Promise.all([
        DataService.getErrorStatistics(parseInt(timeRange)),
        DataService.getAnomalies(),
        DataService.getDataQuality(),
      ]);

      setErrorStats(errorsData);
      setAnomalies(anomaliesData);
      setDataQuality(qualityData);
    } catch (error) {
      console.error('加载监控数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMonitoringData();

    // 每分钟自动刷新
    const interval = setInterval(loadMonitoringData, 60000);
    return () => clearInterval(interval);
  }, [timeRange]);

  const getSeverityColor = (severity: string): 'error' | 'warning' | 'primary' | 'default' => {
    switch (severity) {
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      case 'low':
        return 'primary';
      default:
        return 'default';
    }
  };

  const formatTimeAgo = (dateString: string) => {
    const now = new Date();
    const date = new Date(dateString);
    const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / (1000 * 60));

    if (diffInMinutes < 60) {
      return `${diffInMinutes}分钟前`;
    } else if (diffInMinutes < 1440) {
      return `${Math.floor(diffInMinutes / 60)}小时前`;
    } else {
      return `${Math.floor(diffInMinutes / 1440)}天前`;
    }
  };

  if (loading && !errorStats) {
    return <LoadingSpinner text="加载监控数据..." />;
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 页面标题 */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 2,
        }}
      >
        <Box>
          <Typography variant="h4" component="h1" sx={{ fontWeight: 600, mb: 1 }}>
            系统监控
          </Typography>
          <Typography variant="body2" color="text.secondary">
            实时监控系统健康状态、性能指标和异常情况
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>时间范围</InputLabel>
            <Select value={timeRange} label="时间范围" onChange={e => setTimeRange(e.target.value)}>
              <MenuItem value="1">1小时</MenuItem>
              <MenuItem value="6">6小时</MenuItem>
              <MenuItem value="24">24小时</MenuItem>
              <MenuItem value="168">7天</MenuItem>
            </Select>
          </FormControl>
          <IconButton onClick={loadMonitoringData} disabled={loading}>
            <RefreshCw size={16} />
          </IconButton>
        </Box>
      </Box>

      {/* 主要内容 */}
      <Tabs
        value={selectedTab}
        onChange={(e, newValue) => setSelectedTab(newValue)}
        variant="scrollable"
        scrollButtons="auto"
      >
        <Tab value="overview" icon={<Activity size={16} />} iconPosition="start" label="系统概览" />
        <Tab
          value="errors"
          icon={<AlertTriangle size={16} />}
          iconPosition="start"
          label={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <span>错误分析</span>
              {errorStats && errorStats.total_errors > 0 && (
                <Chip label={errorStats.total_errors} color="error" size="small" />
              )}
            </Box>
          }
        />
        <Tab
          value="anomalies"
          icon={<Shield size={16} />}
          iconPosition="start"
          label={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <span>异常检测</span>
              {anomalies && anomalies.total_anomalies > 0 && (
                <Chip label={anomalies.total_anomalies} color="warning" size="small" />
              )}
            </Box>
          }
        />
        <Tab value="quality" icon={<Database size={16} />} iconPosition="start" label="数据质量" />
      </Tabs>

      <Box sx={{ mt: 2 }}>
        {selectedTab === 'overview' && (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* 健康状态和性能指标 */}
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', xl: 'repeat(2, 1fr)' },
                gap: 3,
              }}
            >
              <SystemHealthCard />
              <PerformanceMetricsCard />
            </Box>

            {/* 关键指标概览 */}
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', md: 'repeat(4, 1fr)' },
                gap: 2,
              }}
            >
              <Card>
                <CardContent sx={{ textAlign: 'center', p: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
                    <AlertTriangle size={24} color="#d32f2f" />
                  </Box>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'error.main' }}>
                    {errorStats?.total_errors || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    总错误数
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    过去{timeRange}小时
                  </Typography>
                </CardContent>
              </Card>

              <Card>
                <CardContent sx={{ textAlign: 'center', p: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
                    <Shield size={24} color="#ed6c02" />
                  </Box>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'warning.main' }}>
                    {anomalies?.total_anomalies || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    异常检测
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    高危: {anomalies?.by_severity.high || 0}
                  </Typography>
                </CardContent>
              </Card>

              <Card>
                <CardContent sx={{ textAlign: 'center', p: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
                    <Database size={24} color="#2e7d32" />
                  </Box>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'success.main' }}>
                    {dataQuality?.overall_score || '--'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    数据质量
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    综合评分
                  </Typography>
                </CardContent>
              </Card>

              <Card>
                <CardContent sx={{ textAlign: 'center', p: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'center', mb: 1 }}>
                    <Server size={24} color="#1976d2" />
                  </Box>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'primary.main' }}>
                    99.9%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    系统可用性
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    过去30天
                  </Typography>
                </CardContent>
              </Card>
            </Box>
          </Box>
        )}

        {selectedTab === 'errors' && (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* 错误统计概览 */}
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' },
                gap: 2,
              }}
            >
              <Card>
                <CardContent sx={{ textAlign: 'center', p: 2 }}>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'error.main' }}>
                    {errorStats?.total_errors || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    总错误数
                  </Typography>
                </CardContent>
              </Card>
              <Card>
                <CardContent sx={{ textAlign: 'center', p: 2 }}>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'warning.main' }}>
                    {errorStats?.total_error_types || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    错误类型
                  </Typography>
                </CardContent>
              </Card>
              <Card>
                <CardContent sx={{ textAlign: 'center', p: 2 }}>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'primary.main' }}>
                    {errorStats?.time_range_hours || 0}h
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    统计时间范围
                  </Typography>
                </CardContent>
              </Card>
            </Box>

            {/* 错误详情表格 */}
            <Card>
              <CardHeader title="错误详情" />
              <CardContent>
                {/* 移动端：卡片列表 */}
                <Box sx={{ display: { xs: 'block', md: 'none' } }}>
                  {errorStats?.error_statistics.length === 0 ? (
                    <Box sx={{ textAlign: 'center', py: 4 }}>
                      <Typography variant="body2" color="text.secondary">
                        暂无错误记录
                      </Typography>
                    </Box>
                  ) : (
                    errorStats?.error_statistics.map((error, index) => (
                      <MobileErrorCard key={index} error={error} />
                    ))
                  )}
                </Box>

                {/* 桌面端：表格 */}
                <Box sx={{ display: { xs: 'none', md: 'block' }, overflowX: 'auto' }}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>错误类型</TableCell>
                        <TableCell>发生次数</TableCell>
                        <TableCell>最后发生时间</TableCell>
                        <TableCell>示例消息</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {errorStats?.error_statistics.length === 0 ? (
                        <TableRow>
                          <TableCell colSpan={4} align="center">
                            <Typography variant="body2" color="text.secondary">
                              暂无错误记录
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ) : (
                        errorStats?.error_statistics.map((error, index) => (
                          <TableRow key={index}>
                            <TableCell>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <AlertTriangle size={16} color="#d32f2f" />
                                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                  {error.error_type}
                                </Typography>
                              </Box>
                            </TableCell>
                            <TableCell>
                              <Chip label={error.count} color="error" size="small" />
                            </TableCell>
                            <TableCell>
                              <Box>
                                <Typography variant="body2">
                                  {new Date(error.last_occurrence).toLocaleString()}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                  {formatTimeAgo(error.last_occurrence)}
                                </Typography>
                              </Box>
                            </TableCell>
                            <TableCell>
                              <Typography
                                variant="body2"
                                color="text.secondary"
                                sx={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis' }}
                              >
                                {error.sample_message}
                              </Typography>
                            </TableCell>
                          </TableRow>
                        ))
                      )}
                    </TableBody>
                  </Table>
                </Box>
              </CardContent>
            </Card>
          </Box>
        )}

        {selectedTab === 'anomalies' && (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            {/* 异常统计 */}
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', md: 'repeat(4, 1fr)' },
                gap: 2,
              }}
            >
              <Card>
                <CardContent sx={{ textAlign: 'center', p: 2 }}>
                  <Typography variant="h4" sx={{ fontWeight: 600 }}>
                    {anomalies?.total_anomalies || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    总异常数
                  </Typography>
                </CardContent>
              </Card>
              <Card>
                <CardContent sx={{ textAlign: 'center', p: 2 }}>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'error.main' }}>
                    {anomalies?.by_severity.high || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    高危异常
                  </Typography>
                </CardContent>
              </Card>
              <Card>
                <CardContent sx={{ textAlign: 'center', p: 2 }}>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'warning.main' }}>
                    {anomalies?.by_severity.medium || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    中危异常
                  </Typography>
                </CardContent>
              </Card>
              <Card>
                <CardContent sx={{ textAlign: 'center', p: 2 }}>
                  <Typography variant="h4" sx={{ fontWeight: 600, color: 'primary.main' }}>
                    {anomalies?.by_severity.low || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    低危异常
                  </Typography>
                </CardContent>
              </Card>
            </Box>

            {/* 异常详情 */}
            <Card>
              <CardHeader
                title="异常详情"
                action={
                  anomalies?.detection_time && (
                    <Typography variant="body2" color="text.secondary">
                      检测时间: {new Date(anomalies.detection_time).toLocaleString()}
                    </Typography>
                  )
                }
              />
              <CardContent>
                <Box sx={{ overflowX: 'auto' }}>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>异常类型</TableCell>
                        <TableCell>严重程度</TableCell>
                        <TableCell>影响组件</TableCell>
                        <TableCell>检测时间</TableCell>
                        <TableCell>描述</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {anomalies?.anomalies.length === 0 ? (
                        <TableRow>
                          <TableCell colSpan={5} align="center">
                            <Typography variant="body2" color="text.secondary">
                              暂无异常检测
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ) : (
                        anomalies?.anomalies.map((anomaly, index) => (
                          <TableRow key={index}>
                            <TableCell>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <Shield size={16} color="#ed6c02" />
                                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                  {anomaly.type}
                                </Typography>
                              </Box>
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={
                                  anomaly.severity === 'high'
                                    ? '高危'
                                    : anomaly.severity === 'medium'
                                      ? '中危'
                                      : '低危'
                                }
                                color={getSeverityColor(anomaly.severity)}
                                size="small"
                              />
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                                {anomaly.affected_component}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Box>
                                <Typography variant="body2">
                                  {new Date(anomaly.detected_at).toLocaleString()}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                  {formatTimeAgo(anomaly.detected_at)}
                                </Typography>
                              </Box>
                            </TableCell>
                            <TableCell>
                              <Typography
                                variant="body2"
                                color="text.secondary"
                                sx={{ maxWidth: 300 }}
                              >
                                {anomaly.description}
                              </Typography>
                            </TableCell>
                          </TableRow>
                        ))
                      )}
                    </TableBody>
                  </Table>
                </Box>
              </CardContent>
            </Card>
          </Box>
        )}

        {selectedTab === 'quality' && (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
            <Card>
              <CardHeader title="数据质量检查" />
              <CardContent>
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Database size={48} color="#ccc" style={{ margin: '0 auto 16px' }} />
                  <Typography variant="body1" color="text.secondary">
                    数据质量检查功能开发中...
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    将包括数据完整性、准确性、一致性等指标
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Box>
        )}
      </Box>
    </Box>
  );
}
