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
  CardHeader,
  CardBody,
  Button,
  Tabs,
  Tab,
  Chip,
  Progress,
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Select,
  SelectItem,
} from '@heroui/react';
import {
  Activity,
  AlertTriangle,
  Shield,
  RefreshCw,
  Database,
  Server,
} from 'lucide-react';
import { DataService } from '../../services/dataService';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import { SystemHealthCard } from '../../components/monitoring/SystemHealthCard';
import { PerformanceMetricsCard } from '../../components/monitoring/PerformanceMetricsCard';

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

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high':
        return 'danger';
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
    <div className="space-y-6">
      {/* 页面标题 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold mb-2">系统监控</h1>
          <p className="text-default-500">实时监控系统健康状态、性能指标和异常情况</p>
        </div>
        <div className="flex items-center space-x-2">
          <Select
            label="时间范围"
            selectedKeys={[timeRange]}
            onSelectionChange={(keys) => setTimeRange(Array.from(keys)[0] as string)}
            className="w-32"
            size="sm"
          >
            <SelectItem key="1">1小时</SelectItem>
            <SelectItem key="6">6小时</SelectItem>
            <SelectItem key="24">24小时</SelectItem>
            <SelectItem key="168">7天</SelectItem>
          </Select>
          <Button
            isIconOnly
            variant="light"
            onPress={loadMonitoringData}
            isLoading={loading}
          >
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* 主要内容 */}
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
        <Tab
          key="overview"
          title={
            <div className="flex items-center space-x-2">
              <Activity className="w-4 h-4" />
              <span>系统概览</span>
            </div>
          }
        >
          <div className="space-y-6 pt-6">
            {/* 健康状态和性能指标 */}
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <SystemHealthCard />
              <PerformanceMetricsCard />
            </div>

            {/* 关键指标概览 */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Card>
                <CardBody className="text-center p-4">
                  <div className="flex items-center justify-center mb-2">
                    <AlertTriangle className="w-6 h-6 text-danger" />
                  </div>
                  <p className="text-2xl font-bold text-danger">
                    {errorStats?.total_errors || 0}
                  </p>
                  <p className="text-sm text-default-500">总错误数</p>
                  <p className="text-xs text-default-400">
                    过去{timeRange}小时
                  </p>
                </CardBody>
              </Card>

              <Card>
                <CardBody className="text-center p-4">
                  <div className="flex items-center justify-center mb-2">
                    <Shield className="w-6 h-6 text-warning" />
                  </div>
                  <p className="text-2xl font-bold text-warning">
                    {anomalies?.total_anomalies || 0}
                  </p>
                  <p className="text-sm text-default-500">异常检测</p>
                  <p className="text-xs text-default-400">
                    高危: {anomalies?.by_severity.high || 0}
                  </p>
                </CardBody>
              </Card>

              <Card>
                <CardBody className="text-center p-4">
                  <div className="flex items-center justify-center mb-2">
                    <Database className="w-6 h-6 text-success" />
                  </div>
                  <p className="text-2xl font-bold text-success">
                    {dataQuality?.overall_score || '--'}
                  </p>
                  <p className="text-sm text-default-500">数据质量</p>
                  <p className="text-xs text-default-400">
                    综合评分
                  </p>
                </CardBody>
              </Card>

              <Card>
                <CardBody className="text-center p-4">
                  <div className="flex items-center justify-center mb-2">
                    <Server className="w-6 h-6 text-primary" />
                  </div>
                  <p className="text-2xl font-bold text-primary">
                    99.9%
                  </p>
                  <p className="text-sm text-default-500">系统可用性</p>
                  <p className="text-xs text-default-400">
                    过去30天
                  </p>
                </CardBody>
              </Card>
            </div>
          </div>
        </Tab>

        <Tab
          key="errors"
          title={
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4" />
              <span>错误分析</span>
              {errorStats && errorStats.total_errors > 0 && (
                <Chip color="danger" size="sm" variant="flat">
                  {errorStats.total_errors}
                </Chip>
              )}
            </div>
          }
        >
          <div className="space-y-6 pt-6">
            {/* 错误统计概览 */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardBody className="text-center p-4">
                  <p className="text-2xl font-bold text-danger">
                    {errorStats?.total_errors || 0}
                  </p>
                  <p className="text-sm text-default-500">总错误数</p>
                </CardBody>
              </Card>
              <Card>
                <CardBody className="text-center p-4">
                  <p className="text-2xl font-bold text-warning">
                    {errorStats?.total_error_types || 0}
                  </p>
                  <p className="text-sm text-default-500">错误类型</p>
                </CardBody>
              </Card>
              <Card>
                <CardBody className="text-center p-4">
                  <p className="text-2xl font-bold text-primary">
                    {errorStats?.time_range_hours || 0}h
                  </p>
                  <p className="text-sm text-default-500">统计时间范围</p>
                </CardBody>
              </Card>
            </div>

            {/* 错误详情表格 */}
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">错误详情</h3>
              </CardHeader>
              <CardBody>
                <Table aria-label="错误统计表格">
                  <TableHeader>
                    <TableColumn>错误类型</TableColumn>
                    <TableColumn>发生次数</TableColumn>
                    <TableColumn>最后发生时间</TableColumn>
                    <TableColumn>示例消息</TableColumn>
                  </TableHeader>
                  <TableBody
                    emptyContent="暂无错误记录"
                  >
                    {errorStats?.error_statistics.map((error, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          <div className="flex items-center space-x-2">
                            <AlertTriangle className="w-4 h-4 text-danger" />
                            <span className="font-medium">{error.error_type}</span>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Chip color="danger" variant="flat" size="sm">
                            {error.count}
                          </Chip>
                        </TableCell>
                        <TableCell>
                          <div className="text-sm">
                            <div>{new Date(error.last_occurrence).toLocaleString()}</div>
                            <div className="text-default-500">
                              {formatTimeAgo(error.last_occurrence)}
                            </div>
                          </div>
                        </TableCell>
                        <TableCell>
                          <div className="text-sm text-default-600 max-w-xs truncate">
                            {error.sample_message}
                          </div>
                        </TableCell>
                      </TableRow>
                    )) || []}
                  </TableBody>
                </Table>
              </CardBody>
            </Card>
          </div>
        </Tab>

        <Tab
          key="anomalies"
          title={
            <div className="flex items-center space-x-2">
              <Shield className="w-4 h-4" />
              <span>异常检测</span>
              {anomalies && anomalies.total_anomalies > 0 && (
                <Chip color="warning" size="sm" variant="flat">
                  {anomalies.total_anomalies}
                </Chip>
              )}
            </div>
          }
        >
          <div className="space-y-6 pt-6">
            {/* 异常统计 */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Card>
                <CardBody className="text-center p-4">
                  <p className="text-2xl font-bold">
                    {anomalies?.total_anomalies || 0}
                  </p>
                  <p className="text-sm text-default-500">总异常数</p>
                </CardBody>
              </Card>
              <Card>
                <CardBody className="text-center p-4">
                  <p className="text-2xl font-bold text-danger">
                    {anomalies?.by_severity.high || 0}
                  </p>
                  <p className="text-sm text-default-500">高危异常</p>
                </CardBody>
              </Card>
              <Card>
                <CardBody className="text-center p-4">
                  <p className="text-2xl font-bold text-warning">
                    {anomalies?.by_severity.medium || 0}
                  </p>
                  <p className="text-sm text-default-500">中危异常</p>
                </CardBody>
              </Card>
              <Card>
                <CardBody className="text-center p-4">
                  <p className="text-2xl font-bold text-primary">
                    {anomalies?.by_severity.low || 0}
                  </p>
                  <p className="text-sm text-default-500">低危异常</p>
                </CardBody>
              </Card>
            </div>

            {/* 异常详情 */}
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold">异常详情</h3>
                  {anomalies?.detection_time && (
                    <p className="text-sm text-default-500">
                      检测时间: {new Date(anomalies.detection_time).toLocaleString()}
                    </p>
                  )}
                </div>
              </CardHeader>
              <CardBody>
                <Table aria-label="异常检测表格">
                  <TableHeader>
                    <TableColumn>异常类型</TableColumn>
                    <TableColumn>严重程度</TableColumn>
                    <TableColumn>影响组件</TableColumn>
                    <TableColumn>检测时间</TableColumn>
                    <TableColumn>描述</TableColumn>
                  </TableHeader>
                  <TableBody
                    emptyContent="暂无异常检测"
                  >
                    {anomalies?.anomalies.map((anomaly, index) => (
                      <TableRow key={index}>
                        <TableCell>
                          <div className="flex items-center space-x-2">
                            <Shield className="w-4 h-4 text-warning" />
                            <span className="font-medium">{anomaly.type}</span>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Chip
                            color={getSeverityColor(anomaly.severity) as any}
                            variant="flat"
                            size="sm"
                          >
                            {anomaly.severity === 'high' ? '高危' :
                             anomaly.severity === 'medium' ? '中危' : '低危'}
                          </Chip>
                        </TableCell>
                        <TableCell>
                          <span className="font-medium">{anomaly.affected_component}</span>
                        </TableCell>
                        <TableCell>
                          <div className="text-sm">
                            <div>{new Date(anomaly.detected_at).toLocaleString()}</div>
                            <div className="text-default-500">
                              {formatTimeAgo(anomaly.detected_at)}
                            </div>
                          </div>
                        </TableCell>
                        <TableCell>
                          <div className="text-sm text-default-600 max-w-xs">
                            {anomaly.description}
                          </div>
                        </TableCell>
                      </TableRow>
                    )) || []}
                  </TableBody>
                </Table>
              </CardBody>
            </Card>
          </div>
        </Tab>

        <Tab
          key="quality"
          title={
            <div className="flex items-center space-x-2">
              <Database className="w-4 h-4" />
              <span>数据质量</span>
            </div>
          }
        >
          <div className="space-y-6 pt-6">
            <Card>
              <CardHeader>
                <h3 className="text-lg font-semibold">数据质量检查</h3>
              </CardHeader>
              <CardBody>
                <div className="text-center py-8">
                  <Database className="w-12 h-12 text-default-300 mx-auto mb-4" />
                  <p className="text-default-500">数据质量检查功能开发中...</p>
                  <p className="text-sm text-default-400 mt-2">
                    将包括数据完整性、准确性、一致性等指标
                  </p>
                </div>
              </CardBody>
            </Card>
          </div>
        </Tab>
      </Tabs>
    </div>
  );
}