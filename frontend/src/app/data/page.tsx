/**
 * 数据管理页面 - 数据概览
 * 
 * 显示数据相关功能：
 * - 远端数据服务状态
 * - 远端股票列表
 */

'use client';

import React, { useEffect, useState } from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Button,
  Chip,
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
} from '@heroui/react';
import {
  Server,
  RefreshCw,
  Wifi,
  WifiOff,
  XCircle,
  Zap,
  Database,
} from 'lucide-react';
import { DataService } from '../../services/dataService';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';

interface ServiceStatus {
  service_url: string;
  is_connected: boolean;
  last_check: string;
  response_time: number;
  error_message?: string;
}

interface RemoteStock {
  ts_code: string;
  name?: string;
  data_range?: {
    start_date: string;
    end_date: string;
    total_days?: number;
  };
  last_update?: string;
  status?: string;
}

export default function DataManagementPage() {
  const [loading, setLoading] = useState(true);
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus | null>(null);
  const [remoteStocks, setRemoteStocks] = useState<RemoteStock[]>([]);

  // 检查服务状态
  const checkServiceStatus = async () => {
    try {
      const status = await DataService.getDataServiceStatus();
      setServiceStatus(status);
    } catch (error) {
      console.error('检查服务状态失败:', error);
    }
  };

  // 加载远端股票列表
  const loadRemoteStocks = async () => {
    try {
      const result = await DataService.getRemoteStockList();
      setRemoteStocks(result.stocks || []);
    } catch (error) {
      console.error('加载远端股票列表失败:', error);
      setRemoteStocks([]);
    }
  };

  // 初始化加载
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([
        checkServiceStatus(),
        loadRemoteStocks()
      ]);
      setLoading(false);
    };
    
    loadData();
  }, []);

  // 刷新数据
  const handleRefresh = async () => {
    setLoading(true);
    await Promise.all([
      checkServiceStatus(),
      loadRemoteStocks()
    ]);
    setLoading(false);
  };

  if (loading) {
    return <LoadingSpinner text="加载数据信息..." />;
  }

  return (
    <div className="space-y-6">
      {/* 页面标题 */}
      <div>
        <h1 className="text-2xl font-bold mb-2">数据管理</h1>
        <p className="text-default-500">查看远端数据服务状态和股票列表</p>
      </div>

      {/* 服务状态 */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between w-full">
            <div className="flex items-center space-x-2">
              <Server className="w-5 h-5" />
              <h3 className="text-lg font-semibold">远端服务状态</h3>
            </div>
            <Button
              color="primary"
              variant="light"
              size="sm"
              startContent={<RefreshCw className="w-4 h-4" />}
              onPress={checkServiceStatus}
            >
              刷新
            </Button>
          </div>
        </CardHeader>
        <CardBody className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-default-600">连接状态</span>
            <div className="flex items-center space-x-2">
              {serviceStatus?.is_connected ? (
                <>
                  <Wifi className="w-4 h-4 text-success" />
                  <Chip color="success" variant="flat" size="sm">已连接</Chip>
                </>
              ) : (
                <>
                  <WifiOff className="w-4 h-4 text-danger" />
                  <Chip color="danger" variant="flat" size="sm">未连接</Chip>
                </>
              )}
            </div>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-default-600">服务地址</span>
            <span className="text-sm font-mono">{serviceStatus?.service_url || '--'}</span>
          </div>
          
          {serviceStatus?.is_connected && (
            <div className="flex items-center justify-between">
              <span className="text-default-600">响应时间</span>
              <div className="flex items-center space-x-1">
                <Zap className="w-3 h-3 text-warning" />
                <span className="font-medium">{serviceStatus.response_time}ms</span>
              </div>
            </div>
          )}
          
          <div className="flex items-center justify-between">
            <span className="text-default-600">最后检查</span>
            <span className="text-sm text-default-500">
              {serviceStatus?.last_check ? new Date(serviceStatus.last_check).toLocaleString() : '--'}
            </span>
          </div>
          
          {serviceStatus?.error_message && (
            <div className="bg-danger-50 border border-danger-200 rounded-lg p-3">
              <div className="flex items-start space-x-2">
                <XCircle className="w-4 h-4 text-danger mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-danger">连接错误</p>
                  <p className="text-xs text-danger-600">{serviceStatus.error_message}</p>
                </div>
              </div>
            </div>
          )}
        </CardBody>
      </Card>

      {/* 远端股票列表 */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between w-full">
            <div className="flex items-center space-x-2">
              <Database className="w-5 h-5" />
              <h3 className="text-lg font-semibold">远端股票列表</h3>
              <Chip size="sm" variant="flat">{remoteStocks.length} 只股票</Chip>
            </div>
            <Button
              color="primary"
              variant="light"
              size="sm"
              startContent={<RefreshCw className="w-4 h-4" />}
              onPress={loadRemoteStocks}
            >
              刷新
            </Button>
          </div>
        </CardHeader>
        <CardBody>
          {remoteStocks.length === 0 ? (
            <div className="text-center py-8 text-default-500">
              <p>暂无股票数据</p>
              <p className="text-sm mt-2">请检查远端服务连接状态</p>
            </div>
          ) : (
            <Table aria-label="远端股票列表">
              <TableHeader>
                <TableColumn>股票代码</TableColumn>
                <TableColumn>股票名称</TableColumn>
                <TableColumn>数据范围</TableColumn>
                <TableColumn>最后更新</TableColumn>
                <TableColumn>状态</TableColumn>
              </TableHeader>
              <TableBody>
                {remoteStocks.map((stock) => (
                  <TableRow key={stock.ts_code}>
                    <TableCell>
                      <span className="font-mono font-medium">{stock.ts_code}</span>
                    </TableCell>
                    <TableCell>{stock.name || '--'}</TableCell>
                    <TableCell>
                      {stock.data_range ? (
                        <div className="text-sm">
                          <div>{stock.data_range.start_date} 至 {stock.data_range.end_date}</div>
                          {stock.data_range.total_days && (
                            <div className="text-default-500 text-xs">
                              {stock.data_range.total_days} 天
                            </div>
                          )}
                        </div>
                      ) : (
                        '--'
                      )}
                    </TableCell>
                    <TableCell>
                      {stock.last_update ? new Date(stock.last_update).toLocaleDateString() : '--'}
                    </TableCell>
                    <TableCell>
                      {stock.status === 'complete' ? (
                        <Chip color="success" variant="flat" size="sm">完整</Chip>
                      ) : stock.status === 'incomplete' ? (
                        <Chip color="warning" variant="flat" size="sm">不完整</Chip>
                      ) : stock.status === 'missing' ? (
                        <Chip color="danger" variant="flat" size="sm">缺失</Chip>
                      ) : (
                        <Chip variant="flat" size="sm">未知</Chip>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
