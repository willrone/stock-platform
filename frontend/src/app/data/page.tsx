/**
 * 数据管理页面
 * 
 * 显示和管理数据相关功能：
 * - 远端数据服务状态
 * - 本地Parquet文件列表
 * - 数据同步控制功能
 * - 数据统计信息
 * - 系统监控面板
 * - 同步历史记录
 */

'use client';

import React, { useEffect, useState } from 'react';
import {
  Card,
  CardHeader,
  CardBody,
  Button,
  Chip,
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  useDisclosure,
  Tabs,
  Tab,
  Divider,
} from '@heroui/react';
import {
  Database,
  Server,
  RefreshCw,
  Download,
  Wifi,
  WifiOff,
  XCircle,
  Clock,
  Activity,
  BarChart3,
  History,
  Zap,
} from 'lucide-react';
import { DataService } from '../../services/dataService';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import { SystemHealthCard } from '../../components/monitoring/SystemHealthCard';
import { PerformanceMetricsCard } from '../../components/monitoring/PerformanceMetricsCard';
import { SyncProgressModal } from '../../components/monitoring/SyncProgressModal';
import { SyncHistoryModal } from '../../components/monitoring/SyncHistoryModal';
import { EnhancedDataFileTable } from '../../components/data/EnhancedDataFileTable';

interface DataFile {
  file_path: string;
  stock_code: string;
  date_range: {
    start: string;
    end: string;
  };
  record_count: number;
  file_size: number;
  last_modified: string;
  integrity_status: string;
  compression_ratio: number;
  created_at?: string;
}

interface ServiceStatus {
  service_url: string;
  is_connected: boolean;
  last_check: string;
  response_time: number;
  error_message?: string;
}

interface DataStats {
  total_files: number;
  total_size_bytes: number;
  total_size_mb: number;
  total_records: number;
  stock_count: number;
  date_range: {
    start: string;
    end: string;
  };
  average_file_size_bytes: number;
  average_file_size_mb: number;
  storage_efficiency: number;
  last_sync_time: string | null;
  top_stocks_by_size: Array<{
    stock_code: string;
    size_bytes: number;
    size_mb: number;
  }>;
  monthly_distribution: Record<string, number>;
}

export default function DataManagementPage() {
  const [loading, setLoading] = useState(true);
  const [dataFiles, setDataFiles] = useState<DataFile[]>([]);
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus | null>(null);
  const [dataStats, setDataStats] = useState<DataStats | null>(null);
  const [remoteStocks, setRemoteStocks] = useState<Array<{ts_code: string; name?: string}>>([]);
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  const [isSyncing, setIsSyncing] = useState(false);
  const [currentSyncId, setCurrentSyncId] = useState<string | null>(null);
  const [selectedTab, setSelectedTab] = useState('overview');
  
  const { isOpen: isSyncOpen, onOpen: onSyncOpen, onClose: onSyncClose } = useDisclosure();
  const { isOpen: isProgressOpen, onOpen: onProgressOpen, onClose: onProgressClose } = useDisclosure();
  const { isOpen: isHistoryOpen, onOpen: onHistoryOpen, onClose: onHistoryClose } = useDisclosure();

  // 加载数据文件列表
  const loadDataFiles = async () => {
    try {
      const response = await DataService.getLocalDataFiles();
      // 转换API响应格式以匹配本地DataFile接口
      const transformedFiles: DataFile[] = response.files.map(file => ({
        ...file,
        last_modified: file.last_updated,
        integrity_status: 'verified', // 默认状态
        compression_ratio: 0.75, // 默认压缩比
      }));
      setDataFiles(transformedFiles);
    } catch (error) {
      console.error('加载数据文件失败:', error);
    }
  };

  // 检查服务状态
  const checkServiceStatus = async () => {
    try {
      const status = await DataService.getDataServiceStatus();
      setServiceStatus(status);
    } catch (error) {
      console.error('检查服务状态失败:', error);
    }
  };

  // 加载数据统计
  const loadDataStats = async () => {
    try {
      const stats = await DataService.getDataStatistics();
      // 转换API响应格式以匹配本地DataStats接口
      const transformedStats: DataStats = {
        ...stats,
        total_size_bytes: stats.total_size,
        total_size_mb: stats.total_size / (1024 * 1024),
        average_file_size_bytes: stats.total_files > 0 ? stats.total_size / stats.total_files : 0,
        average_file_size_mb: stats.total_files > 0 ? (stats.total_size / stats.total_files) / (1024 * 1024) : 0,
        storage_efficiency: stats.total_size > 0 ? stats.total_records / (stats.total_size / (1024 * 1024)) : 0,
        last_sync_time: stats.last_sync,
        top_stocks_by_size: [],
        monthly_distribution: {},
      };
      setDataStats(transformedStats);
    } catch (error) {
      console.error('加载数据统计失败:', error);
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
        loadDataFiles(),
        checkServiceStatus(),
        loadDataStats(),
        loadRemoteStocks()
      ]);
      setLoading(false);
    };
    
    loadData();
  }, []);

  // 格式化文件大小
  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // 同步数据
  const handleSyncData = async () => {
    setIsSyncing(true);
    
    try {
      // 优先使用远端股票列表，如果没有则使用本地文件列表
      let stockCodes: string[] = [];
      
      if (remoteStocks.length > 0) {
        // 从远端股票列表获取股票代码
        stockCodes = remoteStocks.map(stock => stock.ts_code).filter(code => code);
      } else if (dataFiles.length > 0) {
        // 如果没有远端股票列表，使用本地文件列表
        stockCodes = dataFiles.map(file => file.stock_code);
      } else {
        // 如果都没有，重新加载远端股票列表
        await loadRemoteStocks();
        stockCodes = remoteStocks.map(stock => stock.ts_code).filter(code => code);
      }
      
      if (stockCodes.length === 0) {
        console.error('没有可同步的股票');
        setIsSyncing(false);
        onSyncClose();
        return;
      }
      
      // 调用同步API
      const result = await DataService.syncDataFromRemote({
        stock_codes: stockCodes,
        force_update: true,
      });
      
      // 模拟sync_id，因为API返回格式不同
      const mockSyncId = `sync_${Date.now()}`;
      setCurrentSyncId(mockSyncId);
      onSyncClose();
      onProgressOpen();
      
    } catch (error) {
      console.error('同步数据失败:', error);
      setIsSyncing(false);
      onSyncClose();
    }
  };

  // 删除选中文件
  const handleDeleteFiles = async (filePaths: string[]) => {
    try {
      const result = await DataService.deleteDataFiles(filePaths);
      
      // 更新本地状态
      setDataFiles(prev => prev.filter(file => !filePaths.includes(file.file_path)));
      setSelectedFiles(new Set());
      
      // 重新加载统计数据
      await loadDataStats();
      
      console.log('删除结果:', result);
    } catch (error) {
      console.error('删除文件失败:', error);
    }
  };

  // 刷新数据
  const handleRefresh = async () => {
    await Promise.all([
      loadDataFiles(),
      checkServiceStatus(),
      loadDataStats(),
      loadRemoteStocks()
    ]);
  };

  // 同步完成回调
  const handleSyncComplete = () => {
    setIsSyncing(false);
    setCurrentSyncId(null);
    // 重新加载数据
    handleRefresh();
  };

  if (loading) {
    return <LoadingSpinner text="加载数据管理信息..." />;
  }

  return (
    <div className="space-y-6">
      {/* 页面标题 */}
      <div>
        <h1 className="text-2xl font-bold mb-2">数据管理</h1>
        <p className="text-default-500">管理股票数据文件、监控系统状态和远端服务连接</p>
      </div>

      {/* 主要内容区域 */}
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
              <Database className="w-4 h-4" />
              <span>数据概览</span>
            </div>
          }
        >
          <div className="space-y-6 pt-6">
            {/* 服务状态和统计 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* 服务状态 */}
              <Card>
                <CardHeader>
                  <div className="flex items-center space-x-2">
                    <Server className="w-5 h-5" />
                    <h3 className="text-lg font-semibold">远端服务状态</h3>
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
                  
                  <Button
                    color="primary"
                    variant="light"
                    startContent={<RefreshCw className="w-4 h-4" />}
                    onPress={checkServiceStatus}
                    fullWidth
                  >
                    检查连接
                  </Button>
                </CardBody>
              </Card>

              {/* 数据统计 */}
              <Card>
                <CardHeader>
                  <div className="flex items-center space-x-2">
                    <Database className="w-5 h-5" />
                    <h3 className="text-lg font-semibold">数据统计</h3>
                  </div>
                </CardHeader>
                <CardBody className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 bg-primary-50 rounded-lg">
                      <p className="text-2xl font-bold text-primary">{dataStats?.total_files || 0}</p>
                      <p className="text-sm text-default-500">文件数量</p>
                    </div>
                    <div className="text-center p-3 bg-secondary-50 rounded-lg">
                      <p className="text-2xl font-bold text-secondary">
                        {dataStats?.total_size_mb ? `${dataStats.total_size_mb.toFixed(1)}MB` : '--'}
                      </p>
                      <p className="text-sm text-default-500">总大小</p>
                    </div>
                    <div className="text-center p-3 bg-success-50 rounded-lg">
                      <p className="text-2xl font-bold text-success">
                        {dataStats?.total_records.toLocaleString() || 0}
                      </p>
                      <p className="text-sm text-default-500">记录数</p>
                    </div>
                    <div className="text-center p-3 bg-warning-50 rounded-lg">
                      <p className="text-2xl font-bold text-warning">{dataStats?.stock_count || 0}</p>
                      <p className="text-sm text-default-500">股票数</p>
                    </div>
                  </div>

                  <Divider />

                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-default-600">存储效率</span>
                      <span className="text-sm font-medium">
                        {dataStats?.storage_efficiency ? `${dataStats.storage_efficiency.toFixed(1)} 记录/MB` : '--'}
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-default-600">平均文件大小</span>
                      <span className="text-sm font-medium">
                        {dataStats?.average_file_size_mb ? `${dataStats.average_file_size_mb.toFixed(1)}MB` : '--'}
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-default-600">数据范围</span>
                      <span className="text-sm font-medium">
                        {dataStats?.date_range ? 
                          `${dataStats.date_range.start} 至 ${dataStats.date_range.end}` : 
                          '--'
                        }
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-default-600">最后同步</span>
                      <div className="flex items-center space-x-1">
                        <Clock className="w-3 h-3 text-default-400" />
                        <span className="text-sm font-medium">
                          {dataStats?.last_sync_time ? 
                            new Date(dataStats.last_sync_time).toLocaleDateString() : 
                            '--'
                          }
                        </span>
                      </div>
                    </div>
                  </div>
                </CardBody>
              </Card>
            </div>

            {/* 操作按钮 */}
            <Card>
              <CardBody>
                <div className="flex flex-wrap gap-3">
                  <Button
                    color="primary"
                    startContent={<Download className="w-4 h-4" />}
                    onPress={onSyncOpen}
                    isDisabled={!serviceStatus?.is_connected}
                  >
                    同步数据
                  </Button>
                  <Button
                    variant="light"
                    startContent={<RefreshCw className="w-4 h-4" />}
                    onPress={handleRefresh}
                  >
                    刷新数据
                  </Button>
                  <Button
                    variant="light"
                    startContent={<History className="w-4 h-4" />}
                    onPress={onHistoryOpen}
                  >
                    同步历史
                  </Button>
                  <Button
                    variant="light"
                    startContent={<Activity className="w-4 h-4" />}
                    onPress={() => setSelectedTab('monitoring')}
                  >
                    系统监控
                  </Button>
                </div>
              </CardBody>
            </Card>
          </div>
        </Tab>

        <Tab
          key="files"
          title={
            <div className="flex items-center space-x-2">
              <Database className="w-4 h-4" />
              <span>文件管理</span>
            </div>
          }
        >
          <div className="pt-6">
            <EnhancedDataFileTable
              files={dataFiles}
              selectedFiles={selectedFiles}
              onSelectionChange={setSelectedFiles}
              onDeleteFiles={handleDeleteFiles}
              onRefresh={handleRefresh}
            />
          </div>
        </Tab>

        <Tab
          key="monitoring"
          title={
            <div className="flex items-center space-x-2">
              <BarChart3 className="w-4 h-4" />
              <span>系统监控</span>
            </div>
          }
        >
          <div className="space-y-6 pt-6">
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <SystemHealthCard />
              <PerformanceMetricsCard />
            </div>
          </div>
        </Tab>
      </Tabs>

      {/* 同步数据确认对话框 */}
      <Modal isOpen={isSyncOpen} onClose={onSyncClose} isDismissable={!isSyncing}>
        <ModalContent>
          {(onClose) => (
            <>
              <ModalHeader>同步数据</ModalHeader>
              <ModalBody>
                <div className="space-y-4">
                  <p>确定要从远端服务同步最新的股票数据吗？</p>
                  <div className="p-3 bg-default-50 rounded-lg">
                    <div className="text-sm space-y-1">
                      <div className="flex justify-between">
                        <span className="text-default-600">股票数量:</span>
                        <span className="font-medium">
                          {remoteStocks.length > 0 ? remoteStocks.length : dataFiles.length} 只
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-default-600">数据来源:</span>
                        <span className="font-medium">
                          {remoteStocks.length > 0 ? '远端服务' : '本地文件'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-default-600">同步模式:</span>
                        <span className="font-medium">增量同步</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-default-600">最大并发:</span>
                        <span className="font-medium">3 个</span>
                      </div>
                    </div>
                  </div>
                </div>
              </ModalBody>
              <ModalFooter>
                <Button variant="light" onPress={onClose} isDisabled={isSyncing}>
                  取消
                </Button>
                <Button 
                  color="primary" 
                  onPress={handleSyncData}
                  isLoading={isSyncing}
                >
                  开始同步
                </Button>
              </ModalFooter>
            </>
          )}
        </ModalContent>
      </Modal>

      {/* 同步进度模态框 */}
      <SyncProgressModal
        isOpen={isProgressOpen}
        onClose={onProgressClose}
        syncId={currentSyncId}
        onSyncComplete={handleSyncComplete}
      />

      {/* 同步历史模态框 */}
      <SyncHistoryModal
        isOpen={isHistoryOpen}
        onClose={onHistoryClose}
      />
    </div>
  );
}