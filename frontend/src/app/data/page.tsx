'use client';

/**
 * 数据管理页面 - 数据概览
 *
 * 显示数据相关功能：
 * - 远端数据服务状态
 * - 远端股票列表
 */

import React, { useEffect, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  Chip,
  Divider,
  Stack,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Tabs,
  TextField,
  Typography,
} from '@mui/material';
import { Server, RefreshCw, Wifi, WifiOff, XCircle, Zap, Database, Download } from 'lucide-react';
import {
  DataService,
  type DataServiceSummary,
  type RemoteDailyDataResponse,
} from '../../services/dataService';
import { LoadingSpinner } from '../../components/common/LoadingSpinner';
import { wsService } from '../../services/websocket';
import { MobileStockCard } from '../../components/mobile/MobileStockCard';

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

interface LocalStock {
  ts_code: string;
  name?: string;
  data_range?: {
    start_date: string;
    end_date: string;
    total_days?: number;
  };
  file_count?: number;
  total_size?: number;
  record_count?: number;
}

interface DailyQueryForm {
  stockCode: string;
  startDate: string;
  endDate: string;
}

export default function DataManagementPage() {
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [precomputing, setPrecomputing] = useState(false);
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus | null>(null);
  const [dataSummary, setDataSummary] = useState<DataServiceSummary | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [dailyQuery, setDailyQuery] = useState<DailyQueryForm>({
    stockCode: '000001',
    startDate: '2023-06-01',
    endDate: '2023-06-30',
  });
  const [dailyLoading, setDailyLoading] = useState(false);
  const [dailyResult, setDailyResult] = useState<RemoteDailyDataResponse | null>(null);
  const [dailyError, setDailyError] = useState<string | null>(null);
  const [remoteStocks, setRemoteStocks] = useState<RemoteStock[]>([]);
  const [localStocks, setLocalStocks] = useState<LocalStock[]>([]);
  const [activeTab, setActiveTab] = useState<string>('remote');
  const [syncResult, setSyncResult] = useState<{
    success: boolean;
    message: string;
    synced_files?: number;
    total_files?: number;
    total_size_mb?: number;
  } | null>(null);
  const [precomputeTask, setPrecomputeTask] = useState<{
    task_id: string;
    progress: number;
    status: string;
    message?: string;
  } | null>(null);

  // 检查服务状态
  const checkServiceStatus = async () => {
    try {
      const status = await DataService.getDataServiceStatus();
      setServiceStatus(status);
    } catch (error) {
      console.error('检查服务状态失败:', error);
    }
  };

  // 加载数据服务汇总
  const loadDataServiceSummary = async () => {
    if (!serviceStatus?.is_connected) {
      setDataSummary(null);
      return;
    }

    try {
      setSummaryLoading(true);
      const summary = await DataService.getRemoteDataSummary();
      setDataSummary(summary);
    } catch (error) {
      console.error('加载数据服务汇总失败:', error);
      setDataSummary(null);
    } finally {
      setSummaryLoading(false);
    }
  };

  // 查询数据服务日线数据
  const handleQueryDailyData = async () => {
    if (!serviceStatus?.is_connected || dailyLoading) {
      return;
    }

    setDailyLoading(true);
    setDailyError(null);
    setDailyResult(null);

    try {
      const result = await DataService.getRemoteStockDailyData(dailyQuery);
      setDailyResult(result);
      if (!result.success) {
        setDailyError(result.error || '数据服务未返回可用数据');
      }
    } catch (error) {
      console.error('查询数据服务日线失败:', error);
      setDailyError(error instanceof Error ? error.message : '查询失败');
    } finally {
      setDailyLoading(false);
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

  // 加载本地股票列表
  const loadLocalStocks = async () => {
    try {
      const result = await DataService.getLocalStockListDetailed();
      setLocalStocks(result.stocks || []);
    } catch (error) {
      console.error('加载本地股票列表失败:', error);
      setLocalStocks([]);
    }
  };

  // 初始化加载
  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([checkServiceStatus(), loadRemoteStocks(), loadLocalStocks()]);
      setLoading(false);
    };

    loadData();
  }, []);

  useEffect(() => {
    loadDataServiceSummary();
  }, [serviceStatus?.is_connected]);

  // 刷新数据
  const handleRefresh = async () => {
    setLoading(true);
    await Promise.all([checkServiceStatus(), loadRemoteStocks(), loadLocalStocks()]);
    setLoading(false);
  };

  // 同步远端数据
  const handleSyncRemoteData = async () => {
    if (syncing) {
      return;
    }

    setSyncing(true);
    setSyncResult(null);

    try {
      const result = await DataService.syncRemoteData();
      setSyncResult({
        success: result.success,
        message: result.message,
        synced_files: result.synced_files,
        total_files: result.total_files,
        total_size_mb: result.total_size_mb,
      });

      // 如果同步成功，刷新数据
      if (result.success) {
        await handleRefresh();
      }
    } catch (error) {
      console.error('同步远端数据失败:', error);
      setSyncResult({
        success: false,
        message: error instanceof Error ? error.message : '同步失败',
      });
    } finally {
      setSyncing(false);
    }
  };

  // 触发Qlib预计算
  const handleTriggerQlibPrecompute = async () => {
    if (precomputing) {
      return;
    }

    setPrecomputing(true);
    setPrecomputeTask(null);

    try {
      const task = await DataService.triggerQlibPrecompute();
      setPrecomputeTask({
        task_id: task.task_id,
        progress: task.progress,
        status: task.status,
        message: '预计算任务已创建，正在处理...',
      });
    } catch (error) {
      console.error('触发预计算失败:', error);
      setPrecomputeTask({
        task_id: '',
        progress: 0,
        status: 'failed',
        message: error instanceof Error ? error.message : '预计算任务创建失败',
      });
      setPrecomputing(false);
    }
  };

  // WebSocket监听预计算任务进度
  useEffect(() => {
    if (!precomputeTask) {
      return;
    }

    const handleTaskProgress = (data: { task_id: string; progress: number; status: string }) => {
      if (data.task_id === precomputeTask.task_id) {
        setPrecomputeTask(prev =>
          prev
            ? {
                ...prev,
                progress: data.progress,
                status: data.status,
              }
            : null
        );
      }
    };

    const handleTaskCompleted = (data: { task_id: string; results: any }) => {
      if (data.task_id === precomputeTask.task_id) {
        setPrecomputeTask(prev =>
          prev
            ? {
                ...prev,
                progress: 100,
                status: 'completed',
                message: '预计算完成！所有指标和因子已生成，可用于训练/回测。',
              }
            : null
        );
        setPrecomputing(false);
      }
    };

    const handleTaskFailed = (data: { task_id: string; error: string }) => {
      if (data.task_id === precomputeTask.task_id) {
        setPrecomputeTask(prev =>
          prev
            ? {
                ...prev,
                status: 'failed',
                message: `预计算失败: ${data.error}`,
              }
            : null
        );
        setPrecomputing(false);
      }
    };

    wsService.on('task:progress', handleTaskProgress);
    wsService.on('task:completed', handleTaskCompleted);
    wsService.on('task:failed', handleTaskFailed);

    return () => {
      wsService.off('task:progress', handleTaskProgress);
      wsService.off('task:completed', handleTaskCompleted);
      wsService.off('task:failed', handleTaskFailed);
    };
  }, [precomputeTask]);

  if (loading) {
    return <LoadingSpinner text="加载数据信息..." />;
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
            数据管理
          </Typography>
          <Typography variant="body2" color="text.secondary">
            查看远端数据服务状态和股票列表
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
          <Button
            variant="contained"
            color="primary"
            size="large"
            startIcon={<Download size={20} />}
            onClick={handleSyncRemoteData}
            disabled={syncing}
          >
            {syncing ? '同步中...' : '同步远端数据'}
          </Button>
          <Button
            variant="contained"
            color="secondary"
            size="large"
            startIcon={<Zap size={20} />}
            onClick={handleTriggerQlibPrecompute}
            disabled={precomputing || precomputeTask?.status === 'running'}
          >
            {precomputing || precomputeTask?.status === 'running'
              ? '预计算中...'
              : '离线生成 Qlib 指标/因子'}
          </Button>
        </Box>
      </Box>

      {/* 同步结果提示 */}
      {syncResult && (
        <Alert
          severity={syncResult.success ? 'success' : 'error'}
          icon={syncResult.success ? <Zap size={20} /> : <XCircle size={20} />}
        >
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            {syncResult.message}
          </Typography>
          {syncResult.success && syncResult.synced_files !== undefined && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="caption" display="block">
                已同步: {syncResult.synced_files}/{syncResult.total_files} 个文件
              </Typography>
              {syncResult.total_size_mb !== undefined && (
                <Typography variant="caption" display="block">
                  总大小: {syncResult.total_size_mb} MB
                </Typography>
              )}
            </Box>
          )}
        </Alert>
      )}

      {/* 预计算任务进度提示 */}
      {precomputeTask && (
        <Alert
          severity={
            precomputeTask.status === 'completed'
              ? 'success'
              : precomputeTask.status === 'failed'
                ? 'error'
                : 'info'
          }
          icon={
            precomputeTask.status === 'completed' ? (
              <Zap size={20} />
            ) : precomputeTask.status === 'failed' ? (
              <XCircle size={20} />
            ) : (
              <Database size={20} />
            )
          }
        >
          <Typography variant="body2" sx={{ fontWeight: 500 }}>
            {precomputeTask.message || '正在为全市场所有股票计算所有指标和因子，请耐心等待...'}
          </Typography>
          {precomputeTask.status === 'running' && (
            <Box sx={{ mt: 2 }}>
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  mb: 1,
                }}
              >
                <Typography variant="caption" color="text.secondary">
                  进度: {precomputeTask.progress.toFixed(1)}%
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  状态: {precomputeTask.status === 'running' ? '运行中' : precomputeTask.status}
                </Typography>
              </Box>
              <Box
                sx={{
                  width: '100%',
                  height: 8,
                  backgroundColor: 'rgba(0, 0, 0, 0.1)',
                  borderRadius: 1,
                  overflow: 'hidden',
                }}
              >
                <Box
                  sx={{
                    width: `${precomputeTask.progress}%`,
                    height: '100%',
                    backgroundColor: 'primary.main',
                    transition: 'width 0.3s ease',
                  }}
                />
              </Box>
            </Box>
          )}
          {precomputeTask.status === 'completed' && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="caption" display="block" color="success.main">
                预计算完成！所有指标和因子已生成，可用于训练/回测。
              </Typography>
            </Box>
          )}
        </Alert>
      )}

      {/* 服务状态 */}
      <Card>
        <CardHeader
          avatar={<Server size={24} />}
          title="远端服务状态"
          action={
            <Button
              variant="outlined"
              size="small"
              startIcon={<RefreshCw size={16} />}
              onClick={checkServiceStatus}
            >
              刷新
            </Button>
          }
        />
        <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              连接状态
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {serviceStatus?.is_connected ? (
                <>
                  <Wifi size={16} color="#2e7d32" />
                  <Chip label="已连接" color="success" size="small" />
                </>
              ) : (
                <>
                  <WifiOff size={16} color="#d32f2f" />
                  <Chip label="未连接" color="error" size="small" />
                </>
              )}
            </Box>
          </Box>

          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              服务地址
            </Typography>
            <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
              {serviceStatus?.service_url || '--'}
            </Typography>
          </Box>

          {serviceStatus?.is_connected && (
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2" color="text.secondary">
                响应时间
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Zap size={12} color="#ed6c02" />
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {serviceStatus.response_time}ms
                </Typography>
              </Box>
            </Box>
          )}

          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              最后检查
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {serviceStatus?.last_check
                ? new Date(serviceStatus.last_check).toLocaleString()
                : '--'}
            </Typography>
          </Box>

          {serviceStatus?.error_message && (
            <Alert severity="error" icon={<XCircle size={20} />}>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                连接错误
              </Typography>
              <Typography variant="caption">{serviceStatus.error_message}</Typography>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* back_test_data_service API 面板 */}
      <Card
        sx={{
          border: serviceStatus?.is_connected ? '1px solid rgba(46, 125, 50, 0.25)' : undefined,
          background: serviceStatus?.is_connected
            ? 'linear-gradient(135deg, rgba(46,125,50,0.06), rgba(25,118,210,0.04))'
            : undefined,
        }}
      >
        <CardHeader
          avatar={<Database size={24} />}
          title="数据服务 API"
          subheader="连接成功后可直接通过股票平台代理请求 back_test_data_service"
          action={
            <Button
              variant="outlined"
              size="small"
              startIcon={<RefreshCw size={16} />}
              onClick={loadDataServiceSummary}
              disabled={!serviceStatus?.is_connected || summaryLoading}
            >
              {summaryLoading ? '刷新中...' : '刷新汇总'}
            </Button>
          }
        />
        <CardContent sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          {!serviceStatus?.is_connected ? (
            <Alert severity="warning" icon={<WifiOff size={20} />}>
              数据服务未连接。请先启动 back_test_data_service，再刷新状态。
            </Alert>
          ) : (
            <>
              <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
                {[
                  { label: '股票数', value: dataSummary?.total_stocks ?? '--' },
                  {
                    label: '总记录数',
                    value: dataSummary?.total_records?.toLocaleString() ?? '--',
                  },
                  { label: '完整', value: dataSummary?.complete_stocks ?? '--' },
                  { label: '不完整', value: dataSummary?.incomplete_stocks ?? '--' },
                  { label: '缺失', value: dataSummary?.missing_stocks ?? '--' },
                ].map(item => (
                  <Box
                    key={item.label}
                    sx={{
                      flex: 1,
                      minWidth: 120,
                      p: 2,
                      borderRadius: 2,
                      bgcolor: 'background.paper',
                      border: '1px solid',
                      borderColor: 'divider',
                    }}
                  >
                    <Typography variant="caption" color="text.secondary">
                      {item.label}
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 700 }}>
                      {item.value}
                    </Typography>
                  </Box>
                ))}
              </Stack>

              <Box>
                <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
                  日线数据查询
                </Typography>
                <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems="stretch">
                  <TextField
                    label="股票代码"
                    size="small"
                    value={dailyQuery.stockCode}
                    onChange={event =>
                      setDailyQuery(prev => ({ ...prev, stockCode: event.target.value.trim() }))
                    }
                    placeholder="例如 000001"
                    sx={{ minWidth: 160 }}
                  />
                  <TextField
                    label="开始日期"
                    type="date"
                    size="small"
                    value={dailyQuery.startDate}
                    onChange={event =>
                      setDailyQuery(prev => ({ ...prev, startDate: event.target.value }))
                    }
                    InputLabelProps={{ shrink: true }}
                  />
                  <TextField
                    label="结束日期"
                    type="date"
                    size="small"
                    value={dailyQuery.endDate}
                    onChange={event =>
                      setDailyQuery(prev => ({ ...prev, endDate: event.target.value }))
                    }
                    InputLabelProps={{ shrink: true }}
                  />
                  <Button
                    variant="contained"
                    startIcon={<Zap size={16} />}
                    onClick={handleQueryDailyData}
                    disabled={!dailyQuery.stockCode || dailyLoading}
                  >
                    {dailyLoading ? '查询中...' : '请求数据服务'}
                  </Button>
                </Stack>
              </Box>

              {dailyError && (
                <Alert severity="warning" icon={<XCircle size={20} />}>
                  {dailyError}
                </Alert>
              )}

              {dailyResult?.success && (
                <Box>
                  <Divider sx={{ mb: 2 }} />
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body2" sx={{ fontWeight: 600 }}>
                      查询结果：{dailyResult.stock_code || dailyQuery.stockCode}
                    </Typography>
                    <Chip
                      label={`${dailyResult.total_records || 0} 条`}
                      color="success"
                      size="small"
                    />
                  </Box>
                  <Box sx={{ overflowX: 'auto' }}>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>日期</TableCell>
                          <TableCell align="right">开盘</TableCell>
                          <TableCell align="right">最高</TableCell>
                          <TableCell align="right">最低</TableCell>
                          <TableCell align="right">收盘</TableCell>
                          <TableCell align="right">成交量</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {(dailyResult.data || []).slice(0, 12).map(row => (
                          <TableRow key={row.date}>
                            <TableCell>{row.date}</TableCell>
                            <TableCell align="right">{row.open.toFixed(2)}</TableCell>
                            <TableCell align="right">{row.high.toFixed(2)}</TableCell>
                            <TableCell align="right">{row.low.toFixed(2)}</TableCell>
                            <TableCell align="right">{row.close.toFixed(2)}</TableCell>
                            <TableCell align="right">{row.volume.toLocaleString()}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </Box>
                  {(dailyResult.data?.length || 0) > 12 && (
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{ mt: 1, display: 'block' }}
                    >
                      仅预览前 12 条记录。
                    </Typography>
                  )}
                </Box>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {/* 股票列表 - 使用Tabs */}
      <Card>
        <CardHeader
          avatar={<Database size={24} />}
          title="股票列表"
          action={
            <Button
              variant="outlined"
              size="small"
              startIcon={<RefreshCw size={16} />}
              onClick={() => {
                if (activeTab === 'remote') {
                  loadRemoteStocks();
                } else {
                  loadLocalStocks();
                }
              }}
            >
              刷新
            </Button>
          }
        />
        <CardContent>
          <Tabs
            value={activeTab}
            onChange={(e, newValue) => setActiveTab(newValue)}
            aria-label="股票列表页签"
          >
            <Tab
              value="remote"
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <span>远端股票列表</span>
                  <Chip label={remoteStocks.length} size="small" />
                </Box>
              }
            />
            <Tab
              value="local"
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <span>本地股票列表</span>
                  <Chip label={localStocks.length} size="small" />
                </Box>
              }
            />
          </Tabs>

          <Box sx={{ mt: 2 }}>
            {activeTab === 'remote' && (
              <Box>
                {remoteStocks.length === 0 ? (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <Typography variant="body2" color="text.secondary">
                      暂无股票数据
                    </Typography>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{ mt: 1, display: 'block' }}
                    >
                      请检查远端服务连接状态
                    </Typography>
                  </Box>
                ) : (
                  <Box>
                    {/* 移动端：卡片列表 */}
                    <Box sx={{ display: { xs: 'block', md: 'none' } }}>
                      {remoteStocks.map(stock => (
                        <MobileStockCard key={stock.ts_code} stock={stock} type="remote" />
                      ))}
                    </Box>

                    {/* 桌面端：表格 */}
                    <Box sx={{ display: { xs: 'none', md: 'block' }, overflowX: 'auto' }}>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>股票代码</TableCell>
                            <TableCell>股票名称</TableCell>
                            <TableCell>数据范围</TableCell>
                            <TableCell>最后更新</TableCell>
                            <TableCell>状态</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {remoteStocks.map(stock => (
                            <TableRow key={stock.ts_code}>
                              <TableCell>
                                <Typography
                                  variant="body2"
                                  sx={{ fontFamily: 'monospace', fontWeight: 500 }}
                                >
                                  {stock.ts_code}
                                </Typography>
                              </TableCell>
                              <TableCell>{stock.name || '--'}</TableCell>
                              <TableCell>
                                {stock.data_range ? (
                                  <Box>
                                    <Typography variant="body2">
                                      {stock.data_range.start_date} 至 {stock.data_range.end_date}
                                    </Typography>
                                    {stock.data_range.total_days && (
                                      <Typography variant="caption" color="text.secondary">
                                        {stock.data_range.total_days} 天
                                      </Typography>
                                    )}
                                  </Box>
                                ) : (
                                  '--'
                                )}
                              </TableCell>
                              <TableCell>
                                {stock.last_update
                                  ? new Date(stock.last_update).toLocaleDateString()
                                  : '--'}
                              </TableCell>
                              <TableCell>
                                {stock.status === 'complete' ? (
                                  <Chip label="完整" color="success" size="small" />
                                ) : stock.status === 'incomplete' ? (
                                  <Chip label="不完整" color="warning" size="small" />
                                ) : stock.status === 'missing' ? (
                                  <Chip label="缺失" color="error" size="small" />
                                ) : (
                                  <Chip label="未知" size="small" />
                                )}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </Box>
                  </Box>
                )}
              </Box>
            )}

            {activeTab === 'local' && (
              <Box>
                {localStocks.length === 0 ? (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <Typography variant="body2" color="text.secondary">
                      暂无本地股票数据
                    </Typography>
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{ mt: 1, display: 'block' }}
                    >
                      请先同步远端数据
                    </Typography>
                  </Box>
                ) : (
                  <Box>
                    {/* 移动端：卡片列表 */}
                    <Box sx={{ display: { xs: 'block', md: 'none' } }}>
                      {localStocks.map(stock => (
                        <MobileStockCard key={stock.ts_code} stock={stock} type="local" />
                      ))}
                    </Box>

                    {/* 桌面端：表格 */}
                    <Box sx={{ display: { xs: 'none', md: 'block' }, overflowX: 'auto' }}>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>股票代码</TableCell>
                            <TableCell>股票名称</TableCell>
                            <TableCell>数据范围</TableCell>
                            <TableCell>文件数</TableCell>
                            <TableCell>记录数</TableCell>
                            <TableCell>文件大小</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {localStocks.map(stock => (
                            <TableRow key={stock.ts_code}>
                              <TableCell>
                                <Typography
                                  variant="body2"
                                  sx={{ fontFamily: 'monospace', fontWeight: 500 }}
                                >
                                  {stock.ts_code}
                                </Typography>
                              </TableCell>
                              <TableCell>{stock.name || stock.ts_code}</TableCell>
                              <TableCell>
                                {stock.data_range ? (
                                  <Box>
                                    <Typography variant="body2">
                                      {stock.data_range.start_date} 至 {stock.data_range.end_date}
                                    </Typography>
                                    {stock.data_range.total_days && (
                                      <Typography variant="caption" color="text.secondary">
                                        {stock.data_range.total_days} 天
                                      </Typography>
                                    )}
                                  </Box>
                                ) : (
                                  '--'
                                )}
                              </TableCell>
                              <TableCell>{stock.file_count || 0}</TableCell>
                              <TableCell>{stock.record_count?.toLocaleString() || '--'}</TableCell>
                              <TableCell>
                                {stock.total_size
                                  ? `${(stock.total_size / 1024 / 1024).toFixed(2)} MB`
                                  : '--'}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </Box>
                  </Box>
                )}
              </Box>
            )}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}
