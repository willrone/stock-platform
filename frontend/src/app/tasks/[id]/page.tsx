/**
 * 任务详情页面
 * 
 * 显示任务的详细信息，包括：
 * - 任务基本信息
 * - 实时进度更新
 * - 预测结果展示（TradingView图表 + ECharts）
 * - 操作控制
 */

'use client';

import React, { useEffect, useState, useCallback } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
  CardActions,
  Button,
  LinearProgress,
  Chip,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  Dialog,
  DialogContent,
  DialogTitle,
  DialogActions,
  Divider,
  Select,
  MenuItem,
  Tabs,
  Tab,
  Box,
  Typography,
  TableContainer,
  Paper,
  FormControl,
  InputLabel,
  IconButton,
  CircularProgress,
} from '@mui/material';
import {
  ArrowLeft,
  RefreshCw,
  Play,
  Download,
  Trash2,
  ChevronLeft,
  ChevronRight,
  TrendingUp,
  TrendingDown,
  Minus,
  AlertTriangle,
  BarChart3,
  LineChart,
  Activity,
  PieChart,
  Calendar,
  FileText,
  Save,
} from 'lucide-react';
import { useRouter, useParams } from 'next/navigation';
import { useTaskStore, Task } from '../../../stores/useTaskStore';
import { TaskService, PredictionResult } from '../../../services/taskService';
import { BacktestService } from '../../../services/backtestService';
import { BacktestDataAdapter } from '../../../services/backtestDataAdapter';
import { wsService } from '../../../services/websocket';
import { LoadingSpinner } from '../../../components/common/LoadingSpinner';
import BacktestOverview from '../../../components/backtest/BacktestOverview';
import { CostAnalysis } from '../../../components/backtest/CostAnalysis';
import BacktestTaskStatus from '../../../components/backtest/BacktestTaskStatus';
import BacktestProgressMonitor from '../../../components/backtest/BacktestProgressMonitor';
import { TradeHistoryTable } from '../../../components/backtest/TradeHistoryTable';
import { SignalHistoryTable } from '../../../components/backtest/SignalHistoryTable';
import { PositionAnalysis } from '../../../components/backtest/PositionAnalysis';
import { RiskAnalysis } from '../../../components/backtest/RiskAnalysis';
import { PerformanceBreakdown } from '../../../components/backtest/PerformanceBreakdown';
import { SaveStrategyConfigDialog } from '../../../components/backtest/SaveStrategyConfigDialog';
import { StrategyConfigService } from '../../../services/strategyConfigService';
import dynamic from 'next/dynamic';

// 动态导入图表组件
const TradingViewChart = dynamic(() => import('../../../components/charts/TradingViewChart'), {
  ssr: false,
  loading: () => <div className="h-96 flex items-center justify-center">加载图表中...</div>
});

const PredictionChart = dynamic(() => import('../../../components/charts/PredictionChart'), {
  ssr: false,
  loading: () => <div className="h-64 flex items-center justify-center">加载预测图表中...</div>
});

const BacktestChart = dynamic(() => import('../../../components/charts/BacktestChart'), {
  ssr: false,
  loading: () => <div className="h-64 flex items-center justify-center">加载回测图表中...</div>
});

const InteractiveChartsContainer = dynamic(() => import('../../../components/charts/InteractiveChartsContainer'), {
  ssr: false,
  loading: () => <div className="h-96 flex items-center justify-center">加载交互式图表中...</div>
});

export default function TaskDetailPage() {
  const router = useRouter();
  const params = useParams();
  const taskId = params.id as string;

  const { currentTask, setCurrentTask, updateTask } = useTaskStore();
  const [loading, setLoading] = useState(true);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedStock, setSelectedStock] = useState<string>('');
  const [backtestDetailedData, setBacktestDetailedData] = useState<any>(null);
  const [adaptedRiskData, setAdaptedRiskData] = useState<any>(null);
  const [adaptedPerformanceData, setAdaptedPerformanceData] = useState<any>(null);
  const [loadingBacktestData, setLoadingBacktestData] = useState(false);
  const [selectedBacktestTab, setSelectedBacktestTab] = useState<string>('overview');
  const [selectedPredictionTab, setSelectedPredictionTab] = useState<string>('chart');
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [isSaveConfigOpen, setIsSaveConfigOpen] = useState(false);
  const onDeleteOpen = () => setIsDeleteOpen(true);
  const onDeleteClose = () => setIsDeleteOpen(false);
  const onSaveConfigOpen = () => setIsSaveConfigOpen(true);
  const onSaveConfigClose = () => setIsSaveConfigOpen(false);
  const [deleteForce, setDeleteForce] = useState(false);
  const [savingConfig, setSavingConfig] = useState(false);
  const [selectedStocksPage, setSelectedStocksPage] = useState(1); // 已选股票分页

  // 加载回测详细数据
  const loadBacktestDetailedData = async (force: boolean = false) => {
    // 如果数据已加载且不强制刷新，则跳过
    if (!force && backtestDetailedData !== null && !loadingBacktestData) {
      console.log('[TaskDetail] 回测详细数据已加载，跳过重复加载');
      return;
    }
    
    // 检查任务状态（允许传入task参数以支持提前加载）
    const task = currentTask;
    if (!task || task.task_type !== 'backtest' || task.status !== 'completed') {
      console.log('[TaskDetail] 任务状态不满足加载条件:', { 
        hasTask: !!task, 
        taskType: task?.task_type, 
        status: task?.status 
      });
      return;
    }

    setLoadingBacktestData(true);
    try {
      console.log('[TaskDetail] 开始加载回测详细数据...');
      const detailedResult = await BacktestService.getDetailedResult(taskId);
      console.log('[TaskDetail] 后端返回的详细数据:', detailedResult);
      console.log('[TaskDetail] position_analysis 数据:', detailedResult?.position_analysis);
      console.log('[TaskDetail] position_analysis 类型:', typeof detailedResult?.position_analysis);
      console.log('[TaskDetail] position_analysis 是否为数组:', Array.isArray(detailedResult?.position_analysis));
      if (detailedResult?.position_analysis && typeof detailedResult.position_analysis === 'object' && !Array.isArray(detailedResult.position_analysis)) {
        console.log('[TaskDetail] position_analysis.stock_performance:', detailedResult.position_analysis.stock_performance);
        console.log('[TaskDetail] stock_performance 长度:', detailedResult.position_analysis.stock_performance?.length);
      }
      
      setBacktestDetailedData(detailedResult);
      
      // 使用数据适配器转换数据格式
      console.log('[TaskDetail] 开始适配风险分析数据...');
      const riskMetrics = BacktestDataAdapter.adaptRiskMetrics(detailedResult);
      const returnDistribution = BacktestDataAdapter.generateReturnDistribution(detailedResult);
      const rollingMetrics = BacktestDataAdapter.generateRollingMetrics(detailedResult);
      
      setAdaptedRiskData({
        riskMetrics,
        returnDistribution,
        rollingMetrics
      });
      console.log('[TaskDetail] 风险分析数据适配完成:', { riskMetrics, returnDistribution, rollingMetrics });
      
      console.log('[TaskDetail] 开始适配绩效分解数据...');
      const monthlyPerformance = BacktestDataAdapter.adaptMonthlyPerformance(detailedResult);
      const yearlyPerformance = BacktestDataAdapter.generateYearlyPerformance(detailedResult);
      const seasonalAnalysis = BacktestDataAdapter.generateSeasonalAnalysis(detailedResult);
      const benchmarkComparison = BacktestDataAdapter.generateBenchmarkComparison(detailedResult);
      
      setAdaptedPerformanceData({
        monthlyPerformance,
        yearlyPerformance,
        seasonalAnalysis,
        benchmarkComparison
      });
      console.log('[TaskDetail] 绩效分解数据适配完成:', { monthlyPerformance, yearlyPerformance, seasonalAnalysis, benchmarkComparison });
      
    } catch (error) {
      console.error('[TaskDetail] 加载回测详细数据失败:', error);
      // 如果详细数据不存在，使用基础回测数据
      setBacktestDetailedData(null);
      setAdaptedRiskData(null);
      setAdaptedPerformanceData(null);
    } finally {
      setLoadingBacktestData(false);
    }
  };

  // 加载任务详情
  const loadTaskDetail = async () => {
    try {
      const task = await TaskService.getTaskDetail(taskId);
      console.log('加载的任务详情:', {
        task_id: task.task_id,
        task_type: task.task_type,
        status: task.status,
        results: task.results,
        backtest_results: task.backtest_results,
        result: task.result
      });
      setCurrentTask(task);
      
      // 如果任务已完成，加载预测结果
      if (task.status === 'completed' && task.results) {
        // 如果是预测任务，加载预测结果
        if (task.task_type === 'prediction') {
          const results = await TaskService.getTaskResults(taskId);
          setPredictions(results);
          // 默认选择第一个股票
          if (results.length > 0) {
            setSelectedStock(results[0].stock_code);
          }
        } else if (task.task_type === 'backtest') {
          // 如果是回测任务，确保回测结果已加载
          console.log('回测任务详情:', {
            'results.backtest_results': task.results?.backtest_results,
            'backtest_results': task.backtest_results,
            'result': task.result
          });
          // 如果没有股票代码，使用任务配置中的股票代码
          if (!selectedStock && task.stock_codes && task.stock_codes.length > 0) {
            setSelectedStock(task.stock_codes[0]);
          }
          // 加载回测详细数据（初始化加载，强制刷新）
          await loadBacktestDetailedData(true);
        }
      }
    } catch (error) {
      console.error('加载任务详情失败:', error);
    } finally {
      setLoading(false);
    }
  };

  // 初始化加载
  useEffect(() => {
    if (taskId) {
      loadTaskDetail();
      // 订阅任务更新
      wsService.subscribeToTask(taskId);
    }

    return () => {
      if (taskId) {
        wsService.unsubscribeFromTask(taskId);
      }
    };
  }, [taskId]);

  // 当currentTask状态变化时，确保回测详细数据已加载（仅在任务状态变为completed时触发一次）
  // 使用 ref 来跟踪是否已经触发过加载，避免重复加载
  const hasTriggeredLoadRef = React.useRef(false);
  useEffect(() => {
    if (currentTask && 
        currentTask.task_type === 'backtest' && 
        currentTask.status === 'completed' &&
        !hasTriggeredLoadRef.current) {
      // 只在第一次检测到任务完成时触发加载
      hasTriggeredLoadRef.current = true;
      console.log('[TaskDetail] 检测到回测任务已完成，触发数据加载');
      loadBacktestDetailedData();
    }
    // 如果任务状态变化，重置 ref
    if (currentTask?.status !== 'completed') {
      hasTriggeredLoadRef.current = false;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentTask?.status, currentTask?.task_type]);

  // 当currentTask状态变化时，确保回测详细数据已加载（移除selectedBacktestTab依赖，避免干扰页签切换）

  // WebSocket实时更新
  useEffect(() => {
    const handleTaskProgress = (data: { task_id: string; progress: number; status: string }) => {
      if (data.task_id === taskId) {
        updateTask(data.task_id, {
          progress: data.progress,
          status: data.status as Task['status'],
        });
        
        if (currentTask) {
          setCurrentTask({
            ...currentTask,
            progress: data.progress,
            status: data.status as Task['status'],
          });
        }
      }
    };

    const handleTaskCompleted = async (data: { task_id: string; results: any }) => {
      if (data.task_id === taskId) {
        // 重新加载任务详情以获取完整数据
        try {
          const task = await TaskService.getTaskDetail(taskId);
          const updatedTask = {
            ...task,
            status: 'completed' as const,
            progress: 100,
            completed_at: new Date().toISOString(),
          };
          
          setCurrentTask(updatedTask);
          updateTask(data.task_id, updatedTask);
          
          // 如果是预测任务，加载预测结果
          if (task.task_type === 'prediction') {
            const results = await TaskService.getTaskResults(taskId);
            setPredictions(results);
            if (results.length > 0) {
              setSelectedStock(results[0].stock_code);
            }
          } else if (task.task_type === 'backtest') {
            // 回测任务，确保回测结果已加载
            console.log('回测任务完成，回测结果:', task.results?.backtest_results);
            // 重新加载回测详细数据（强制刷新，因为任务刚完成）
            await loadBacktestDetailedData(true);
          }
        } catch (error) {
          console.error('加载任务详情失败:', error);
        }
        
        console.log('任务执行完成');
      }
    };

    const handleTaskFailed = (data: { task_id: string; error: string }) => {
      if (data.task_id === taskId) {
        const updatedTask = {
          ...currentTask!,
          status: 'failed' as const,
          error_message: data.error,
        };
        
        setCurrentTask(updatedTask);
        updateTask(data.task_id, updatedTask);
        console.error('任务执行失败');
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
  }, [taskId, currentTask, updateTask, setCurrentTask]);

  // 刷新任务
  const handleRefresh = async () => {
    setRefreshing(true);
    await loadTaskDetail();
    setRefreshing(false);
  };

  // 重新运行任务
  const handleRetry = async () => {
    try {
      await TaskService.retryTask(taskId);
      console.log('任务已重新启动');
      await loadTaskDetail();
    } catch (error) {
      console.error('重新运行失败');
    }
  };

  // 删除任务
  const handleDelete = async () => {
    try {
      await TaskService.deleteTask(taskId, deleteForce);
      console.log(`任务删除成功${deleteForce ? '（强制删除）' : ''}`);
      router.push('/tasks');
    } catch (error: any) {
      console.error('删除任务失败:', error);
      // 如果任务正在运行或可能是僵尸任务，显示强制删除选项
      if (error.message?.includes('正在运行中') || 
          error.message?.includes('运行中') ||
          currentTask?.status === 'running') {
        setDeleteForce(true);
        onDeleteOpen();
      }
    }
  };

  // 导出结果
  const handleExport = async () => {
    try {
      const blob = await TaskService.exportTaskResults(taskId, 'csv');
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `task_${taskId}_results.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      console.log('结果导出成功');
    } catch (error) {
      console.error('导出失败');
    }
  };

  // 获取策略配置信息
  const getStrategyConfig = () => {
    if (!currentTask || currentTask.task_type !== 'backtest') {
      return null;
    }

    // 优先从任务配置中获取（最可靠）
    if (currentTask.config?.backtest_config?.strategy_config) {
      return {
        strategyName: currentTask.config.backtest_config.strategy_name,
        parameters: currentTask.config.backtest_config.strategy_config,
      };
    }

    // 从回测结果中获取
    const backtestData = currentTask.result || currentTask.results?.backtest_results || currentTask.backtest_results;
    if (backtestData) {
      const backtestConfig = backtestData.backtest_config;
      if (backtestConfig && backtestConfig.strategy_config) {
        return {
          strategyName: backtestConfig.strategy_name || currentTask.config?.backtest_config?.strategy_name,
          parameters: backtestConfig.strategy_config,
        };
      }
    }

    return null;
  };

  // 保存策略配置
  const handleSaveConfig = async (configName: string, description: string) => {
    const configInfo = getStrategyConfig();
    if (!configInfo) {
      throw new Error('无法获取策略配置信息');
    }

    setSavingConfig(true);
    try {
      await StrategyConfigService.createConfig({
        config_name: configName,
        strategy_name: configInfo.strategyName,
        parameters: configInfo.parameters,
        description: description,
      });
      console.log('策略配置保存成功');
    } catch (error: any) {
      console.error('保存策略配置失败:', error);
      throw error;
    } finally {
      setSavingConfig(false);
    }
  };

  // 返回任务列表
  const handleBack = () => {
    router.push('/tasks');
  };

  // 获取状态标签
  const getStatusChip = (status: Task['status']) => {
    const statusConfig = {
      created: { color: 'default' as const, text: '已创建' },
      running: { color: 'primary' as const, text: '运行中' },
      completed: { color: 'success' as const, text: '已完成' },
      failed: { color: 'error' as const, text: '失败' },
    };
    
    const config = statusConfig[status] || statusConfig.created;
    return <Chip label={config.text} color={config.color} size="small" />;
  };

  // 获取预测方向图标
  const getPredictionIcon = (direction: number) => {
    if (direction > 0) return <TrendingUp className="w-4 h-4 text-success" />;
    if (direction < 0) return <TrendingDown className="w-4 h-4 text-danger" />;
    return <Minus className="w-4 h-4 text-default-500" />;
  };

  // 获取预测方向文本
  const getPredictionText = (direction: number) => {
    if (direction > 0) return '上涨';
    if (direction < 0) return '下跌';
    return '持平';
  };

  if (loading) {
    return <LoadingSpinner text="加载任务详情..." />;
  }

  if (!currentTask) {
    return (
      <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: 384, gap: 2 }}>
        <Typography variant="body2" color="text.secondary">任务不存在或已被删除</Typography>
        <Button variant="contained" color="primary" onClick={handleBack}>
          返回任务列表
        </Button>
      </Box>
    );
  }

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
      {/* 页面标题 */}
      <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <IconButton onClick={handleBack} size="small">
            <ArrowLeft size={20} />
          </IconButton>
          <Box>
            <Typography variant="h4" component="h1" sx={{ fontWeight: 600 }}>
              {currentTask.task_name}
            </Typography>
            <Typography variant="caption" color="text.secondary">任务ID: {currentTask.task_id}</Typography>
          </Box>
          {getStatusChip(currentTask.status)}
        </Box>
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            variant="outlined"
            startIcon={<RefreshCw size={16} />}
            onClick={handleRefresh}
            disabled={refreshing}
          >
            刷新
          </Button>
          
          {currentTask.status === 'failed' && (
            <Button
              variant="contained"
              color="primary"
              startIcon={<Play size={16} />}
              onClick={handleRetry}
            >
              重新运行
            </Button>
          )}
          
          {currentTask.status === 'completed' && (
            <Button
              variant="outlined"
              color="secondary"
              startIcon={<Download size={16} />}
              onClick={handleExport}
            >
              导出结果
            </Button>
          )}
          
          <Button
            variant="outlined"
            color="error"
            startIcon={<Trash2 size={16} />}
            onClick={onDeleteOpen}
          >
            删除
          </Button>
        </Box>
      </Box>

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '2fr 1fr' }, gap: 3 }}>
        {/* 主要内容区域 */}
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          {/* 任务进度 */}
          {currentTask.task_type === 'backtest' && (currentTask.status === 'running' || currentTask.status === 'created') ? (
            <BacktestProgressMonitor
              taskId={taskId}
              onComplete={(results) => {
                console.log('回测完成:', results);
                // 刷新任务数据
                loadTaskDetail();
              }}
              onError={(error) => {
                console.error('回测错误:', error);
                // 刷新任务数据以获取最新状态
                loadTaskDetail();
              }}
              onCancel={() => {
                console.log('回测已取消');
                // 刷新任务数据
                loadTaskDetail();
              }}
            />
          ) : (
            /* 通用任务进度显示 */
            <Card>
              <CardHeader title="任务进度" />
              <CardContent>
                <Box sx={{ mb: 2 }}>
                  <LinearProgress
                    variant="determinate"
                    value={currentTask.progress}
                    color={currentTask.status === 'failed' ? 'error' : 'primary'}
                    sx={{ height: 10, borderRadius: 5 }}
                  />
                </Box>
                {currentTask.status === 'running' && (
                  <Typography variant="caption" color="text.secondary">
                    任务正在执行中，请耐心等待...
                  </Typography>
                )}
                {currentTask.status === 'failed' && currentTask.error_message && (
                  <Box sx={{ bgcolor: 'error.light', border: 1, borderColor: 'error.main', borderRadius: 1, p: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                      <AlertTriangle size={20} color="#d32f2f" style={{ marginTop: 2 }} />
                      <Box>
                        <Typography variant="body2" sx={{ fontWeight: 500, color: 'error.dark' }}>
                          任务执行失败
                        </Typography>
                        <Typography variant="caption" sx={{ color: 'error.dark', mt: 0.5, display: 'block' }}>
                          {currentTask.error_message}
                        </Typography>
                      </Box>
                    </Box>
                  </Box>
                )}
              </CardContent>
            </Card>
          )}

          {/* 根据任务类型显示不同内容 */}
          {currentTask.task_type === 'backtest' ? (
            /* 回测任务专用标签页 */
            <Card>
              <CardContent>
                <Box>
                  <Tabs 
                    value={selectedBacktestTab}
                    onChange={(e, newValue) => {
                      const tabKey = newValue as string;
                      setSelectedBacktestTab(tabKey);
                      console.log('[TaskDetail] 切换到页签:', tabKey);
                      
                      // 如果切换到持仓分析页签，确保数据已加载
                      if (tabKey === 'positions' && 
                          currentTask && 
                          currentTask.task_type === 'backtest' && 
                          currentTask.status === 'completed' &&
                          !backtestDetailedData && 
                          !loadingBacktestData) {
                        console.log('[TaskDetail] 切换到持仓分析页签，触发数据加载');
                        loadBacktestDetailedData();
                      }
                    }}
                    aria-label="回测结果展示"
                    variant="scrollable"
                    scrollButtons="auto"
                  >
                    <Tab label={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <BarChart3 size={16} />
                        <span>概览</span>
                      </Box>
                    } value="overview" />
                    <Tab label={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <LineChart size={16} />
                        <span>交互式图表</span>
                      </Box>
                    } value="charts" />
                    <Tab label={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <FileText size={16} />
                        <span>交易记录</span>
                      </Box>
                    } value="trades" />
                    <Tab label={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <AlertTriangle size={16} />
                        <span>信号记录</span>
                      </Box>
                    } value="signals" />
                    <Tab label={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <PieChart size={16} />
                        <span>持仓分析</span>
                      </Box>
                    } value="positions" />
                    <Tab label={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Calendar size={16} />
                        <span>月度分析</span>
                      </Box>
                    } value="monthly" />
                    <Tab label={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <Activity size={16} />
                        <span>风险分析</span>
                      </Box>
                    } value="risk" />
                    <Tab label={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                        <TrendingUp size={16} />
                        <span>绩效分解</span>
                      </Box>
                    } value="performance" />
                  </Tabs>

                  <Box sx={{ mt: 2 }}>
                    {selectedBacktestTab === 'overview' && (
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                        {/* 策略配置信息和保存按钮 */}
                        {(() => {
                          const configInfo = getStrategyConfig();
                          if (configInfo) {
                            return (
                              <Card>
                                <CardHeader
                                  title={
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                                      <Box>
                                        <Typography variant="h6" component="h4" sx={{ fontWeight: 600 }}>
                                          策略配置
                                        </Typography>
                                        <Typography variant="caption" color="text.secondary">
                                          策略: {configInfo.strategyName}
                                        </Typography>
                                      </Box>
                                      <Button
                                        variant="outlined"
                                        color="primary"
                                        size="small"
                                        startIcon={<Save size={16} />}
                                        onClick={onSaveConfigOpen}
                                        disabled={!configInfo.strategyName || Object.keys(configInfo.parameters).length === 0}
                                      >
                                        保存配置
                                      </Button>
                                    </Box>
                                  }
                                />
                                <CardContent>
                                  {Object.keys(configInfo.parameters).length > 0 ? (
                                    <Box sx={{ bgcolor: 'grey.100', borderRadius: 1, p: 1.5 }}>
                                      <Box component="pre" sx={{ fontSize: '0.75rem', color: 'text.secondary', whiteSpace: 'pre-wrap', fontFamily: 'monospace', m: 0 }}>
                                        {Object.entries(configInfo.parameters)
                                          .map(([key, value]) => {
                                            if (typeof value === 'object' && value !== null) {
                                              return `${key}: ${JSON.stringify(value, null, 2)}`;
                                            }
                                            return `${key}: ${value}`;
                                          })
                                          .join('\n')}
                                      </Box>
                                    </Box>
                                  ) : (
                                    <Typography variant="caption" color="text.secondary">暂无策略参数配置</Typography>
                                  )}
                                </CardContent>
                              </Card>
                            );
                          }
                          return null;
                        })()}
                        <BacktestOverview 
                          backtestData={currentTask.result || currentTask.results?.backtest_results || currentTask.backtest_results}
                          loading={loadingBacktestData}
                        />
                        <CostAnalysis
                          backtestData={currentTask.result || currentTask.results?.backtest_results || currentTask.backtest_results}
                          loading={loadingBacktestData}
                        />
                      </Box>
                    )}

                    {selectedBacktestTab === 'charts' && (
                      <Box sx={{ mt: 2 }}>
                        <InteractiveChartsContainer 
                          taskId={taskId}
                          stockCode={selectedStock || currentTask?.stock_codes?.[0]}
                          stockCodes={currentTask?.stock_codes || []}
                          backtestData={(() => {
                            const data = currentTask?.results?.backtest_results || 
                                       currentTask?.backtest_results ||
                                       (currentTask?.task_type === 'backtest' ? currentTask?.result : null);
                            return data;
                          })()}
                        />
                      </Box>
                    )}

                    {selectedBacktestTab === 'trades' && (
                      <Box sx={{ mt: 2 }}>
                        <TradeHistoryTable 
                          taskId={taskId}
                          onTradeClick={(trade) => {
                            console.log('查看交易详情:', trade);
                          }}
                        />
                      </Box>
                    )}

                    {selectedBacktestTab === 'signals' && (
                      <Box sx={{ mt: 2 }}>
                        <SignalHistoryTable 
                          taskId={taskId}
                          onSignalClick={(signal) => {
                            console.log('查看信号详情:', signal);
                          }}
                        />
                      </Box>
                    )}

                    {selectedBacktestTab === 'positions' && (
                      <Box sx={{ mt: 2 }}>
                      {(() => {
                        // 如果数据正在加载，显示加载中
                        if (loadingBacktestData) {
                          return (
                            <div className="text-center text-default-500 py-8">
                              <PieChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
                              <p>持仓分析数据加载中...</p>
                            </div>
                          );
                        }
                        
                        // 如果数据未加载（null），但不在加载中，可能是数据不存在或加载失败
                        if (backtestDetailedData === null) {
                          return (
                            <div className="text-center text-default-500 py-8">
                              <PieChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
                              <p>暂无持仓分析数据</p>
                            </div>
                          );
                        }
                        
                        const posAnalysis = backtestDetailedData?.position_analysis;
                        
                        // 如果position_analysis为null或undefined，显示无数据
                        if (posAnalysis === null || posAnalysis === undefined) {
                          return (
                            <div className="text-center text-default-500 py-8">
                              <PieChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
                              <p>暂无持仓分析数据</p>
                            </div>
                          );
                        }
                        
                        // 检查新格式（对象，包含stock_performance）
                        if (typeof posAnalysis === 'object' && !Array.isArray(posAnalysis)) {
                          // 检查是否有stock_performance字段
                          if (posAnalysis.stock_performance !== undefined) {
                            const stockPerf = posAnalysis.stock_performance;
                            if (Array.isArray(stockPerf) && stockPerf.length > 0) {
                              return (
                                <PositionAnalysis 
                                  positionAnalysis={posAnalysis}
                                  stockCodes={currentTask.stock_codes || []}
                                  taskId={taskId}
                                />
                              );
                            }
                            // stock_performance存在但为空数组
                            return (
                              <div className="text-center text-default-500 py-8">
                                <PieChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
                                <p>暂无持仓分析数据</p>
                              </div>
                            );
                          }
                          // 对象格式但没有stock_performance字段，可能是空对象或其他格式
                          // 检查是否是完全空对象
                          if (Object.keys(posAnalysis).length === 0) {
                            return (
                              <div className="text-center text-default-500 py-8">
                                <PieChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
                                <p>暂无持仓分析数据</p>
                              </div>
                            );
                          }
                          // 有其他字段但没有stock_performance，尝试直接使用（兼容其他可能的格式）
                          return (
                            <PositionAnalysis 
                              positionAnalysis={posAnalysis}
                              stockCodes={currentTask.stock_codes || []}
                              taskId={taskId}
                            />
                          );
                        }
                        
                        // 检查旧格式（数组）
                        if (Array.isArray(posAnalysis)) {
                          if (posAnalysis.length > 0) {
                            return (
                              <PositionAnalysis 
                                positionAnalysis={posAnalysis}
                                stockCodes={currentTask.stock_codes || []}
                              />
                            );
                          }
                          // 数组为空
                          return (
                            <div className="text-center text-default-500 py-8">
                              <PieChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
                              <p>暂无持仓分析数据</p>
                            </div>
                          );
                        }
                        
                        // 其他情况，显示无数据
                        return (
                          <div className="text-center text-default-500 py-8">
                            <PieChart className="w-12 h-12 mx-auto mb-4 opacity-50" />
                            <p>暂无持仓分析数据</p>
                          </div>
                        );
                      })()}
                      </Box>
                    )}

                  {selectedBacktestTab === 'monthly' && (
                    <Box sx={{ mt: 2 }}>
                      {backtestDetailedData?.monthly_returns ? (
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                          <Typography variant="h6" component="h4" sx={{ fontWeight: 600 }}>
                            月度收益热力图
                          </Typography>
                          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: 0.5 }}>
                            {backtestDetailedData.monthly_returns.map((monthData: any) => (
                              <Box
                                key={`${monthData.year}-${monthData.month}`}
                                sx={{
                                  p: 1,
                                  textAlign: 'center',
                                  fontSize: '0.75rem',
                                  borderRadius: 1,
                                  bgcolor: monthData.monthly_return >= 0 ? 'success.light' : 'error.light',
                                  color: monthData.monthly_return >= 0 ? 'success.dark' : 'error.dark',
                                }}
                                title={`${monthData.year}年${monthData.month}月: ${(monthData.monthly_return * 100).toFixed(2)}%`}
                              >
                                {monthData.month}月
                                <br />
                                {(monthData.monthly_return * 100).toFixed(1)}%
                              </Box>
                            ))}
                          </Box>
                        </Box>
                      ) : (
                        <Box sx={{ textAlign: 'center', py: 4 }}>
                          <Calendar size={48} color="#999" style={{ margin: '0 auto 16px' }} />
                          <Typography variant="body2" color="text.secondary">月度分析数据加载中...</Typography>
                        </Box>
                      )}
                    </Box>
                  )}

                  {selectedBacktestTab === 'risk' && (
                    <Box sx={{ mt: 2 }}>
                      {adaptedRiskData ? (
                        <RiskAnalysis 
                          taskId={taskId}
                          riskMetrics={adaptedRiskData.riskMetrics}
                          returnDistribution={adaptedRiskData.returnDistribution}
                          rollingMetrics={adaptedRiskData.rollingMetrics}
                        />
                      ) : (
                        <Box sx={{ textAlign: 'center', py: 4 }}>
                          <Activity size={48} color="#999" style={{ margin: '0 auto 16px' }} />
                          <Typography variant="body2" color="text.secondary">风险分析数据加载中...</Typography>
                        </Box>
                      )}
                    </Box>
                  )}

                  {selectedBacktestTab === 'performance' && (
                    <Box sx={{ mt: 2 }}>
                      {adaptedPerformanceData ? (
                        <PerformanceBreakdown 
                          taskId={taskId}
                          monthlyPerformance={adaptedPerformanceData.monthlyPerformance}
                          yearlyPerformance={adaptedPerformanceData.yearlyPerformance}
                          seasonalAnalysis={adaptedPerformanceData.seasonalAnalysis}
                          benchmarkComparison={adaptedPerformanceData.benchmarkComparison}
                        />
                      ) : (
                        <Box sx={{ textAlign: 'center', py: 4 }}>
                          <TrendingUp size={48} color="#999" style={{ margin: '0 auto 16px' }} />
                          <Typography variant="body2" color="text.secondary">绩效分解数据加载中...</Typography>
                        </Box>
                      )}
                    </Box>
                  )}
                  </Box>
                </Box>
              </CardContent>
            </Card>
          ) : (
            /* 预测任务的原有内容 */
            <>
              {/* 任务信息 */}
              <Card>
                <CardHeader title="任务信息" />
                <CardContent>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 2 }}>
                      <Box>
                        <Typography variant="caption" color="text.secondary">模型</Typography>
                        <Chip label={currentTask.model_id} color="secondary" size="small" sx={{ mt: 0.5 }} />
                      </Box>
                      <Box>
                        <Typography variant="caption" color="text.secondary">股票数量</Typography>
                        <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>{currentTask.stock_codes.length}</Typography>
                      </Box>
                      <Box>
                        <Typography variant="caption" color="text.secondary">创建时间</Typography>
                        <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>{new Date(currentTask.created_at).toLocaleString()}</Typography>
                      </Box>
                      {currentTask.completed_at && (
                        <Box>
                          <Typography variant="caption" color="text.secondary">完成时间</Typography>
                          <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>{new Date(currentTask.completed_at).toLocaleString()}</Typography>
                        </Box>
                      )}
                    </Box>

                    <Divider />

                    <Box>
                      <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>选择的股票</Typography>
                      <Box 
                        sx={{ 
                          height: 200,
                          overflow: 'hidden',
                          display: 'flex',
                          flexDirection: 'column',
                          border: '1px solid',
                          borderColor: 'divider',
                          borderRadius: 1,
                          p: 1.5
                        }}
                      >
                        {currentTask.stock_codes && currentTask.stock_codes.length > 0 ? (
                          <>
                            <Box 
                              sx={{ 
                                flex: 1,
                                overflowY: 'auto',
                                display: 'flex',
                                flexWrap: 'wrap',
                                gap: 1,
                                alignContent: 'flex-start',
                                pb: 1
                              }}
                            >
                              {(() => {
                                const STOCKS_PER_PAGE = 12;
                                const totalPages = Math.ceil(currentTask.stock_codes.length / STOCKS_PER_PAGE);
                                const startIndex = (selectedStocksPage - 1) * STOCKS_PER_PAGE;
                                const endIndex = startIndex + STOCKS_PER_PAGE;
                                const currentStocks = currentTask.stock_codes.slice(startIndex, endIndex);
                                
                                return currentStocks.map(code => (
                                  <Chip key={code} label={code} size="small" />
                                ));
                              })()}
                            </Box>
                            
                            {(() => {
                              const STOCKS_PER_PAGE = 12;
                              const totalPages = Math.ceil(currentTask.stock_codes.length / STOCKS_PER_PAGE);
                              
                              if (totalPages > 1) {
                                return (
                                  <Box sx={{ 
                                    display: 'flex', 
                                    justifyContent: 'center', 
                                    alignItems: 'center', 
                                    gap: 1,
                                    pt: 1,
                                    borderTop: '1px solid',
                                    borderColor: 'divider'
                                  }}>
                                    <IconButton
                                      size="small"
                                      disabled={selectedStocksPage === 1}
                                      onClick={() => setSelectedStocksPage(prev => Math.max(1, prev - 1))}
                                    >
                                      <ChevronLeft size={16} />
                                    </IconButton>
                                    
                                    <Typography variant="caption" color="text.secondary">
                                      第 {selectedStocksPage} / {totalPages} 页
                                    </Typography>
                                    
                                    <IconButton
                                      size="small"
                                      disabled={selectedStocksPage >= totalPages}
                                      onClick={() => setSelectedStocksPage(prev => Math.min(totalPages, prev + 1))}
                                    >
                                      <ChevronRight size={16} />
                                    </IconButton>
                                  </Box>
                                );
                              }
                              return null;
                            })()}
                            
                            <Box sx={{ 
                              pt: 1,
                              mt: 1,
                              borderTop: '1px solid',
                              borderColor: 'divider',
                              display: 'flex',
                              justifyContent: 'center'
                            }}>
                              <Typography variant="body2" color="text.secondary">
                                已选择 <strong>{currentTask.stock_codes.length}</strong> 只股票
                              </Typography>
                            </Box>
                          </>
                        ) : (
                          <Box sx={{ 
                            display: 'flex', 
                            alignItems: 'center', 
                            justifyContent: 'center', 
                            height: '100%' 
                          }}>
                            <Typography variant="body2" color="text.secondary">
                              暂无选择的股票
                            </Typography>
                          </Box>
                        )}
                      </Box>
                    </Box>
                  </Box>
                </CardContent>
              </Card>

              {/* 预测结果和图表 */}
              {currentTask.status === 'completed' && predictions.length > 0 && (
                <Card>
                  <CardHeader
                    title={
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                        <Typography variant="h6" component="h3" sx={{ fontWeight: 600 }}>
                          预测结果
                        </Typography>
                        <FormControl size="small" sx={{ minWidth: 192 }}>
                          <InputLabel>选择股票</InputLabel>
                          <Select
                            value={selectedStock || ''}
                            label="选择股票"
                            onChange={(e) => setSelectedStock(e.target.value)}
                          >
                            {predictions.map((prediction) => (
                              <MenuItem key={prediction.stock_code} value={prediction.stock_code}>
                                {prediction.stock_code}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      </Box>
                    }
                  />
                  <CardContent>
                    <Box>
                      <Tabs value={selectedPredictionTab} onChange={(e, newValue) => setSelectedPredictionTab(newValue)} aria-label="预测结果展示">
                        <Tab label={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <LineChart size={16} />
                            <span>价格走势</span>
                          </Box>
                        } value="chart" />
                        <Tab label={
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <BarChart3 size={16} />
                            <span>预测分析</span>
                          </Box>
                        } value="prediction" />
                        {currentTask.task_type === 'backtest' && (
                          <Tab label={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              <Activity size={16} />
                              <span>回测结果</span>
                            </Box>
                          } value="backtest" />
                        )}
                        <Tab label="数据表格" value="table" />
                      </Tabs>

                      <Box sx={{ mt: 2 }}>
                        {selectedPredictionTab === 'chart' && selectedStock && (
                          <TradingViewChart 
                            stockCode={selectedStock}
                            prediction={predictions.find(p => p.stock_code === selectedStock)}
                          />
                        )}

                        {selectedPredictionTab === 'prediction' && selectedStock && (
                          <PredictionChart 
                            taskId={taskId}
                            stockCode={selectedStock}
                            prediction={predictions.find(p => p.stock_code === selectedStock)}
                          />
                        )}

                        {selectedPredictionTab === 'backtest' && currentTask.task_type === 'backtest' && (
                          <BacktestChart 
                            stockCode={selectedStock || (currentTask?.stock_codes?.[0] || '')}
                            backtestData={(() => {
                              const data = currentTask?.results?.backtest_results || 
                                         currentTask?.backtest_results ||
                                         (currentTask?.task_type === 'backtest' ? currentTask?.result : null);
                              console.log('回测数据来源检查:', {
                                'results.backtest_results': currentTask?.results?.backtest_results,
                                'backtest_results': currentTask?.backtest_results,
                                'result': currentTask?.result,
                                '最终数据': data
                              });
                              return data;
                            })()}
                          />
                        )}

                        {selectedPredictionTab === 'table' && (
                          <TableContainer component={Paper}>
                            <Table aria-label="预测结果表格">
                              <TableHead>
                                <TableRow>
                                  <TableCell>股票代码</TableCell>
                                  <TableCell>预测方向</TableCell>
                                  <TableCell>预测收益率</TableCell>
                                  <TableCell>置信度</TableCell>
                                  <TableCell>置信区间</TableCell>
                                  <TableCell>VaR</TableCell>
                                </TableRow>
                              </TableHead>
                              <TableBody>
                                {predictions.map((prediction) => (
                                  <TableRow key={prediction.stock_code}>
                                    <TableCell>
                                      <Chip label={prediction.stock_code} size="small" />
                                    </TableCell>
                                    <TableCell>
                                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        {getPredictionIcon(prediction.predicted_direction)}
                                        <Typography variant="body2">{getPredictionText(prediction.predicted_direction)}</Typography>
                                      </Box>
                                    </TableCell>
                                    <TableCell>
                                      <Typography variant="body2" sx={{ 
                                        color: prediction.predicted_return > 0 
                                          ? 'success.main' 
                                          : prediction.predicted_return < 0 
                                            ? 'error.main' 
                                            : 'text.secondary'
                                      }}>
                                        {(prediction.predicted_return * 100).toFixed(2)}%
                                      </Typography>
                                    </TableCell>
                                    <TableCell>
                                      <Box sx={{ width: 80 }}>
                                        <LinearProgress 
                                          variant="determinate" 
                                          value={prediction.confidence_score * 100} 
                                          sx={{ height: 8, borderRadius: 4 }}
                                        />
                                      </Box>
                                    </TableCell>
                                    <TableCell>
                                      <Typography variant="caption" color="text.secondary">
                                        [{(prediction.confidence_interval.lower * 100).toFixed(2)}%, {(prediction.confidence_interval.upper * 100).toFixed(2)}%]
                                      </Typography>
                                    </TableCell>
                                    <TableCell>
                                      <Typography variant="body2" color="error.main">
                                        {(prediction.risk_assessment.value_at_risk * 100).toFixed(2)}%
                                      </Typography>
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </TableContainer>
                        )}
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </Box>

        {/* 侧边栏 */}
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
          {/* 根据任务类型显示不同的侧边栏内容 */}
          {currentTask.task_type === 'backtest' ? (
            /* 回测任务侧边栏 */
            <BacktestTaskStatus 
              task={currentTask}
              onRetry={handleRetry}
              onStop={() => {/* TODO: 实现停止功能 */}}
              loading={refreshing}
            />
          ) : (
            /* 预测任务侧边栏 */
            <>
              {/* 统计信息 */}
              {currentTask.results && (
                <Card>
                  <CardHeader title="统计信息" />
                  <CardContent>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="h4" sx={{ fontWeight: 600, color: 'primary.main' }}>
                          {currentTask.results.total_stocks}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">总股票数</Typography>
                      </Box>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="h4" sx={{ fontWeight: 600, color: 'success.main' }}>
                          {currentTask.results.successful_predictions}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">成功预测</Typography>
                      </Box>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="h4" sx={{ fontWeight: 600, color: 'secondary.main' }}>
                          {((currentTask.results.average_confidence || 0) * 100).toFixed(1)}%
                        </Typography>
                        <Typography variant="caption" color="text.secondary">平均置信度</Typography>
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              )}

              {/* 快速操作 */}
              <Card>
                <CardHeader title="快速操作" />
                <CardContent>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <Button
                      variant="outlined"
                      startIcon={<RefreshCw size={16} />}
                      onClick={handleRefresh}
                      disabled={refreshing}
                      fullWidth
                    >
                      刷新状态
                    </Button>
                    
                    {currentTask.status === 'failed' && (
                      <Button
                        variant="contained"
                        color="primary"
                        startIcon={<Play size={16} />}
                        onClick={handleRetry}
                        fullWidth
                      >
                        重新运行
                      </Button>
                    )}
                    
                    {currentTask.status === 'completed' && (
                      <Button
                        variant="outlined"
                        color="secondary"
                        startIcon={<Download size={16} />}
                        onClick={handleExport}
                        fullWidth
                      >
                        导出结果
                      </Button>
                    )}
                    
                    <Button
                      variant="outlined"
                      color="error"
                      startIcon={<Trash2 size={16} />}
                      onClick={onDeleteOpen}
                      fullWidth
                    >
                      删除任务
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </>
          )}
        </Box>
      </Box>

      {/* 保存策略配置对话框 */}
      {(() => {
        const configInfo = getStrategyConfig();
        if (configInfo) {
          return (
            <SaveStrategyConfigDialog
              isOpen={isSaveConfigOpen}
              onClose={onSaveConfigClose}
              strategyName={configInfo.strategyName}
              parameters={configInfo.parameters}
              onSave={handleSaveConfig}
              loading={savingConfig}
            />
          );
        }
        return null;
      })()}

      {/* 删除确认对话框 */}
      <Dialog open={isDeleteOpen} onClose={onDeleteClose}>
        <DialogTitle>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AlertTriangle size={20} color="#d32f2f" />
            <Typography variant="h6" component="span">确认删除</Typography>
          </Box>
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            确定要删除这个任务吗？此操作不可撤销。
          </Typography>
          {currentTask?.status === 'running' && (
            <Box sx={{ mt: 2, p: 2, bgcolor: 'warning.light', border: 1, borderColor: 'warning.main', borderRadius: 1 }}>
              <Typography variant="body2" sx={{ color: 'warning.dark', mb: 1 }}>
                ⚠️ 该任务当前正在运行中
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <input
                  type="checkbox"
                  checked={deleteForce}
                  onChange={(e) => setDeleteForce(e.target.checked)}
                  style={{ width: 16, height: 16 }}
                />
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  强制删除（将中断正在运行的任务）
                </Typography>
              </Box>
            </Box>
          )}
        </DialogContent>
        <DialogActions>
          <Button variant="outlined" onClick={() => {
            setDeleteForce(false);
            onDeleteClose();
          }}>
            取消
          </Button>
          <Button variant="contained" color="error" onClick={() => {
            handleDelete();
            onDeleteClose();
          }}>
            {deleteForce ? '强制删除' : '删除'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
