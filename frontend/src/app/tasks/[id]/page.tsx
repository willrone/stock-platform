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

import React, { useEffect, useState } from 'react';
import {
  Card,
  CardHeader,
  CardContent,
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
  Copy,
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
import { SaveStrategyConfigDialog } from '../../../components/backtest/SaveStrategyConfigDialog';
import { StrategyConfigService } from '../../../services/strategyConfigService';
import dynamic from 'next/dynamic';

// 动态导入图表组件
const TradingViewChart = dynamic(() => import('../../../components/charts/TradingViewChart'), {
  ssr: false,
  loading: () => <div className="h-96 flex items-center justify-center">加载图表中...</div>,
});

const PredictionChart = dynamic(() => import('../../../components/charts/PredictionChart'), {
  ssr: false,
  loading: () => <div className="h-64 flex items-center justify-center">加载预测图表中...</div>,
});

const BacktestChart = dynamic(() => import('../../../components/charts/BacktestChart'), {
  ssr: false,
  loading: () => <div className="h-64 flex items-center justify-center">加载回测图表中...</div>,
});

const InteractiveChartsContainer = dynamic(
  () => import('../../../components/charts/InteractiveChartsContainer'),
  {
    ssr: false,
    loading: () => <div className="h-96 flex items-center justify-center">加载交互式图表中...</div>,
  }
);

// 依赖 ECharts 的组件需客户端加载，避免服务端 vendor-chunks/echarts.js 报错
const PositionAnalysis = dynamic(
  () => import('../../../components/backtest/PositionAnalysis').then(mod => ({ default: mod.PositionAnalysis })),
  { ssr: false, loading: () => <div className="h-64 flex items-center justify-center">加载持仓分析中...</div> }
);
const RiskAnalysis = dynamic(
  () => import('../../../components/backtest/RiskAnalysis').then(mod => ({ default: mod.RiskAnalysis })),
  { ssr: false, loading: () => <div className="h-64 flex items-center justify-center">加载风险分析中...</div> }
);
const PerformanceBreakdown = dynamic(
  () => import('../../../components/backtest/PerformanceBreakdown').then(mod => ({ default: mod.PerformanceBreakdown })),
  { ssr: false, loading: () => <div className="h-64 flex items-center justify-center">加载绩效分析中...</div> }
);

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
        status: task?.status,
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
      console.log(
        '[TaskDetail] position_analysis 是否为数组:',
        Array.isArray(detailedResult?.position_analysis)
      );
      if (
        detailedResult?.position_analysis &&
        typeof detailedResult.position_analysis === 'object' &&
        !Array.isArray(detailedResult.position_analysis)
      ) {
        console.log(
          '[TaskDetail] position_analysis.stock_performance:',
          detailedResult.position_analysis.stock_performance
        );
        console.log(
          '[TaskDetail] stock_performance 长度:',
          detailedResult.position_analysis.stock_performance?.length
        );
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
        rollingMetrics,
      });
      console.log('[TaskDetail] 风险分析数据适配完成:', {
        riskMetrics,
        returnDistribution,
        rollingMetrics,
      });

      console.log('[TaskDetail] 开始适配绩效分解数据...');
      const monthlyPerformance = BacktestDataAdapter.adaptMonthlyPerformance(detailedResult);
      const yearlyPerformance = BacktestDataAdapter.generateYearlyPerformance(detailedResult);
      const seasonalAnalysis = BacktestDataAdapter.generateSeasonalAnalysis(detailedResult);
      const benchmarkComparison = BacktestDataAdapter.generateBenchmarkComparison(detailedResult);

      setAdaptedPerformanceData({
        monthlyPerformance,
        yearlyPerformance,
        seasonalAnalysis,
        benchmarkComparison,
      });
      console.log('[TaskDetail] 绩效分解数据适配完成:', {
        monthlyPerformance,
        yearlyPerformance,
        seasonalAnalysis,
        benchmarkComparison,
      });
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
        result: task.result,
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
            backtest_results: task.backtest_results,
            result: task.result,
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
    if (
      currentTask &&
      currentTask.task_type === 'backtest' &&
      currentTask.status === 'completed' &&
      !hasTriggeredLoadRef.current
    ) {
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
      if (
        error.message?.includes('正在运行中') ||
        error.message?.includes('运行中') ||
        currentTask?.status === 'running'
      ) {
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

  // 重建任务
  const handleRebuild = () => {
    if (!currentTask) return;

    // 构建配置参数
    const params = new URLSearchParams();
    params.set('rebuild', 'true');
    params.set('task_type', currentTask.task_type || 'backtest');
    params.set('task_name', `${currentTask.task_name} (重建)`);
    
    // 股票代码
    if (currentTask.stock_codes && currentTask.stock_codes.length > 0) {
      params.set('stock_codes', currentTask.stock_codes.join(','));
    }

    if (currentTask.task_type === 'backtest') {
      // 回测任务配置 - 优先从 result.backtest_config 读取，其次从 config.backtest_config
      const resultBc = currentTask.result?.backtest_config;
      const configBc = currentTask.config?.backtest_config;
      const bc = resultBc || configBc;
      
      if (bc) {
        if (bc.strategy_name) params.set('strategy_name', bc.strategy_name);
        
        // 日期格式转换：从 ISO 格式转为 YYYY-MM-DD
        if (bc.start_date) {
          const startDate = bc.start_date.split('T')[0];
          params.set('start_date', startDate);
        }
        if (bc.end_date) {
          const endDate = bc.end_date.split('T')[0];
          params.set('end_date', endDate);
        }
        
        if (bc.initial_cash !== undefined) params.set('initial_cash', bc.initial_cash.toString());
        if (bc.commission_rate !== undefined) params.set('commission_rate', bc.commission_rate.toString());
        if (bc.slippage_rate !== undefined) params.set('slippage_rate', bc.slippage_rate.toString());
        if (bc.enable_performance_profiling !== undefined) params.set('enable_performance_profiling', bc.enable_performance_profiling.toString());
        
        // 策略配置
        if (bc.strategy_config) {
          params.set('strategy_config', JSON.stringify(bc.strategy_config));
        }
      }
      
      // 跳转到回测创建页面
      router.push(`/tasks/create?${params.toString()}`);
    } else if (currentTask.task_type === 'hyperparameter_optimization') {
      // 超参优化任务��置
      const cfg = currentTask.config;
      const optConfig = cfg?.optimization_config;
      
      // 策略名称（在 optimization_config 中）
      if (optConfig?.strategy_name) {
        params.set('strategy_name', optConfig.strategy_name);
      }
      
      // 日期范围（在 config 顶层）
      if (cfg?.start_date) {
        const startDate = cfg.start_date.split('T')[0];
        params.set('start_date', startDate);
      }
      if (cfg?.end_date) {
        const endDate = cfg.end_date.split('T')[0];
        params.set('end_date', endDate);
      }
      
      // 优化配置（从 objective_config 中读取）
      const objConfig = optConfig?.objective_config;
      if (objConfig?.objective_metric) {
        params.set('objective_metric', Array.isArray(objConfig.objective_metric) 
          ? objConfig.objective_metric.join(',') 
          : objConfig.objective_metric);
      }
      if (objConfig?.direction) params.set('direction', objConfig.direction);
      if (optConfig?.n_trials !== undefined) params.set('n_trials', optConfig.n_trials.toString());
      if (optConfig?.optimization_method) params.set('optimization_method', optConfig.optimization_method);
      if (optConfig?.timeout !== undefined && optConfig.timeout !== null) {
        params.set('timeout', optConfig.timeout.toString());
      }
      
      // 参数空间配置
      if (optConfig?.param_space) {
        params.set('param_space', JSON.stringify(optConfig.param_space));
      }
      
      // 跳转到优化任务创建页面
      router.push(`/optimization/create?${params.toString()}`);
    } else if (currentTask.task_type === 'prediction') {
      // 预测任务配置
      if (currentTask.model_id) params.set('model_id', currentTask.model_id);
      
      const predConfig = currentTask.config?.prediction_config;
      if (predConfig) {
        if (predConfig.horizon) params.set('horizon', predConfig.horizon);
        if (predConfig.confidence_level !== undefined) {
          // 注意：存储为小数（0.95），需要转换为百分比（95）
          params.set('confidence_level', (predConfig.confidence_level * 100).toString());
        }
        if (predConfig.risk_assessment !== undefined) {
          params.set('risk_assessment', predConfig.risk_assessment.toString());
        }
      }
      
      // 跳转到预测任务创建页面
      router.push(`/tasks/create?${params.toString()}`);
    }
  };

  // 获取策略配置信息（支持 config 嵌套 backtest_config 或扁平 strategy_name/strategy_config）
  const getStrategyConfig = () => {
    if (!currentTask || currentTask.task_type !== 'backtest') {
      return null;
    }

    const cfg = currentTask.config;
    const bc = cfg?.backtest_config;
    const backtestData =
      currentTask.result || currentTask.results?.backtest_results || currentTask.backtest_results;
    const resultBc = backtestData?.backtest_config;

    // 调试日志：检查各个位置的策略配置
    console.log('[getStrategyConfig] 调试信息:', {
      cfg: cfg,
      'cfg?.backtest_config': bc,
      'cfg?.strategy_config': cfg?.strategy_config,
      backtestData: backtestData,
      resultBc: resultBc,
      'resultBc?.strategy_config': resultBc?.strategy_config,
    });

    let strategyName =
      bc?.strategy_name ??
      cfg?.strategy_name ??
      resultBc?.strategy_name ??
      (backtestData as any)?.strategy_name ??
      '未知策略';

    const parameters: Record<string, any> =
      bc?.strategy_config != null
        ? bc.strategy_config
        : cfg?.strategy_config != null
          ? cfg.strategy_config
          : resultBc?.strategy_config != null
            ? resultBc.strategy_config
            : {};

    if (strategyName === '未知策略' && Array.isArray((parameters as any)?.strategies)) {
      strategyName = 'portfolio';
    }

    console.log('[getStrategyConfig] 最终结果:', {
      strategyName,
      parameters,
      parametersKeys: Object.keys(parameters),
    });

    return { strategyName, parameters };
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

  const getStrategyDisplayName = (strategyName: string) => {
    return strategyName === 'portfolio' ? '组合策略' : strategyName;
  };

  // 组合策略在未配置子策略时与后端默认一致，用于展示
  const DEFAULT_PORTFOLIO_STRATEGIES = [
    { name: 'bollinger', weight: 1, config: { period: 20, std_dev: 2, entry_threshold: 0.02 } },
    { name: 'cci', weight: 1, config: { period: 20, oversold: -100, overbought: 100 } },
    { name: 'macd', weight: 1, config: { fast_period: 12, slow_period: 26, signal_period: 9 } },
  ];

  const renderStrategyParameters = (parameters: Record<string, any>) => {
    const raw = Array.isArray(parameters.strategies) ? parameters.strategies : null;
    // 组合策略子列表为空时用默认展示，避免显示「0 个」；非组合策略（raw 为 null）不进入本块
    const strategies =
      raw === null ? null : raw.length > 0 ? raw : DEFAULT_PORTFOLIO_STRATEGIES;

    if (strategies && strategies.length > 0) {
      return (
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 1 }}>
            <Chip
              size="small"
              color="secondary"
              label={`组合策略 · ${strategies.length} 个${raw?.length === 0 ? '（默认）' : ''}`}
            />
            <Chip
              size="small"
              variant="outlined"
              label={`信号整合: ${parameters.integration_method || 'weighted_voting'}`}
            />
          </Box>
          <Box
            sx={{
              display: 'grid',
              gap: 2,
              gridTemplateColumns: { xs: '1fr', md: 'repeat(2, minmax(0, 1fr))' },
            }}
          >
            {strategies.map((strategy: any, index: number) => (
              <Box
                key={`${strategy?.name || 'strategy'}-${index}`}
                sx={{
                  border: 1,
                  borderColor: 'divider',
                  borderRadius: 2,
                  p: 2,
                  bgcolor: 'background.paper',
                  boxShadow: '0 4px 14px rgba(15, 23, 42, 0.06)',
                }}
              >
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    mb: 1,
                  }}
                >
                  <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
                    {strategy?.name || `策略${index + 1}`}
                  </Typography>
                  <Chip
                    size="small"
                    color="primary"
                    label={`权重 ${
                      typeof strategy?.weight === 'number'
                        ? strategy.weight.toFixed(2)
                        : strategy?.weight ?? '-'
                    }`}
                  />
                </Box>
                {strategy?.config && Object.keys(strategy.config).length > 0 ? (
                  <Box
                    component="pre"
                    sx={{
                      fontSize: '0.75rem',
                      color: 'text.secondary',
                      whiteSpace: 'pre-wrap',
                      fontFamily: 'monospace',
                      m: 0,
                      p: 1.5,
                      borderRadius: 1,
                      bgcolor: 'grey.50',
                      border: 1,
                      borderColor: 'divider',
                      maxHeight: 200,
                      overflow: 'auto',
                    }}
                  >
                    {JSON.stringify(strategy.config, null, 2)}
                  </Box>
                ) : (
                  <Box
                    sx={{
                      borderRadius: 1,
                      bgcolor: 'grey.50',
                      border: 1,
                      borderColor: 'divider',
                      p: 1.5,
                    }}
                  >
                    <Typography variant="caption" color="text.secondary">
                      暂无参数
                    </Typography>
                  </Box>
                )}
              </Box>
            ))}
          </Box>
        </Box>
      );
    }

    return (
      <Box sx={{ bgcolor: 'grey.100', borderRadius: 1, p: 1.5 }}>
        <Box
          component="pre"
          sx={{
            fontSize: '0.75rem',
            color: 'text.secondary',
            whiteSpace: 'pre-wrap',
            fontFamily: 'monospace',
            m: 0,
          }}
        >
          {Object.entries(parameters)
            .map(([key, value]) => {
              if (typeof value === 'object' && value !== null) {
                return `${key}: ${JSON.stringify(value, null, 2)}`;
              }
              return `${key}: ${value}`;
            })
            .join('\n')}
        </Box>
      </Box>
    );
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
    if (direction > 0) {
      return <TrendingUp className="w-4 h-4 text-success" />;
    }
    if (direction < 0) {
      return <TrendingDown className="w-4 h-4 text-danger" />;
    }
    return <Minus className="w-4 h-4 text-default-500" />;
  };

  // 获取预测方向文本
  const getPredictionText = (direction: number) => {
    if (direction > 0) {
      return '上涨';
    }
    if (direction < 0) {
      return '下跌';
    }
    return '持平';
  };

  if (loading) {
    return <LoadingSpinner text="加载任务详情..." />;
  }

  if (!currentTask) {
    return (
      <Box
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: 384,
          gap: 2,
        }}
      >
        <Typography variant="body2" color="text.secondary">
          任务不存在或已被删除
        </Typography>
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
            <Typography variant="caption" color="text.secondary">
              任务ID: {currentTask.task_id}
            </Typography>
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

          <Button
            variant="outlined"
            color="secondary"
            startIcon={<Copy size={16} />}
            onClick={handleRebuild}
          >
            重建任务
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
          {currentTask.task_type === 'backtest' &&
          (currentTask.status === 'running' || currentTask.status === 'created') ? (
            <BacktestProgressMonitor
              taskId={taskId}
              onComplete={results => {
                console.log('回测完成:', results);
                // 刷新任务数据
                loadTaskDetail();
              }}
              onError={error => {
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
                {currentTask.task_type === 'hyperparameter_optimization' &&
                  currentTask.optimization_info && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        已完成轮次: {currentTask.optimization_info.completed_trials} /{' '}
                        {currentTask.optimization_info.n_trials}
                      </Typography>
                    </Box>
                  )}
                {currentTask.status === 'running' && (
                  <Typography variant="caption" color="text.secondary">
                    任务正在执行中，请耐心等待...
                  </Typography>
                )}
                {currentTask.status === 'failed' && currentTask.error_message && (
                  <Box
                    sx={{
                      bgcolor: 'error.light',
                      border: 1,
                      borderColor: 'error.main',
                      borderRadius: 1,
                      p: 2,
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                      <AlertTriangle size={20} color="#d32f2f" style={{ marginTop: 2 }} />
                      <Box>
                        <Typography variant="body2" sx={{ fontWeight: 500, color: 'error.dark' }}>
                          任务执行失败
                        </Typography>
                        <Typography
                          variant="caption"
                          sx={{ color: 'error.dark', mt: 0.5, display: 'block' }}
                        >
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
                      if (
                        tabKey === 'positions' &&
                        currentTask &&
                        currentTask.task_type === 'backtest' &&
                        currentTask.status === 'completed' &&
                        !backtestDetailedData &&
                        !loadingBacktestData
                      ) {
                        console.log('[TaskDetail] 切换到持仓分析页签，触发数据加载');
                        loadBacktestDetailedData();
                      }
                    }}
                    aria-label="回测结果展示"
                    variant="scrollable"
                    scrollButtons="auto"
                  >
                    <Tab
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <BarChart3 size={16} />
                          <span>概览</span>
                        </Box>
                      }
                      value="overview"
                    />
                    <Tab
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <LineChart size={16} />
                          <span>交互式图表</span>
                        </Box>
                      }
                      value="charts"
                    />
                    <Tab
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <FileText size={16} />
                          <span>交易记录</span>
                        </Box>
                      }
                      value="trades"
                    />
                    <Tab
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <AlertTriangle size={16} />
                          <span>信号记录</span>
                        </Box>
                      }
                      value="signals"
                    />
                    <Tab
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <PieChart size={16} />
                          <span>持仓分析</span>
                        </Box>
                      }
                      value="positions"
                    />
                    <Tab
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <Calendar size={16} />
                          <span>月度分析</span>
                        </Box>
                      }
                      value="monthly"
                    />
                    <Tab
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <Activity size={16} />
                          <span>风险分析</span>
                        </Box>
                      }
                      value="risk"
                    />
                    <Tab
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <TrendingUp size={16} />
                          <span>绩效分解</span>
                        </Box>
                      }
                      value="performance"
                    />
                    <Tab
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                          <Activity size={16} />
                          <span>性能分析</span>
                        </Box>
                      }
                      value="perf_monitor"
                    />
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
                                    <Box
                                      sx={{
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        alignItems: 'center',
                                        width: '100%',
                                      }}
                                    >
                                      <Box>
                                        <Typography
                                          variant="h6"
                                          component="h4"
                                          sx={{ fontWeight: 600 }}
                                        >
                                          策略配置
                                        </Typography>
                                        <Typography variant="caption" color="text.secondary">
                                          策略: {getStrategyDisplayName(configInfo.strategyName)}
                                        </Typography>
                                      </Box>
                                      <Button
                                        variant="outlined"
                                        color="primary"
                                        size="small"
                                        startIcon={<Save size={16} />}
                                        onClick={onSaveConfigOpen}
                                        disabled={
                                          !configInfo.strategyName ||
                                          configInfo.strategyName === '未知策略' ||
                                          Object.keys(configInfo.parameters).length === 0
                                        }
                                      >
                                        保存配置
                                      </Button>
                                    </Box>
                                  }
                                />
                                <CardContent>
                                  {Object.keys(configInfo.parameters).length > 0 ? (
                                    renderStrategyParameters(configInfo.parameters)
                                  ) : (
                                    <Typography variant="caption" color="text.secondary">
                                      暂无策略参数配置
                                    </Typography>
                                  )}
                                </CardContent>
                              </Card>
                            );
                          }
                          return null;
                        })()}
                        <BacktestOverview
                          backtestData={
                            currentTask.result ||
                            currentTask.results?.backtest_results ||
                            currentTask.backtest_results
                          }
                          loading={loadingBacktestData}
                        />
                        <CostAnalysis
                          backtestData={
                            currentTask.result ||
                            currentTask.results?.backtest_results ||
                            currentTask.backtest_results
                          }
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
                            const data =
                              currentTask?.results?.backtest_results ||
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
                          onTradeClick={trade => {
                            console.log('查看交易详情:', trade);
                          }}
                        />
                      </Box>
                    )}

                    {selectedBacktestTab === 'signals' && (
                      <Box sx={{ mt: 2 }}>
                        <SignalHistoryTable
                          taskId={taskId}
                          onSignalClick={signal => {
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
                            <Box
                              sx={{
                                display: 'grid',
                                gridTemplateColumns: 'repeat(12, 1fr)',
                                gap: 0.5,
                              }}
                            >
                              {backtestDetailedData.monthly_returns.map((monthData: any) => (
                                <Box
                                  key={`${monthData.year}-${monthData.month}`}
                                  sx={{
                                    p: 1,
                                    textAlign: 'center',
                                    fontSize: '0.75rem',
                                    borderRadius: 1,
                                    bgcolor:
                                      monthData.monthly_return >= 0
                                        ? 'success.light'
                                        : 'error.light',
                                    color:
                                      monthData.monthly_return >= 0 ? 'success.dark' : 'error.dark',
                                  }}
                                  title={`${monthData.year}年${monthData.month}月: ${(
                                    monthData.monthly_return * 100
                                  ).toFixed(2)}%`}
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
                            <Typography variant="body2" color="text.secondary">
                              月度分析数据加载中...
                            </Typography>
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
                            <Typography variant="body2" color="text.secondary">
                              风险分析数据加载中...
                            </Typography>
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
                            <Typography variant="body2" color="text.secondary">
                              绩效分解数据加载中...
                            </Typography>
                          </Box>
                        )}
                      </Box>
                    )}

                    {selectedBacktestTab === 'perf_monitor' && (
                      <Box sx={{ mt: 2 }}>
                        {(() => {
                          const backtestData =
                            currentTask.result ||
                            currentTask.results?.backtest_results ||
                            currentTask.backtest_results;
                          const perf = backtestData?.performance_analysis as any;

                          if (!backtestData) {
                            return (
                              <Box sx={{ textAlign: 'center', py: 4 }}>
                                <Activity
                                  size={48}
                                  color="#999"
                                  style={{ margin: '0 auto 16px' }}
                                />
                                <Typography variant="body2" color="text.secondary">
                                  暂无回测结果数据，无法展示性能分析。
                                </Typography>
                              </Box>
                            );
                          }

                          if (!perf) {
                            return (
                              <Box sx={{ textAlign: 'center', py: 4 }}>
                                <Activity
                                  size={48}
                                  color="#999"
                                  style={{ margin: '0 auto 16px' }}
                                />
                                <Typography variant="body2" color="text.secondary">
                                  当前回测未启用性能监控，或后端尚未写入性能报告。
                                </Typography>
                                <Typography
                                  variant="caption"
                                  color="text.secondary"
                                  sx={{ display: 'block', mt: 1 }}
                                >
                                  请确认后端创建回测执行器时已开启
                                  enable_performance_profiling，并返回 performance_analysis 字段。
                                </Typography>
                              </Box>
                            );
                          }

                          const summary = perf.summary || {};
                          const stages = perf.stages || {};
                          const functionCalls = perf.function_calls || {};
                          const parallel = perf.parallel_efficiency || {};

                          return (
                            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                              {/* 总览卡片 */}
                              <Card>
                                <CardHeader title="整体性能概要" />
                                <CardContent>
                                  <Box
                                    sx={{
                                      display: 'grid',
                                      gridTemplateColumns: { xs: '1fr', md: 'repeat(3, 1fr)' },
                                      gap: 2,
                                    }}
                                  >
                                    <Box>
                                      <Typography variant="caption" color="text.secondary">
                                        总执行时间
                                      </Typography>
                                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                                        {(summary.total_time || 0).toFixed(2)} 秒
                                      </Typography>
                                    </Box>
                                    <Box>
                                      <Typography variant="caption" color="text.secondary">
                                        总信号数 / 交易数
                                      </Typography>
                                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                                        {(summary.total_signals || 0).toLocaleString()} /{' '}
                                        {(summary.total_trades || 0).toLocaleString()}
                                      </Typography>
                                    </Box>
                                    <Box>
                                      <Typography variant="caption" color="text.secondary">
                                        处理速度
                                      </Typography>
                                      <Typography variant="h6" sx={{ fontWeight: 600 }}>
                                        {(summary.days_per_second || 0).toFixed(2)} 天/秒
                                      </Typography>
                                    </Box>
                                  </Box>
                                </CardContent>
                              </Card>

                              {/* 阶段耗时表 */}
                              <Card>
                                <CardHeader title="阶段耗时与资源占用" />
                                <CardContent>
                                  <TableContainer component={Paper}>
                                    <Table size="small">
                                      <TableHead>
                                        <TableRow>
                                          <TableCell>阶段</TableCell>
                                          <TableCell align="right">耗时 (秒)</TableCell>
                                          <TableCell align="right">占比</TableCell>
                                          <TableCell align="right">峰值内存 (MB)</TableCell>
                                          <TableCell align="right">平均 CPU (%)</TableCell>
                                        </TableRow>
                                      </TableHead>
                                      <TableBody>
                                        {Object.entries(stages).map(
                                          ([name, data]: [string, any]) => (
                                            <TableRow key={name}>
                                              <TableCell>
                                                <Typography variant="body2">
                                                  {name === 'total_backtest' ? '整体回测' : name}
                                                </Typography>
                                              </TableCell>
                                              <TableCell align="right">
                                                {(data.duration || 0).toFixed(2)}
                                              </TableCell>
                                              <TableCell align="right">
                                                {(data.percentage || 0).toFixed(1)}%
                                              </TableCell>
                                              <TableCell align="right">
                                                {(
                                                  data.memory_peak_mb ??
                                                  data.memory_after_mb ??
                                                  0
                                                ).toFixed(2)}
                                              </TableCell>
                                              <TableCell align="right">
                                                {(data.cpu_avg_percent || 0).toFixed(1)}
                                              </TableCell>
                                            </TableRow>
                                          )
                                        )}
                                      </TableBody>
                                    </Table>
                                  </TableContainer>
                                </CardContent>
                              </Card>

                              {/* 函数调用 Top N */}
                              {Object.keys(functionCalls).length > 0 && (
                                <Card>
                                  <CardHeader title="最耗时的函数 (Top 10)" />
                                  <CardContent>
                                    <TableContainer component={Paper}>
                                      <Table size="small">
                                        <TableHead>
                                          <TableRow>
                                            <TableCell>函数名</TableCell>
                                            <TableCell align="right">调用次数</TableCell>
                                            <TableCell align="right">总耗时 (秒)</TableCell>
                                            <TableCell align="right">平均耗时 (毫秒)</TableCell>
                                          </TableRow>
                                        </TableHead>
                                        <TableBody>
                                          {Object.entries(functionCalls)
                                            .slice(0, 10)
                                            .map(([name, data]: [string, any]) => (
                                              <TableRow key={name}>
                                                <TableCell>{name}</TableCell>
                                                <TableCell align="right">
                                                  {data.call_count || 0}
                                                </TableCell>
                                                <TableCell align="right">
                                                  {(data.total_time || 0).toFixed(4)}
                                                </TableCell>
                                                <TableCell align="right">
                                                  {((data.avg_time || 0) * 1000).toFixed(2)}
                                                </TableCell>
                                              </TableRow>
                                            ))}
                                        </TableBody>
                                      </Table>
                                    </TableContainer>
                                  </CardContent>
                                </Card>
                              )}

                              {/* 并行化效率 */}
                              {Object.keys(parallel).length > 0 && (
                                <Card>
                                  <CardHeader title="并行化效率" />
                                  <CardContent>
                                    <TableContainer component={Paper}>
                                      <Table size="small">
                                        <TableHead>
                                          <TableRow>
                                            <TableCell>操作</TableCell>
                                            <TableCell align="right">顺序时间 (秒)</TableCell>
                                            <TableCell align="right">并行时间 (秒)</TableCell>
                                            <TableCell align="right">加速比</TableCell>
                                            <TableCell align="right">效率 (%)</TableCell>
                                            <TableCell align="right">Worker 数</TableCell>
                                          </TableRow>
                                        </TableHead>
                                        <TableBody>
                                          {Object.entries(parallel).map(
                                            ([name, data]: [string, any]) => (
                                              <TableRow key={name}>
                                                <TableCell>{name}</TableCell>
                                                <TableCell align="right">
                                                  {(data.sequential_time || 0).toFixed(4)}
                                                </TableCell>
                                                <TableCell align="right">
                                                  {(data.parallel_time || 0).toFixed(4)}
                                                </TableCell>
                                                <TableCell align="right">
                                                  {(data.speedup || 0).toFixed(2)}x
                                                </TableCell>
                                                <TableCell align="right">
                                                  {(data.efficiency_percent || 0).toFixed(1)}
                                                </TableCell>
                                                <TableCell align="right">
                                                  {data.worker_count || 0}
                                                </TableCell>
                                              </TableRow>
                                            )
                                          )}
                                        </TableBody>
                                      </Table>
                                    </TableContainer>
                                  </CardContent>
                                </Card>
                              )}
                            </Box>
                          );
                        })()}
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
                      {currentTask.task_type === 'hyperparameter_optimization' ? (
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            已完成轮次
                          </Typography>
                          <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                            {currentTask.optimization_info?.completed_trials ?? 0} /{' '}
                            {currentTask.optimization_info?.n_trials ?? 0}
                          </Typography>
                        </Box>
                      ) : (
                        <>
                          <Box>
                            <Typography variant="caption" color="text.secondary">
                              模型
                            </Typography>
                            <Chip
                              label={currentTask.model_id}
                              color="secondary"
                              size="small"
                              sx={{ mt: 0.5 }}
                            />
                          </Box>
                          <Box>
                            <Typography variant="caption" color="text.secondary">
                              股票数量
                            </Typography>
                            <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                              {currentTask.stock_codes.length}
                            </Typography>
                          </Box>
                        </>
                      )}
                      <Box>
                        <Typography variant="caption" color="text.secondary">
                          创建时间
                        </Typography>
                        <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                          {new Date(currentTask.created_at).toLocaleString()}
                        </Typography>
                      </Box>
                      {currentTask.completed_at && (
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            完成时间
                          </Typography>
                          <Typography variant="body2" sx={{ fontWeight: 500, mt: 0.5 }}>
                            {new Date(currentTask.completed_at).toLocaleString()}
                          </Typography>
                        </Box>
                      )}
                    </Box>

                    <Divider />

                    <Box>
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{ mb: 1, display: 'block' }}
                      >
                        选择的股票
                      </Typography>
                      <Box
                        sx={{
                          height: 200,
                          overflow: 'hidden',
                          display: 'flex',
                          flexDirection: 'column',
                          border: '1px solid',
                          borderColor: 'divider',
                          borderRadius: 1,
                          p: 1.5,
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
                                pb: 1,
                              }}
                            >
                              {(() => {
                                const STOCKS_PER_PAGE = 12;
                                const startIndex = (selectedStocksPage - 1) * STOCKS_PER_PAGE;
                                const endIndex = startIndex + STOCKS_PER_PAGE;
                                const currentStocks = currentTask.stock_codes.slice(
                                  startIndex,
                                  endIndex
                                );

                                return currentStocks.map(code => (
                                  <Chip key={code} label={code} size="small" />
                                ));
                              })()}
                            </Box>

                            {(() => {
                              const STOCKS_PER_PAGE = 12;
                              const totalPages = Math.ceil(
                                currentTask.stock_codes.length / STOCKS_PER_PAGE
                              );

                              if (totalPages > 1) {
                                return (
                                  <Box
                                    sx={{
                                      display: 'flex',
                                      justifyContent: 'center',
                                      alignItems: 'center',
                                      gap: 1,
                                      pt: 1,
                                      borderTop: '1px solid',
                                      borderColor: 'divider',
                                    }}
                                  >
                                    <IconButton
                                      size="small"
                                      disabled={selectedStocksPage === 1}
                                      onClick={() =>
                                        setSelectedStocksPage(prev => Math.max(1, prev - 1))
                                      }
                                    >
                                      <ChevronLeft size={16} />
                                    </IconButton>

                                    <Typography variant="caption" color="text.secondary">
                                      第 {selectedStocksPage} / {totalPages} 页
                                    </Typography>

                                    <IconButton
                                      size="small"
                                      disabled={selectedStocksPage >= totalPages}
                                      onClick={() =>
                                        setSelectedStocksPage(prev =>
                                          Math.min(totalPages, prev + 1)
                                        )
                                      }
                                    >
                                      <ChevronRight size={16} />
                                    </IconButton>
                                  </Box>
                                );
                              }
                              return null;
                            })()}

                            <Box
                              sx={{
                                pt: 1,
                                mt: 1,
                                borderTop: '1px solid',
                                borderColor: 'divider',
                                display: 'flex',
                                justifyContent: 'center',
                              }}
                            >
                              <Typography variant="body2" color="text.secondary">
                                已选择 <strong>{currentTask.stock_codes.length}</strong> 只股票
                              </Typography>
                            </Box>
                          </>
                        ) : (
                          <Box
                            sx={{
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              height: '100%',
                            }}
                          >
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
                      <Box
                        sx={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          width: '100%',
                        }}
                      >
                        <Typography variant="h6" component="h3" sx={{ fontWeight: 600 }}>
                          预测结果
                        </Typography>
                        <FormControl size="small" sx={{ minWidth: 192 }}>
                          <InputLabel>选择股票</InputLabel>
                          <Select
                            value={selectedStock || ''}
                            label="选择股票"
                            onChange={e => setSelectedStock(e.target.value)}
                          >
                            {predictions.map(prediction => (
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
                      <Tabs
                        value={selectedPredictionTab}
                        onChange={(e, newValue) => setSelectedPredictionTab(newValue)}
                        aria-label="预测结果展示"
                      >
                        <Tab
                          label={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              <LineChart size={16} />
                              <span>价格走势</span>
                            </Box>
                          }
                          value="chart"
                        />
                        <Tab
                          label={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                              <BarChart3 size={16} />
                              <span>预测分析</span>
                            </Box>
                          }
                          value="prediction"
                        />
                        {currentTask.task_type === 'backtest' && (
                          <Tab
                            label={
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                                <Activity size={16} />
                                <span>回测结果</span>
                              </Box>
                            }
                            value="backtest"
                          />
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

                        {selectedPredictionTab === 'backtest' &&
                          currentTask.task_type === 'backtest' && (
                            <BacktestChart
                              stockCode={selectedStock || currentTask?.stock_codes?.[0] || ''}
                              backtestData={(() => {
                                const data =
                                  currentTask?.results?.backtest_results ||
                                  currentTask?.backtest_results ||
                                  (currentTask?.task_type === 'backtest'
                                    ? currentTask?.result
                                    : null);
                                console.log('回测数据来源检查:', {
                                  'results.backtest_results':
                                    currentTask?.results?.backtest_results,
                                  backtest_results: currentTask?.backtest_results,
                                  result: currentTask?.result,
                                  最终数据: data,
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
                                {predictions.map(prediction => (
                                  <TableRow key={prediction.stock_code}>
                                    <TableCell>
                                      <Chip label={prediction.stock_code} size="small" />
                                    </TableCell>
                                    <TableCell>
                                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                        {getPredictionIcon(prediction.predicted_direction)}
                                        <Typography variant="body2">
                                          {getPredictionText(prediction.predicted_direction)}
                                        </Typography>
                                      </Box>
                                    </TableCell>
                                    <TableCell>
                                      <Typography
                                        variant="body2"
                                        sx={{
                                          color:
                                            prediction.predicted_return > 0
                                              ? 'success.main'
                                              : prediction.predicted_return < 0
                                                ? 'error.main'
                                                : 'text.secondary',
                                        }}
                                      >
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
                                        [{(prediction.confidence_interval.lower * 100).toFixed(2)}%,{' '}
                                        {(prediction.confidence_interval.upper * 100).toFixed(2)}%]
                                      </Typography>
                                    </TableCell>
                                    <TableCell>
                                      <Typography variant="body2" color="error.main">
                                        {(prediction.risk_assessment.value_at_risk * 100).toFixed(
                                          2
                                        )}
                                        %
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
              onStop={() => {
                /* TODO: 实现停止功能 */
              }}
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
                        <Typography variant="caption" color="text.secondary">
                          总股票数
                        </Typography>
                      </Box>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="h4" sx={{ fontWeight: 600, color: 'success.main' }}>
                          {currentTask.results.successful_predictions}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          成功预测
                        </Typography>
                      </Box>
                      <Box sx={{ textAlign: 'center' }}>
                        <Typography variant="h4" sx={{ fontWeight: 600, color: 'secondary.main' }}>
                          {((currentTask.results.average_confidence || 0) * 100).toFixed(1)}%
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          平均置信度
                        </Typography>
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
            <Typography variant="h6" component="span">
              确认删除
            </Typography>
          </Box>
        </DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            确定要删除这个任务吗？此操作不可撤销。
          </Typography>
          {currentTask?.status === 'running' && (
            <Box
              sx={{
                mt: 2,
                p: 2,
                bgcolor: 'warning.light',
                border: 1,
                borderColor: 'warning.main',
                borderRadius: 1,
              }}
            >
              <Typography variant="body2" sx={{ color: 'warning.dark', mb: 1 }}>
                ⚠️ 该任务当前正在运行中
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <input
                  type="checkbox"
                  checked={deleteForce}
                  onChange={e => setDeleteForce(e.target.checked)}
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
          <Button
            variant="outlined"
            onClick={() => {
              setDeleteForce(false);
              onDeleteClose();
            }}
          >
            取消
          </Button>
          <Button
            variant="contained"
            color="error"
            onClick={() => {
              handleDelete();
              onDeleteClose();
            }}
          >
            {deleteForce ? '强制删除' : '删除'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
