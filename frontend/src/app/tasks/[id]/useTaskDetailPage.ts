import { useParams, useRouter } from 'next/navigation';
import { useEffect, useMemo, useRef, useState } from 'react';

import { SaveStrategyConfigDialog } from '../../../components/backtest/SaveStrategyConfigDialog';
import { BacktestDataAdapter } from '../../../services/backtestDataAdapter';
import { BacktestService } from '../../../services/backtestService';
import { StrategyConfigService } from '../../../services/strategyConfigService';
import { TaskService, type PredictionResult } from '../../../services/taskService';
import { wsService } from '../../../services/websocket';
import { useTaskStore, type Task } from '../../../stores/useTaskStore';
import { getStrategyConfig } from './taskDetailUtils';
import type { TaskDetailPageModel } from './types';

void SaveStrategyConfigDialog;

export function useTaskDetailPage(): TaskDetailPageModel {
  const router = useRouter();
  const params = useParams();
  const taskId = params.id as string;

  const { currentTask, setCurrentTask, updateTask } = useTaskStore();
  const [loading, setLoading] = useState(true);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedStock, setSelectedStock] = useState('');
  const [backtestDetailedData, setBacktestDetailedData] = useState<any>(null);
  const [adaptedRiskData, setAdaptedRiskData] = useState<any>(null);
  const [adaptedPerformanceData, setAdaptedPerformanceData] = useState<any>(null);
  const [loadingBacktestData, setLoadingBacktestData] = useState(false);
  const [selectedBacktestTab, setSelectedBacktestTab] = useState('overview');
  const [selectedPredictionTab, setSelectedPredictionTab] = useState('chart');
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [isSaveConfigOpen, setIsSaveConfigOpen] = useState(false);
  const [deleteForce, setDeleteForce] = useState(false);
  const [savingConfig, setSavingConfig] = useState(false);
  const [selectedStocksPage, setSelectedStocksPage] = useState(1);
  const hasTriggeredLoadRef = useRef(false);

  const loadBacktestDetailedData = async (force: boolean = false): Promise<void> => {
    if (!force && backtestDetailedData !== null && !loadingBacktestData) {
      return;
    }

    const task = currentTask;
    if (!task || task.task_type !== 'backtest' || task.status !== 'completed') {
      return;
    }

    setLoadingBacktestData(true);
    try {
      const detailedResult = await BacktestService.getDetailedResult(taskId);
      setBacktestDetailedData(detailedResult);

      const riskMetrics = BacktestDataAdapter.adaptRiskMetrics(detailedResult);
      const returnDistribution = BacktestDataAdapter.generateReturnDistribution(detailedResult);
      const rollingMetrics = BacktestDataAdapter.generateRollingMetrics(detailedResult);
      setAdaptedRiskData({ riskMetrics, returnDistribution, rollingMetrics });

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
    } catch (_error) {
      setBacktestDetailedData(null);
      setAdaptedRiskData(null);
      setAdaptedPerformanceData(null);
    } finally {
      setLoadingBacktestData(false);
    }
  };

  const loadTaskDetail = async (): Promise<void> => {
    try {
      const task = await TaskService.getTaskDetail(taskId);
      setCurrentTask(task);

      if (task.status === 'completed' && task.results) {
        if (task.task_type === 'prediction') {
          const results = await TaskService.getTaskResults(taskId);
          setPredictions(results);
          if (results.length > 0) {
            setSelectedStock(results[0].stock_code);
          }
        } else if (task.task_type === 'backtest') {
          if (!selectedStock && task.stock_codes && task.stock_codes.length > 0) {
            setSelectedStock(task.stock_codes[0]);
          }
          await loadBacktestDetailedData(true);
        }
      }
    } catch (_error) {
      // noop
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (taskId) {
      void loadTaskDetail();
      wsService.subscribeToTask(taskId);
    }

    return () => {
      if (taskId) {
        wsService.unsubscribeFromTask(taskId);
      }
    };
  }, [taskId]);

  useEffect(() => {
    if (
      currentTask &&
      currentTask.task_type === 'backtest' &&
      currentTask.status === 'completed' &&
      !hasTriggeredLoadRef.current
    ) {
      hasTriggeredLoadRef.current = true;
      void loadBacktestDetailedData();
    }

    if (currentTask?.status !== 'completed') {
      hasTriggeredLoadRef.current = false;
    }
  }, [currentTask?.status, currentTask?.task_type]);

  useEffect(() => {
    const handleTaskProgress = (data: { task_id: string; progress: number; status: string }) => {
      if (data.task_id !== taskId) {
        return;
      }

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
    };

    const handleTaskCompleted = async (data: { task_id: string; results: any }) => {
      if (data.task_id !== taskId) {
        return;
      }

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

        if (task.task_type === 'prediction') {
          const results = await TaskService.getTaskResults(taskId);
          setPredictions(results);
          if (results.length > 0) {
            setSelectedStock(results[0].stock_code);
          }
        } else if (task.task_type === 'backtest') {
          await loadBacktestDetailedData(true);
        }
      } catch (_error) {
        // noop
      }
    };

    const handleTaskFailed = (data: { task_id: string; error: string }) => {
      if (data.task_id !== taskId || !currentTask) {
        return;
      }

      const updatedTask = {
        ...currentTask,
        status: 'failed' as const,
        error_message: data.error,
      };

      setCurrentTask(updatedTask);
      updateTask(data.task_id, updatedTask);
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

  const handleRefresh = async (): Promise<void> => {
    setRefreshing(true);
    await loadTaskDetail();
    setRefreshing(false);
  };

  const handleRetry = async (): Promise<void> => {
    try {
      await TaskService.retryTask(taskId);
      await loadTaskDetail();
    } catch (_error) {
      // noop
    }
  };

  const handleDelete = async (): Promise<void> => {
    try {
      await TaskService.deleteTask(taskId, deleteForce);
      router.push('/tasks');
    } catch (error: any) {
      if (
        error.message?.includes('正在运行中') ||
        error.message?.includes('运行中') ||
        currentTask?.status === 'running'
      ) {
        setDeleteForce(true);
        setIsDeleteOpen(true);
      }
    }
  };

  const handleExport = async (): Promise<void> => {
    try {
      const blob = await TaskService.exportTaskResults(taskId, 'csv');
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `task_${taskId}_results.csv`;
      document.body.appendChild(link);
      link.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(link);
    } catch (_error) {
      // noop
    }
  };

  const strategyConfigInfo = useMemo(() => getStrategyConfig(currentTask), [currentTask]);

  const handleSaveConfig = async (
    configName: string,
    description: string
  ): Promise<void> => {
    if (!strategyConfigInfo) {
      throw new Error('无法获取策略配置信息');
    }

    setSavingConfig(true);
    try {
      await StrategyConfigService.createConfig({
        config_name: configName,
        strategy_name: strategyConfigInfo.strategyName,
        parameters: strategyConfigInfo.parameters,
        description,
      });
    } finally {
      setSavingConfig(false);
    }
  };

  const handleBack = (): void => {
    router.push('/tasks');
  };

  const handleRebuild = (): void => {
    router.push(`/tasks/create?rebuild=${taskId}`);
  };

  return {
    taskId,
    currentTask,
    loading,
    predictions,
    refreshing,
    selectedStock,
    setSelectedStock,
    backtestDetailedData,
    adaptedRiskData,
    adaptedPerformanceData,
    loadingBacktestData,
    selectedBacktestTab,
    setSelectedBacktestTab,
    selectedPredictionTab,
    setSelectedPredictionTab,
    isDeleteOpen,
    isSaveConfigOpen,
    deleteForce,
    setDeleteForce,
    savingConfig,
    selectedStocksPage,
    setSelectedStocksPage,
    strategyConfigInfo,
    loadBacktestDetailedData,
    loadTaskDetail,
    handleRefresh,
    handleRetry,
    handleDelete,
    handleExport,
    handleSaveConfig,
    handleBack,
    handleRebuild,
    openDeleteDialog: () => setIsDeleteOpen(true),
    closeDeleteDialog: () => setIsDeleteOpen(false),
    openSaveConfigDialog: () => setIsSaveConfigOpen(true),
    closeSaveConfigDialog: () => setIsSaveConfigOpen(false),
  };
}
