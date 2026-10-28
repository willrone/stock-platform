import { useParams, useRouter } from 'next/navigation';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { SaveStrategyConfigDialog } from '../../../components/backtest/SaveStrategyConfigDialog';
import {
  BacktestDataAdapter,
  type BacktestDetailedPerformanceViewModel,
  type BacktestDetailedRiskViewModel,
} from '../../../services/backtestDataAdapter';
import { BacktestService } from '../../../services/backtestService';
import { StrategyConfigService } from '../../../services/strategyConfigService';
import { TaskService, type PredictionResult } from '../../../services/taskService';
import { wsService } from '../../../services/websocket';
import type { Task } from '../../../stores/useTaskStore';
import type {
  BacktestDetailedResult,
  BacktestOverviewData,
  BacktestSummaryData,
} from '../../../types/backtest';
import { getStrategyConfig } from './taskDetailUtils';
import type { TaskDetailPageModel, TaskDetailLoadError } from './types';

void SaveStrategyConfigDialog;

/**
 * 【P1-04】把加载异常按 HTTP 状态分类。
 *
 * apiRequest 抛出的错误对象上带 `status`（见 services/api.ts 的 ApiError），
 * 这里做兜底解析，避免任何失败都被误报成"任务不存在或已被删除"。
 */
function classifyTaskDetailError(error: unknown): TaskDetailLoadError {
  const raw = error as {
    status?: number;
    response?: { status?: number };
    message?: string;
  } | null;
  const status = raw?.status ?? raw?.response?.status;
  const message =
    (typeof raw?.message === 'string' && raw.message.trim()) ||
    '加载任务详情失败';

  if (status === 404) {
    return { kind: 'not_found', message: '任务不存在或已被删除', status };
  }
  if (status === 401 || status === 403) {
    return {
      kind: 'unauthorized',
      message: '登录状态已失效，请重新登录后再查看任务详情',
      status,
    };
  }
  if (status === 429) {
    return {
      kind: 'rate_limited',
      message: '请求过于频繁，请稍候重试',
      status,
    };
  }
  return { kind: 'network', message, status };
}

export function useTaskDetailPage(): TaskDetailPageModel {
  const router = useRouter();
  const params = useParams();
  const taskId = params.id as string;

  /**
   * 【state P0-8】当前任务详情由本页本地持有，不放全局 store。
   * 避免从列表页带过来的陈旧任务覆盖本页刚拉到的详情。
   */
  const [currentTask, setCurrentTask] = useState<Task | null>(null);

  /** 局部更新当前任务（WebSocket 进度用） */
  const updateTask = useCallback((taskId: string, patch: Partial<Task>) => {
    setCurrentTask(prev => (prev && prev.task_id === taskId ? { ...prev, ...patch } : prev));
  }, []);

  const [loading, setLoading] = useState(true);
  /** 【P1-04】区分 404 / 401 / 429 / 网络错误，不再一律当"任务不存在" */
  const [loadError, setLoadError] = useState<TaskDetailLoadError | null>(null);
  /** 【P3-03】导出状态与失败原因 */
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedStock, setSelectedStock] = useState('');
  const [backtestDetailedData, setBacktestDetailedData] = useState<BacktestDetailedResult | null>(
    null
  );
  const [adaptedRiskData, setAdaptedRiskData] = useState<BacktestDetailedRiskViewModel | null>(
    null
  );
  const [adaptedPerformanceData, setAdaptedPerformanceData] =
    useState<BacktestDetailedPerformanceViewModel | null>(null);
  const [loadingBacktestData, setLoadingBacktestData] = useState(false);
  const [selectedBacktestTab, setSelectedBacktestTab] = useState('overview');
  const [selectedPredictionTab, setSelectedPredictionTab] = useState('chart');
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [isSaveConfigOpen, setIsSaveConfigOpen] = useState(false);
  const [configSnapshot, setConfigSnapshot] = useState<Record<string, any>>({});
  const [configSnapshotLoading, setConfigSnapshotLoading] = useState(false);
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

      setAdaptedRiskData(BacktestDataAdapter.adaptDetailedRiskViewModel(detailedResult));
      setAdaptedPerformanceData(
        BacktestDataAdapter.adaptDetailedPerformanceViewModel(detailedResult)
      );
    } catch (_error) {
      setBacktestDetailedData(null);
      setAdaptedRiskData(null);
      setAdaptedPerformanceData(null);
    } finally {
      setLoadingBacktestData(false);
    }
  };

  const loadTaskDetail = async (): Promise<void> => {
    setLoadError(null);
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

      // 同步加载配置快照
      loadConfigSnapshot();
    } catch (error: unknown) {
      // 【P1-04】按 HTTP 状态分类，不能一律报"任务不存在"
      setCurrentTask(null);
      setLoadError(classifyTaskDetailError(error));
    } finally {
      setLoading(false);
    }
  };

  const loadConfigSnapshot = async (): Promise<void> => {
    setConfigSnapshotLoading(true);
    try {
      const snapshot = await TaskService.getTaskConfig(taskId);
      setConfigSnapshot(snapshot || {});
    } catch (_error) {
      // 静默处理，旧任务可能没有保存快照
      setConfigSnapshot({});
    } finally {
      setConfigSnapshotLoading(false);
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

    const handleTaskCompleted = async (data: { task_id: string; results: unknown }) => {
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
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : '';
      if (
        message.includes('正在运行中') ||
        message.includes('运行中') ||
        currentTask?.status === 'running'
      ) {
        setDeleteForce(true);
        setIsDeleteOpen(true);
      }
    }
  };

  const handleExport = async (): Promise<void> => {
    // 【P3-03】导出失败必须提示，不能静默 noop
    setExporting(true);
    setExportError(null);
    try {
      const result = await TaskService.exportTaskResults(taskId, 'excel');

      if (result.blob) {
        const url = window.URL.createObjectURL(result.blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = result.fileName;
        document.body.appendChild(link);
        link.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(link);
        return;
      }

      if (result.downloadUrl) {
        const link = document.createElement('a');
        link.href = result.downloadUrl;
        link.download = result.fileName;
        link.rel = 'noopener';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        return;
      }

      setExportError('导出接口未返回可下载内容');
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : '导出失败';
      setExportError(message);
    } finally {
      setExporting(false);
    }
  };

  const strategyConfigInfo = useMemo(() => getStrategyConfig(currentTask), [currentTask]);
  const backtestSummaryData = useMemo<BacktestSummaryData | null>(
    () => BacktestDataAdapter.resolveTaskBacktestSummary(currentTask),
    [currentTask]
  );
  const backtestOverviewData = useMemo<BacktestOverviewData | null>(
    () => BacktestDataAdapter.adaptOverviewData(backtestSummaryData),
    [backtestSummaryData]
  );

  const handleSaveConfig = async (configName: string, description: string): Promise<void> => {
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
    loadError,
    exporting,
    exportError,
    predictions,
    refreshing,
    selectedStock,
    setSelectedStock,
    backtestDetailedData,
    backtestSummaryData,
    backtestOverviewData,
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
    configSnapshot,
    configSnapshotLoading,
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
