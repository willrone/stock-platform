/**
 * 任务操作 Hook（刷新、重试、删除、导出、重建等）
 */

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { Task } from '@/stores/useTaskStore';
import { TaskService } from '@/services/taskService';

export function useTaskActions(
  taskId: string,
  currentTask: Task | null,
  onRefresh: () => Promise<void>
) {
  const router = useRouter();
  const [refreshing, setRefreshing] = useState(false);
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [deleteForce, setDeleteForce] = useState(false);

  // 刷新任务
  const handleRefresh = async () => {
    setRefreshing(true);
    await onRefresh();
    setRefreshing(false);
  };

  // 重新运行任务
  const handleRetry = async () => {
    try {
      await TaskService.retryTask(taskId);
      console.log('任务已重新启动');
      await onRefresh();
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
        setIsDeleteOpen(true);
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
    if (!currentTask) {
      return;
    }

    // 构建配置参数
    const params = new URLSearchParams();
    params.set('rebuild', 'true');
    params.set('task_type', currentTask.task_type);
    params.set('task_name', `${currentTask.task_name} (重建)`);

    // 股票代码
    if (currentTask.stock_codes && currentTask.stock_codes.length > 0) {
      params.set('stock_codes', currentTask.stock_codes.join(','));
    }

    if (currentTask.task_type === 'backtest') {
      // 回测任务配置
      const cfg = currentTask.config;
      const bc = cfg?.backtest_config || cfg;

      if (bc) {
        if (bc.strategy_name) {
          params.set('strategy_name', bc.strategy_name);
        }
        if (bc.start_date) {
          params.set('start_date', bc.start_date);
        }
        if (bc.end_date) {
          params.set('end_date', bc.end_date);
        }
        if (bc.initial_cash !== undefined) {
          params.set('initial_cash', bc.initial_cash.toString());
        }
        if (bc.commission_rate !== undefined) {
          params.set('commission_rate', bc.commission_rate.toString());
        }
        if (bc.slippage_rate !== undefined) {
          params.set('slippage_rate', bc.slippage_rate.toString());
        }
        if (bc.enable_performance_profiling !== undefined) {
          params.set('enable_performance_profiling', bc.enable_performance_profiling.toString());
        }

        // 策略配置
        if (bc.strategy_config) {
          params.set('strategy_config', JSON.stringify(bc.strategy_config));
        }
      }
    } else if (currentTask.task_type === 'prediction') {
      // 预测任务配置
      if (currentTask.model_id) {
        params.set('model_id', currentTask.model_id);
      }

      const predConfig = currentTask.config?.prediction_config;
      if (predConfig) {
        if (predConfig.horizon) {
          params.set('horizon', predConfig.horizon.toString());
        }
        if (predConfig.confidence_level !== undefined) {
          params.set('confidence_level', (predConfig.confidence_level * 100).toString());
        }
        if (predConfig.risk_assessment !== undefined) {
          params.set('risk_assessment', predConfig.risk_assessment.toString());
        }
      }
    }

    // 跳转到创建页面
    router.push(`/tasks/create?${params.toString()}`);
  };

  return {
    refreshing,
    isDeleteOpen,
    deleteForce,
    setIsDeleteOpen,
    setDeleteForce,
    handleRefresh,
    handleRetry,
    handleDelete,
    handleExport,
    handleRebuild,
  };
}
