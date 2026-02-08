/**
 * 任务详情数据加载和管理 Hook
 */

import { useState, useEffect } from 'react';
import { useTaskStore, Task } from '@/stores/useTaskStore';
import { TaskService, PredictionResult } from '@/services/taskService';
import { wsService } from '@/services/websocket';

export function useTaskDetail(taskId: string) {
  const { currentTask, setCurrentTask } = useTaskStore();
  const [loading, setLoading] = useState(true);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [selectedStock, setSelectedStock] = useState<string>('');

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
          // 如果没有股票代码，使用任务配置中的股票代码
          if (!selectedStock && task.stock_codes && task.stock_codes.length > 0) {
            setSelectedStock(task.stock_codes[0]);
          }
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

  return {
    currentTask,
    loading,
    predictions,
    selectedStock,
    setSelectedStock,
    loadTaskDetail,
  };
}
