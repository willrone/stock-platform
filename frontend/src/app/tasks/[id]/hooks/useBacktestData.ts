/**
 * 回测详细数据加载 Hook
 */

import { useState, useEffect, useRef } from 'react';
import { Task } from '@/stores/useTaskStore';
import { BacktestService } from '@/services/backtestService';
import { BacktestDataAdapter } from '@/services/backtestDataAdapter';

export function useBacktestData(taskId: string, currentTask: Task | null) {
  const [backtestDetailedData, setBacktestDetailedData] = useState<any>(null);
  const [adaptedRiskData, setAdaptedRiskData] = useState<any>(null);
  const [adaptedPerformanceData, setAdaptedPerformanceData] = useState<any>(null);
  const [loadingBacktestData, setLoadingBacktestData] = useState(false);
  const hasTriggeredLoadRef = useRef(false);

  // 加载回测详细数据
  const loadBacktestDetailedData = async (force: boolean = false) => {
    // 如果数据已加载且不强制刷新，则跳过
    if (!force && backtestDetailedData !== null && !loadingBacktestData) {
      console.log('[useBacktestData] 回测详细数据已加载，跳过重复加载');
      return;
    }

    // 检查任务状态
    const task = currentTask;
    if (!task || task.task_type !== 'backtest' || task.status !== 'completed') {
      console.log('[useBacktestData] 任务状态不满足加载条件:', {
        hasTask: !!task,
        taskType: task?.task_type,
        status: task?.status,
      });
      return;
    }

    setLoadingBacktestData(true);
    try {
      console.log('[useBacktestData] 开始加载回测详细数据...');
      const detailedResult = await BacktestService.getDetailedResult(taskId);
      console.log('[useBacktestData] 后端返回的详细数据:', detailedResult);

      setBacktestDetailedData(detailedResult);

      // 使用数据适配器转换数据格式
      console.log('[useBacktestData] 开始适配风险分析数据...');
      const riskMetrics = BacktestDataAdapter.adaptRiskMetrics(detailedResult);
      const returnDistribution = BacktestDataAdapter.generateReturnDistribution(detailedResult);
      const rollingMetrics = BacktestDataAdapter.generateRollingMetrics(detailedResult);

      setAdaptedRiskData({
        riskMetrics,
        returnDistribution,
        rollingMetrics,
      });

      console.log('[useBacktestData] 开始适配绩效分解数据...');
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
    } catch (error) {
      console.error('[useBacktestData] 加载回测详细数据失败:', error);
      setBacktestDetailedData(null);
      setAdaptedRiskData(null);
      setAdaptedPerformanceData(null);
    } finally {
      setLoadingBacktestData(false);
    }
  };

  // 当任务状态变为 completed 时自动加载
  useEffect(() => {
    if (
      currentTask &&
      currentTask.task_type === 'backtest' &&
      currentTask.status === 'completed' &&
      !hasTriggeredLoadRef.current
    ) {
      hasTriggeredLoadRef.current = true;
      console.log('[useBacktestData] 检测到回测任务已完成，触发数据加载');
      loadBacktestDetailedData();
    }
    // 如果任务状态变化，重置 ref
    if (currentTask?.status !== 'completed') {
      hasTriggeredLoadRef.current = false;
    }
  }, [currentTask?.status, currentTask?.task_type]);

  return {
    backtestDetailedData,
    adaptedRiskData,
    adaptedPerformanceData,
    loadingBacktestData,
    loadBacktestDetailedData,
  };
}
