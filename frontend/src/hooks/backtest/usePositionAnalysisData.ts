/**
 * 持仓分析数据处理 Hook
 */

import { useMemo } from 'react';
import {
  PositionData,
  EnhancedPositionAnalysis,
  normalizePositionData,
  sortPositions,
  calculateStatistics,
  SortConfig,
} from '@/utils/backtest/positionDataUtils';
import {
  generatePieChartData,
  generateBarChartData,
  generateTreemapData,
  generateWeightChartData,
  generateCapitalChartData,
  PortfolioSnapshot,
  PositionWeights,
} from '@/utils/backtest/chartDataUtils';

export const usePositionAnalysisData = (
  positionAnalysis: PositionData[] | EnhancedPositionAnalysis | null,
  sortConfig: SortConfig,
  portfolioSnapshots: PortfolioSnapshot[]
) => {
  // 数据格式转换
  const normalizedData = useMemo(() => {
    return normalizePositionData(positionAnalysis);
  }, [positionAnalysis]);

  // 获取股票表现数据
  const stockPerformance = useMemo(() => {
    return normalizedData?.stock_performance || [];
  }, [normalizedData]);

  // 排序后的持仓数据
  const sortedPositions = useMemo(() => {
    return sortPositions(stockPerformance, sortConfig);
  }, [stockPerformance, sortConfig]);

  // 饼图数据
  const pieChartData = useMemo(() => {
    return generatePieChartData(stockPerformance);
  }, [stockPerformance]);

  // 柱状图数据
  const barChartData = useMemo(() => {
    return generateBarChartData(stockPerformance);
  }, [stockPerformance]);

  // 树状图数据
  const treemapData = useMemo(() => {
    return generateTreemapData(stockPerformance);
  }, [stockPerformance]);

  // 持仓权重数据
  const weightChartData = useMemo(() => {
    return generateWeightChartData((normalizedData?.position_weights as PositionWeights) ?? null);
  }, [normalizedData]);

  // 资金分配图表数据
  const capitalChartData = useMemo(() => {
    return generateCapitalChartData(portfolioSnapshots);
  }, [portfolioSnapshots]);

  // 统计信息
  const statistics = useMemo(() => {
    return calculateStatistics(stockPerformance);
  }, [stockPerformance]);

  return {
    normalizedData,
    stockPerformance,
    sortedPositions,
    pieChartData,
    barChartData,
    treemapData,
    weightChartData,
    capitalChartData,
    statistics,
  };
};
