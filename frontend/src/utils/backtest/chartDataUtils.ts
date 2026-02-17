/**
 * 图表数据处理工具函数
 */

import { PositionData } from './positionDataUtils';

export interface PositionWeights {
  current_weights?: Record<string, number>;
  [key: string]: unknown;
}

export interface PortfolioSnapshot {
  snapshot_date: string;
  portfolio_value: number;
  cash: number;
  [key: string]: unknown;
}

/**
 * 生成饼图数据（基于总收益）
 */
export const generatePieChartData = (stockPerformance: PositionData[]) => {
  if (!stockPerformance || stockPerformance.length === 0) {
    return [];
  }

  return stockPerformance
    .filter(pos => Math.abs(pos.total_return) > 0)
    .map(pos => ({
      name: pos.stock_code,
      value: Math.abs(pos.total_return),
      originalValue: pos.total_return,
    }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 10); // 只显示前10个
};

/**
 * 生成柱状图数据
 */
export const generateBarChartData = (stockPerformance: PositionData[]) => {
  if (!stockPerformance || stockPerformance.length === 0) {
    return [];
  }

  return stockPerformance
    .map(pos => ({
      stock_code: pos.stock_code,
      total_return: pos.total_return,
      win_rate: pos.win_rate * 100,
      trade_count: pos.trade_count,
      avg_holding_period: pos.avg_holding_period,
    }))
    .sort((a, b) => b.total_return - a.total_return)
    .slice(0, 15); // 显示前15个
};

/**
 * 生成树状图数据（用于权重可视化）
 */
export const generateTreemapData = (stockPerformance: PositionData[]) => {
  if (!stockPerformance || stockPerformance.length === 0) {
    return [];
  }

  return stockPerformance
    .filter(pos => Math.abs(pos.total_return) > 0)
    .map(pos => ({
      name: pos.stock_code,
      value: Math.abs(pos.total_return),
      originalValue: pos.total_return,
      itemStyle: {
        color: pos.total_return >= 0 ? '#10b981' : '#ef4444',
      },
    }))
    .sort((a, b) => b.value - a.value);
};

/**
 * 生成持仓权重数据（基于真实权重）
 */
export const generateWeightChartData = (positionWeights: PositionWeights | null) => {
  const weights = positionWeights?.current_weights;
  if (!weights || Object.keys(weights).length === 0) {
    return null;
  }

  return Object.entries(weights)
    .map(([stock_code, weight]) => ({
      name: stock_code,
      value: (weight as number) * 100, // 转换为百分比
    }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 15);
};

/**
 * 生成资金分配图表数据
 */
export const generateCapitalChartData = (portfolioSnapshots: PortfolioSnapshot[]) => {
  if (!portfolioSnapshots || portfolioSnapshots.length === 0) {
    return null;
  }

  const dates: string[] = [];
  const totalCapital: number[] = [];
  const positionCapital: number[] = [];
  const freeCapital: number[] = [];

  portfolioSnapshots.forEach(snapshot => {
    dates.push(snapshot.snapshot_date);
    totalCapital.push(snapshot.portfolio_value);
    freeCapital.push(snapshot.cash);
    positionCapital.push(snapshot.portfolio_value - snapshot.cash);
  });

  return {
    dates,
    totalCapital,
    positionCapital,
    freeCapital,
  };
};
