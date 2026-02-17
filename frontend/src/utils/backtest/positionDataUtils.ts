/**
 * 持仓数据处理工具函数
 */

export interface PositionData {
  stock_code: string;
  stock_name: string;
  total_return: number;
  trade_count: number;
  win_rate: number;
  avg_holding_period: number;
  winning_trades: number;
  losing_trades: number;
  avg_return_per_trade?: number;
  return_ratio?: number;
  trade_frequency?: number;
  avg_win?: number;
  avg_loss?: number;
  largest_win?: number;
  largest_loss?: number;
  profit_factor?: number;
  max_holding_period?: number;
  min_holding_period?: number;
  avg_buy_price?: number;
  avg_sell_price?: number;
  price_improvement?: number;
  total_volume?: number;
  total_commission?: number;
  commission_ratio?: number;
}

export interface EnhancedPositionAnalysis {
  stock_performance: PositionData[];
  position_weights?: any;
  trading_patterns?: any;
  holding_periods?: any;
  concentration_risk?: any;
}

export interface SortConfig {
  key: keyof PositionData;
  direction: 'asc' | 'desc';
}

/**
 * 数据格式转换：兼容新旧两种格式
 */
export const normalizePositionData = (
  positionAnalysis: PositionData[] | EnhancedPositionAnalysis | null
): EnhancedPositionAnalysis | null => {
  if (!positionAnalysis) {
    return null;
  }

  // 如果是数组格式（旧格式），直接使用
  if (Array.isArray(positionAnalysis)) {
    return {
      stock_performance: positionAnalysis,
      position_weights: undefined,
      trading_patterns: undefined,
      holding_periods: undefined,
      concentration_risk: undefined,
    };
  }

  // 如果是对象格式（新格式），检查是否有 stock_performance
  if (typeof positionAnalysis === 'object' && positionAnalysis !== null) {

    // 确保 stock_performance 存在且是数组
    if (positionAnalysis.stock_performance && Array.isArray(positionAnalysis.stock_performance)) {
      return positionAnalysis as EnhancedPositionAnalysis;
    } else {
      console.warn('[PositionAnalysis] stock_performance 不存在或不是数组');
      return null;
    }
  }

  console.warn('[PositionAnalysis] 未知的数据格式');
  return null;
};

/**
 * 排序持仓数据
 */
export const sortPositions = (
  positions: PositionData[],
  sortConfig: SortConfig
): PositionData[] => {
  if (!positions || positions.length === 0) {
    return [];
  }

  return [...positions].sort((a, b) => {
    const aValue = a[sortConfig.key];
    const bValue = b[sortConfig.key];

    if (typeof aValue === 'number' && typeof bValue === 'number') {
      return sortConfig.direction === 'asc' ? aValue - bValue : bValue - aValue;
    }

    if (typeof aValue === 'string' && typeof bValue === 'string') {
      return sortConfig.direction === 'asc'
        ? aValue.localeCompare(bValue)
        : bValue.localeCompare(aValue);
    }

    return 0;
  });
};

/**
 * 计算统计信息
 */
export const calculateStatistics = (stockPerformance: PositionData[]) => {
  if (!stockPerformance || stockPerformance.length === 0) {
    return {
      totalStocks: 0,
      profitableStocks: 0,
      totalReturn: 0,
      avgWinRate: 0,
      avgHoldingPeriod: 0,
      bestPerformer: null,
      worstPerformer: null,
    };
  }

  const totalStocks = stockPerformance.length;
  const profitableStocks = stockPerformance.filter(pos => pos.total_return > 0).length;
  const totalReturn = stockPerformance.reduce((sum, pos) => sum + pos.total_return, 0);
  const avgWinRate = stockPerformance.reduce((sum, pos) => sum + pos.win_rate, 0) / totalStocks;
  const avgHoldingPeriod =
    stockPerformance.reduce((sum, pos) => sum + pos.avg_holding_period, 0) / totalStocks;

  const bestPerformer = stockPerformance.reduce((best, pos) =>
    pos.total_return > best.total_return ? pos : best
  );
  const worstPerformer = stockPerformance.reduce((worst, pos) =>
    pos.total_return < worst.total_return ? pos : worst
  );

  return {
    totalStocks,
    profitableStocks,
    totalReturn,
    avgWinRate,
    avgHoldingPeriod,
    bestPerformer,
    worstPerformer,
  };
};
