/**
 * 回测详细结果服务
 * 处理与回测详细数据相关的API调用
 */

import { apiRequest } from './api';

// 回测详细结果接口
export interface BacktestDetailedResult {
  task_id: string;
  backtest_id: string;
  extended_risk_metrics: {
    sortino_ratio: number;
    calmar_ratio: number;
    max_drawdown_duration: number;
    var_95: number;
    downside_deviation: number;
  };
  monthly_returns: Array<{
    year: number;
    month: number;
    date: string;
    monthly_return: number;
    cumulative_return: number;
  }>;
  position_analysis: {
    // 股票表现数据（兼容原有格式）
    stock_performance: Array<{
      stock_code: string;
      stock_name: string;
      total_return: number;
      trade_count: number;
      win_rate: number;
      avg_holding_period: number;
      winning_trades: number;
      losing_trades: number;
      // 扩展字段
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
    }>;
    // 持仓权重分析
    position_weights?: {
      weight_statistics?: Array<{
        stock_code: string;
        avg_weight: number;
        max_weight: number;
        min_weight: number;
        weight_volatility: number;
        observations: number;
      }>;
      weight_changes?: Array<{
        date: string;
        stock_code: string;
        prev_weight: number;
        curr_weight: number;
        weight_change: number;
        change_type: string;
      }>;
      concentration_metrics?: {
        time_series?: Array<{
          date: string;
          hhi: number;
          effective_stocks: number;
          top_1_concentration: number;
          top_3_concentration: number;
          top_5_concentration: number;
          total_positions: number;
        }>;
        averages?: {
          avg_hhi: number;
          avg_effective_stocks: number;
          avg_top_1_concentration: number;
          avg_top_3_concentration: number;
          avg_top_5_concentration: number;
          avg_total_positions: number;
        };
      };
      current_weights?: Record<string, number>;
    };
    // 交易模式分析
    trading_patterns?: {
      time_patterns?: {
        monthly_distribution?: Array<{
          month: number;
          count: number;
          percentage: number;
        }>;
        weekday_distribution?: Array<{
          weekday: number;
          weekday_name: string;
          count: number;
          percentage: number;
        }>;
      };
      size_patterns?: {
        avg_trade_size: number;
        median_trade_size: number;
        max_trade_size: number;
        min_trade_size: number;
        trade_size_std: number;
        total_volume: number;
      };
      frequency_patterns?: {
        avg_interval_days: number;
        median_interval_days: number;
        min_interval_days: number;
        max_interval_days: number;
        avg_monthly_trades: number;
        max_monthly_trades: number;
        total_trading_days: number;
      };
      success_patterns?: {
        total_closed_trades: number;
        winning_trades: number;
        losing_trades: number;
        breakeven_trades: number;
        win_rate: number;
        loss_rate: number;
        avg_win_amount: number;
        avg_loss_amount: number;
      };
    };
    // 持仓时间分析
    holding_periods?: {
      avg_holding_period: number;
      median_holding_period: number;
      max_holding_period: number;
      min_holding_period: number;
      holding_period_std: number;
      total_positions_closed: number;
      short_term_positions: number;
      medium_term_positions: number;
      long_term_positions: number;
    };
    // 风险集中度分析
    concentration_risk?: {
      trade_concentration?: {
        hhi: number;
        effective_stocks: number;
        top_1_weight: number;
        top_3_weight: number;
        top_5_weight: number;
        total_stocks: number;
      };
      position_concentration?: {
        hhi: number;
        effective_positions: number;
        top_1_weight: number;
        top_3_weight: number;
        top_5_weight: number;
        total_positions: number;
      };
    };
  } | Array<{
    // 兼容旧格式
    stock_code: string;
    stock_name: string;
    total_return: number;
    trade_count: number;
    win_rate: number;
    avg_holding_period: number;
    winning_trades: number;
    losing_trades: number;
  }>;
  drawdown_analysis: {
    max_drawdown: number;
    max_drawdown_date: string;
    max_drawdown_start: string;
    max_drawdown_end: string;
    max_drawdown_duration: number;
    drawdown_curve: Array<{
      date: string;
      drawdown: number;
    }>;
  };
}

// 组合快照接口
export interface PortfolioSnapshot {
  id: number;
  task_id: string;
  snapshot_date: string;
  portfolio_value: number;
  cash: number;
  positions_count: number;
  total_return: number;
  daily_return: number;
  positions: Record<string, any>;
}

// 交易记录接口
export interface TradeRecord {
  id: number;
  task_id: string;
  trade_id: string;
  stock_code: string;
  action: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  timestamp: string;
  commission: number;
  pnl: number;
  holding_days?: number;
}

// 交易统计接口
export interface TradeStatistics {
  total_trades: number;
  buy_trades: number;
  sell_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  avg_profit: number;
  avg_loss: number;
  profit_factor: number;
  total_commission: number;
  total_pnl: number;
}

// 信号记录接口
export interface SignalRecord {
  id: number;
  task_id: string;
  backtest_id: string;
  signal_id: string;
  stock_code: string;
  stock_name?: string;
  signal_type: 'BUY' | 'SELL';
  timestamp: string;
  price: number;
  strength: number;
  reason?: string;
  metadata?: Record<string, any>;
  executed: boolean;
  created_at: string;
}

// 信号统计接口
export interface SignalStatistics {
  total_signals: number;
  buy_signals: number;
  sell_signals: number;
  executed_signals: number;
  unexecuted_signals: number;
  execution_rate: number;
  avg_strength: number;
}

// 基准数据接口
export interface BenchmarkData {
  id: number;
  task_id: string;
  benchmark_symbol: string;
  benchmark_name: string;
  correlation: number;
  beta: number;
  alpha: number;
  tracking_error: number;
  information_ratio: number;
  excess_return: number;
  benchmark_returns: Array<{
    date: string;
    return: number;
    cumulative_return: number;
  }>;
}

// 图表缓存统计接口
export interface CacheStatistics {
  total_cached_charts: number;
  cache_hit_rate: number;
  expired_charts: number;
  cache_size_mb: number;
  most_cached_chart_types: Array<{
    chart_type: string;
    count: number;
  }>;
}

export class BacktestService {
  /**
   * 获取回测详细结果
   */
  static async getDetailedResult(taskId: string): Promise<BacktestDetailedResult> {
    return apiRequest.get<BacktestDetailedResult>(`/backtest-detailed/${taskId}/detailed-result`);
  }

  /**
   * 获取组合快照数据
   */
  static async getPortfolioSnapshots(
    taskId: string,
    startDate?: string,
    endDate?: string,
    limit: number = 100
  ): Promise<{
    snapshots: PortfolioSnapshot[];
    total_count: number;
  }> {
    const params: any = { limit };
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;

    return apiRequest.get(`/backtest-detailed/${taskId}/portfolio-snapshots`, params);
  }

  /**
   * 获取交易记录
   */
  static async getTradeRecords(
    taskId: string,
    options: {
      stockCode?: string;
      action?: 'BUY' | 'SELL';
      startDate?: string;
      endDate?: string;
      offset?: number;
      limit?: number;
      orderBy?: string;
      orderDesc?: boolean;
    } = {}
  ): Promise<{
    trades: TradeRecord[];
    pagination: {
      offset: number;
      limit: number;
      count: number;
    };
  }> {
    const params: any = {
      offset: options.offset || 0,
      limit: options.limit || 50,
      order_by: options.orderBy || 'timestamp',
      order_desc: options.orderDesc !== false,
    };

    if (options.stockCode) params.stock_code = options.stockCode;
    if (options.action) params.action = options.action;
    if (options.startDate) params.start_date = options.startDate;
    if (options.endDate) params.end_date = options.endDate;

    return apiRequest.get(`/backtest-detailed/${taskId}/trade-records`, params);
  }

  /**
   * 获取交易统计信息
   */
  static async getTradeStatistics(taskId: string): Promise<TradeStatistics> {
    return apiRequest.get<TradeStatistics>(`/backtest-detailed/${taskId}/trade-statistics`);
  }

  /**
   * 获取信号记录
   */
  static async getSignalRecords(
    taskId: string,
    options: {
      stockCode?: string;
      signalType?: 'BUY' | 'SELL';
      startDate?: string;
      endDate?: string;
      executed?: boolean;
      offset?: number;
      limit?: number;
      orderBy?: string;
      orderDesc?: boolean;
    } = {}
  ): Promise<{
    signals: SignalRecord[];
    pagination: {
      offset: number;
      limit: number;
      count: number;
    };
  }> {
    const params: any = {
      offset: options.offset || 0,
      limit: options.limit || 50,
      order_by: options.orderBy || 'timestamp',
      order_desc: options.orderDesc !== false,
    };

    if (options.stockCode) params.stock_code = options.stockCode;
    if (options.signalType) params.signal_type = options.signalType;
    if (options.startDate) params.start_date = options.startDate;
    if (options.endDate) params.end_date = options.endDate;
    if (options.executed !== undefined) params.executed = options.executed;

    return apiRequest.get(`/backtest-detailed/${taskId}/signal-records`, params);
  }

  /**
   * 获取信号统计信息
   */
  static async getSignalStatistics(taskId: string): Promise<SignalStatistics> {
    return apiRequest.get<SignalStatistics>(`/backtest-detailed/${taskId}/signal-statistics`);
  }

  /**
   * 获取基准对比数据
   */
  static async getBenchmarkData(
    taskId: string,
    benchmarkSymbol: string = '000300.SH'
  ): Promise<BenchmarkData> {
    return apiRequest.get<BenchmarkData>(
      `/backtest-detailed/${taskId}/benchmark-data`,
      { benchmark_symbol: benchmarkSymbol }
    );
  }

  /**
   * 缓存图表数据
   */
  static async cacheChartData(
    taskId: string,
    chartType: string,
    chartData: Record<string, any>,
    expiryHours: number = 24
  ): Promise<{ task_id: string; chart_type: string }> {
    return apiRequest.post(`/backtest-detailed/${taskId}/cache-chart`, {
      chart_type: chartType,
      chart_data: chartData,
      expiry_hours: expiryHours,
    });
  }

  /**
   * 获取缓存的图表数据
   */
  static async getCachedChartData(
    taskId: string,
    chartType: string
  ): Promise<Record<string, any> | null> {
    try {
      return await apiRequest.get(`/backtest-detailed/${taskId}/cached-chart/${chartType}`);
    } catch (error: any) {
      // 检查是否是404错误（缓存不存在）
      if (error.response?.status === 404 || error.status === 404 || error.message?.includes('404')) {
        console.log(`[BacktestService] 缓存不存在: taskId=${taskId}, chartType=${chartType}`);
        return null; // 缓存不存在
      }
      throw error;
    }
  }

  /**
   * 使缓存失效
   */
  static async invalidateCache(
    taskId: string,
    chartType?: string
  ): Promise<{ task_id: string; chart_type?: string }> {
    const url = chartType 
      ? `/backtest-detailed/${taskId}/cache?chart_type=${chartType}`
      : `/backtest-detailed/${taskId}/cache`;
    return apiRequest.delete(url);
  }

  /**
   * 获取缓存统计信息
   */
  static async getCacheStatistics(): Promise<CacheStatistics> {
    return apiRequest.get<CacheStatistics>('/backtest-detailed/cache/statistics');
  }

  /**
   * 清理过期缓存
   */
  static async cleanupExpiredCache(): Promise<{ deleted_count: number }> {
    return apiRequest.delete('/backtest-detailed/cache/cleanup');
  }

  /**
   * 删除任务的所有详细数据
   */
  static async deleteTaskData(taskId: string): Promise<{ task_id: string }> {
    return apiRequest.delete(`/backtest-detailed/${taskId}/data`);
  }

  /**
   * 获取图表数据（带缓存）
   */
  static async getChartData(
    taskId: string,
    chartType: string,
    forceRefresh: boolean = false
  ): Promise<Record<string, any>> {
    console.log(`[BacktestService] 开始获取图表数据: taskId=${taskId}, chartType=${chartType}, forceRefresh=${forceRefresh}`);
    
    // 如果不强制刷新，先尝试获取缓存数据
    if (!forceRefresh) {
      console.log(`[BacktestService] 尝试获取缓存数据...`);
      try {
        const cachedData = await this.getCachedChartData(taskId, chartType);
        if (cachedData) {
          console.log(`[BacktestService] 找到缓存数据，直接返回`);
          return cachedData;
        } else {
          console.log(`[BacktestService] 未找到缓存数据，需要生成新数据`);
        }
      } catch (error) {
        console.warn(`[BacktestService] 获取缓存数据失败:`, error);
      }
    }

    // 根据图表类型生成数据
    let chartData: Record<string, any>;
    console.log(`[BacktestService] 开始生成 ${chartType} 类型的图表数据...`);

    try {
      switch (chartType) {
        case 'equity_curve':
          console.log(`[BacktestService] 生成权益曲线数据...`);
          chartData = await this.generateEquityCurveData(taskId);
          break;
        case 'drawdown_curve':
          console.log(`[BacktestService] 生成回撤曲线数据...`);
          chartData = await this.generateDrawdownCurveData(taskId);
          break;
        case 'monthly_heatmap':
          console.log(`[BacktestService] 生成月度热力图数据...`);
          chartData = await this.generateMonthlyHeatmapData(taskId);
          break;
        case 'trade_distribution':
          console.log(`[BacktestService] 生成交易分布数据...`);
          chartData = await this.generateTradeDistributionData(taskId);
          break;
        case 'position_weights':
          console.log(`[BacktestService] 生成持仓权重数据...`);
          chartData = await this.generatePositionWeightsData(taskId);
          break;
        default:
          console.error(`[BacktestService] 不支持的图表类型: ${chartType}`);
          throw new Error(`不支持的图表类型: ${chartType}`);
      }
      
      console.log(`[BacktestService] 图表数据生成成功:`, chartData);
    } catch (error) {
      console.error(`[BacktestService] 生成图表数据失败:`, error);
      throw error;
    }

    // 缓存生成的数据
    try {
      console.log(`[BacktestService] 尝试缓存图表数据...`);
      await this.cacheChartData(taskId, chartType, chartData);
      console.log(`[BacktestService] 图表数据缓存成功`);
    } catch (error) {
      console.warn('[BacktestService] 缓存图表数据失败:', error);
    }

    console.log(`[BacktestService] 图表数据获取完成`);
    return chartData;
  }

  /**
   * 生成权益曲线数据
   */
  private static async generateEquityCurveData(taskId: string): Promise<Record<string, any>> {
    try {
      // 获取所有数据，不限制数量（传入一个很大的数字，或者不传limit让后端返回所有）
      const snapshots = await this.getPortfolioSnapshots(taskId, undefined, undefined, 100000);
      
      if (!snapshots || !snapshots.snapshots || snapshots.snapshots.length === 0) {
        console.warn(`[BacktestService] 组合快照数据为空: taskId=${taskId}`);
        return {
          dates: [],
          portfolioValues: [],
          returns: [],
          dailyReturns: [],
        };
      }
      
      // 按日期排序，确保数据顺序正确
      const sortedSnapshots = [...snapshots.snapshots].sort((a, b) => 
        new Date(a.snapshot_date).getTime() - new Date(b.snapshot_date).getTime()
      );
      
      console.log(`[BacktestService] 生成权益曲线数据: taskId=${taskId}, 数据量=${sortedSnapshots.length}, 日期范围=${sortedSnapshots[0]?.snapshot_date} 至 ${sortedSnapshots[sortedSnapshots.length - 1]?.snapshot_date}`);
      
      return {
        dates: sortedSnapshots.map(s => s.snapshot_date),
        portfolioValues: sortedSnapshots.map(s => s.portfolio_value),
        returns: sortedSnapshots.map(s => s.total_return),
        dailyReturns: sortedSnapshots.map(s => s.daily_return || 0),
      };
    } catch (error: any) {
      console.error(`[BacktestService] 生成权益曲线数据失败:`, error);
      if (error.response?.status === 404 || error.status === 404 || error.message?.includes('404')) {
        console.warn(`[BacktestService] 组合快照数据不存在，返回空数据`);
        return {
          dates: [],
          portfolioValues: [],
          returns: [],
          dailyReturns: [],
        };
      }
      throw error;
    }
  }

  /**
   * 生成回撤曲线数据
   */
  private static async generateDrawdownCurveData(taskId: string): Promise<Record<string, any>> {
    try {
      const detailedResult = await this.getDetailedResult(taskId);
      
      if (!detailedResult || !detailedResult.drawdown_analysis) {
        console.warn(`[BacktestService] 回撤分析数据为空: taskId=${taskId}`);
        return {
          dates: [],
          drawdowns: [],
          maxDrawdown: 0,
          maxDrawdownDate: '',
          maxDrawdownDuration: 0,
        };
      }
      
      return {
        dates: detailedResult.drawdown_analysis.drawdown_curve?.map((d: any) => d.date) || [],
        drawdowns: detailedResult.drawdown_analysis.drawdown_curve?.map((d: any) => d.drawdown) || [],
        maxDrawdown: detailedResult.drawdown_analysis.max_drawdown || 0,
        maxDrawdownDate: detailedResult.drawdown_analysis.max_drawdown_date || '',
        maxDrawdownDuration: detailedResult.drawdown_analysis.max_drawdown_duration || 0,
      };
    } catch (error: any) {
      console.error(`[BacktestService] 生成回撤曲线数据失败:`, error);
      if (error.response?.status === 404 || error.status === 404 || error.message?.includes('404')) {
        console.warn(`[BacktestService] 详细结果数据不存在，返回空数据`);
        return {
          dates: [],
          drawdowns: [],
          maxDrawdown: 0,
          maxDrawdownDate: '',
          maxDrawdownDuration: 0,
        };
      }
      throw error;
    }
  }

  /**
   * 生成月度热力图数据
   */
  private static async generateMonthlyHeatmapData(taskId: string): Promise<Record<string, any>> {
    try {
      const detailedResult = await this.getDetailedResult(taskId);
      
      if (!detailedResult || !detailedResult.monthly_returns || detailedResult.monthly_returns.length === 0) {
        console.warn(`[BacktestService] 月度收益数据为空: taskId=${taskId}`);
        return {
          monthlyReturns: [],
          years: [],
          months: Array.from({ length: 12 }, (_, i) => i + 1),
        };
      }
      
      const uniqueYears = new Set(detailedResult.monthly_returns.map((m: any) => m.year));
      const years = Array.from(uniqueYears).sort();
      
      return {
        monthlyReturns: detailedResult.monthly_returns,
        years: years,
        months: Array.from({ length: 12 }, (_, i) => i + 1),
      };
    } catch (error: any) {
      console.error(`[BacktestService] 生成月度热力图数据失败:`, error);
      if (error.response?.status === 404 || error.status === 404 || error.message?.includes('404')) {
        console.warn(`[BacktestService] 详细结果数据不存在，返回空数据`);
        return {
          monthlyReturns: [],
          years: [],
          months: Array.from({ length: 12 }, (_, i) => i + 1),
        };
      }
      throw error;
    }
  }

  /**
   * 生成交易分布数据
   */
  private static async generateTradeDistributionData(taskId: string): Promise<Record<string, any>> {
    const trades = await this.getTradeRecords(taskId, { limit: 1000 });
    const stats = await this.getTradeStatistics(taskId);
    
    return {
      trades: trades.trades,
      statistics: stats,
      profitDistribution: this.calculateProfitDistribution(trades.trades),
      timeDistribution: this.calculateTimeDistribution(trades.trades),
    };
  }

  /**
   * 生成持仓权重数据
   */
  private static async generatePositionWeightsData(taskId: string): Promise<Record<string, any>> {
    const detailedResult = await this.getDetailedResult(taskId);
    
    // 处理 position_analysis 可能是数组或对象的情况
    const positionAnalysis = detailedResult.position_analysis;
    const stockPerformance = Array.isArray(positionAnalysis)
      ? positionAnalysis
      : (positionAnalysis?.stock_performance || []);
    
    return {
      positionAnalysis: positionAnalysis,
      totalReturn: stockPerformance.reduce((sum, p) => sum + p.total_return, 0),
    };
  }

  /**
   * 计算盈亏分布
   */
  private static calculateProfitDistribution(trades: TradeRecord[]): Record<string, any> {
    const sellTrades = trades.filter(t => t.action === 'SELL' && t.pnl !== 0);
    const profits = sellTrades.map(t => t.pnl);
    
    // 创建盈亏区间
    const min = Math.min(...profits);
    const max = Math.max(...profits);
    const binCount = 20;
    const binSize = (max - min) / binCount;
    
    const bins = Array.from({ length: binCount }, (_, i) => ({
      range: [min + i * binSize, min + (i + 1) * binSize],
      count: 0,
    }));
    
    profits.forEach(profit => {
      const binIndex = Math.min(Math.floor((profit - min) / binSize), binCount - 1);
      bins[binIndex].count++;
    });
    
    return {
      bins,
      totalTrades: sellTrades.length,
      avgProfit: profits.reduce((sum, p) => sum + p, 0) / profits.length,
    };
  }

  /**
   * 计算时间分布
   */
  private static calculateTimeDistribution(trades: TradeRecord[]): Record<string, any> {
    const tradesByHour = Array.from({ length: 24 }, () => 0);
    const tradesByDay = Array.from({ length: 7 }, () => 0);
    const tradesByMonth = Array.from({ length: 12 }, () => 0);
    
    trades.forEach(trade => {
      const date = new Date(trade.timestamp);
      tradesByHour[date.getHours()]++;
      tradesByDay[date.getDay()]++;
      tradesByMonth[date.getMonth()]++;
    });
    
    return {
      hourly: tradesByHour,
      daily: tradesByDay,
      monthly: tradesByMonth,
    };
  }
}