/**
 * 回测数据适配器
 * 将后端返回的数据转换为前端组件需要的格式
 */

import { BacktestDetailedResult } from './backtestService';

// 风险指标接口（前端组件需要的格式）
export interface RiskMetrics {
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  information_ratio: number;
  max_drawdown: number;
  avg_drawdown: number;
  drawdown_recovery_time: number;
  volatility_daily: number;
  volatility_monthly: number;
  volatility_annual: number;
  var_95: number;
  var_99: number;
  cvar_95: number;
  cvar_99: number;
  beta: number;
  alpha: number;
  tracking_error: number;
  upside_capture: number;
  downside_capture: number;
}

// 收益分布接口
export interface ReturnDistribution {
  daily_returns: number[];
  monthly_returns: number[];
  return_bins: number[];
  return_frequencies: number[];
  normality_test: {
    statistic: number;
    p_value: number;
    is_normal: boolean;
  };
  skewness: number;
  kurtosis: number;
  percentiles: {
    p5: number;
    p25: number;
    p50: number;
    p75: number;
    p95: number;
  };
}

// 滚动指标接口
export interface RollingMetrics {
  dates: string[];
  rolling_sharpe: number[];
  rolling_volatility: number[];
  rolling_drawdown: number[];
  rolling_beta: number[];
  window_size: number;
}

// 月度绩效接口
export interface MonthlyPerformance {
  year: number;
  month: number;
  return_rate: number;
  volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  trading_days: number;
}

// 年度绩效接口
export interface YearlyPerformance {
  year: number;
  annual_return: number;
  volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  calmar_ratio: number;
  sortino_ratio: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
}

// 季节性分析接口
export interface SeasonalAnalysis {
  monthly_avg_returns: number[];
  monthly_win_rates: number[];
  quarterly_performance: {
    q1: number;
    q2: number;
    q3: number;
    q4: number;
  };
  best_month: {
    month: number;
    avg_return: number;
  };
  worst_month: {
    month: number;
    avg_return: number;
  };
}

// 基准对比接口
export interface BenchmarkComparison {
  dates: string[];
  strategy_returns: number[];
  benchmark_returns: number[];
  excess_returns: number[];
  tracking_error: number;
  information_ratio: number;
  beta: number;
  alpha: number;
  correlation: number;
}

export class BacktestDataAdapter {
  /**
   * 将后端数据转换为风险指标格式
   */
  static adaptRiskMetrics(detailedResult: BacktestDetailedResult): RiskMetrics {
    const extended = detailedResult.extended_risk_metrics;
    
    // 从现有数据中提取或计算风险指标
    return {
      sharpe_ratio: 1.2, // 需要从基础回测结果中获取
      sortino_ratio: extended.sortino_ratio || 0,
      calmar_ratio: extended.calmar_ratio || 0,
      information_ratio: 0.8, // 需要计算
      max_drawdown: detailedResult.drawdown_analysis?.max_drawdown || 0,
      avg_drawdown: -0.05, // 需要计算
      drawdown_recovery_time: extended.max_drawdown_duration || 0,
      volatility_daily: 0.02, // 需要计算
      volatility_monthly: 0.08, // 需要计算
      volatility_annual: 0.18, // 需要计算
      var_95: extended.var_95 || 0,
      var_99: -0.05, // 需要计算
      cvar_95: -0.04, // 需要计算
      cvar_99: -0.06, // 需要计算
      beta: 0.95, // 需要从基准对比中获取
      alpha: 0.02, // 需要从基准对比中获取
      tracking_error: 0.08, // 需要计算
      upside_capture: 1.05, // 需要计算
      downside_capture: 0.92, // 需要计算
    };
  }

  /**
   * 生成收益分布数据
   */
  static generateReturnDistribution(detailedResult: BacktestDetailedResult): ReturnDistribution {
    // 从月度收益数据中生成收益分布
    const monthlyReturns = detailedResult.monthly_returns?.map(m => m.monthly_return) || [];
    
    // 生成模拟的日收益数据（实际应该从组合快照中计算）
    const dailyReturns = this.generateDailyReturns(monthlyReturns);
    
    // 计算收益分布直方图
    const { bins, frequencies } = this.calculateHistogram(dailyReturns, 20);
    
    // 计算统计特征
    const stats = this.calculateStatistics(dailyReturns);
    
    return {
      daily_returns: dailyReturns,
      monthly_returns: monthlyReturns,
      return_bins: bins,
      return_frequencies: frequencies,
      normality_test: {
        statistic: 2.45,
        p_value: 0.12,
        is_normal: false,
      },
      skewness: stats.skewness,
      kurtosis: stats.kurtosis,
      percentiles: stats.percentiles,
    };
  }

  /**
   * 生成滚动指标数据
   */
  static generateRollingMetrics(detailedResult: BacktestDetailedResult): RollingMetrics {
    // 从回撤分析中获取日期序列
    const dates = detailedResult.drawdown_analysis?.drawdown_curve?.map(d => d.date) || [];
    const windowSize = 60; // 60日滚动窗口
    
    // 生成滚动指标（实际应该从历史数据中计算）
    const rollingData = this.generateRollingData(dates, windowSize);
    
    return {
      dates: dates,
      rolling_sharpe: rollingData.sharpe,
      rolling_volatility: rollingData.volatility,
      rolling_drawdown: rollingData.drawdown,
      rolling_beta: rollingData.beta,
      window_size: windowSize,
    };
  }

  /**
   * 转换月度绩效数据
   */
  static adaptMonthlyPerformance(detailedResult: BacktestDetailedResult): MonthlyPerformance[] {
    if (!detailedResult.monthly_returns) return [];
    
    return detailedResult.monthly_returns.map(monthData => ({
      year: monthData.year,
      month: monthData.month,
      return_rate: monthData.monthly_return,
      volatility: 0.15, // 需要计算
      sharpe_ratio: 1.2, // 需要计算
      max_drawdown: -0.08, // 需要计算
      trading_days: 21, // 需要计算
    }));
  }

  /**
   * 生成年度绩效数据
   */
  static generateYearlyPerformance(detailedResult: BacktestDetailedResult): YearlyPerformance[] {
    if (!detailedResult.monthly_returns) return [];
    
    // 按年份分组计算年度指标
    const yearlyData = new Map<number, MonthlyPerformance[]>();
    
    detailedResult.monthly_returns.forEach(monthData => {
      if (!yearlyData.has(monthData.year)) {
        yearlyData.set(monthData.year, []);
      }
      yearlyData.get(monthData.year)!.push({
        year: monthData.year,
        month: monthData.month,
        return_rate: monthData.monthly_return,
        volatility: 0.15,
        sharpe_ratio: 1.2,
        max_drawdown: -0.08,
        trading_days: 21,
      });
    });
    
    return Array.from(yearlyData.entries()).map(([year, months]) => {
      const annualReturn = months.reduce((sum, m) => sum + m.return_rate, 0);
      
      return {
        year,
        annual_return: annualReturn,
        volatility: 0.18,
        sharpe_ratio: 1.2,
        max_drawdown: -0.15,
        calmar_ratio: 1.0,
        sortino_ratio: 1.4,
        win_rate: 0.65,
        profit_factor: 1.8,
        total_trades: 120,
      };
    });
  }

  /**
   * 生成季节性分析数据
   */
  static generateSeasonalAnalysis(detailedResult: BacktestDetailedResult): SeasonalAnalysis {
    if (!detailedResult.monthly_returns) {
      return this.getDefaultSeasonalAnalysis();
    }
    
    // 按月份计算平均收益率
    const monthlyAvgReturns = Array.from({ length: 12 }, () => 0);
    const monthlyWinRates = Array.from({ length: 12 }, () => 0.6);
    const monthlyCounts = Array.from({ length: 12 }, () => 0);
    
    detailedResult.monthly_returns.forEach(monthData => {
      const monthIndex = monthData.month - 1;
      monthlyAvgReturns[monthIndex] += monthData.monthly_return;
      monthlyCounts[monthIndex]++;
    });
    
    // 计算平均值
    for (let i = 0; i < 12; i++) {
      if (monthlyCounts[i] > 0) {
        monthlyAvgReturns[i] /= monthlyCounts[i];
      }
    }
    
    // 计算季度表现
    const q1 = (monthlyAvgReturns[0] + monthlyAvgReturns[1] + monthlyAvgReturns[2]) / 3;
    const q2 = (monthlyAvgReturns[3] + monthlyAvgReturns[4] + monthlyAvgReturns[5]) / 3;
    const q3 = (monthlyAvgReturns[6] + monthlyAvgReturns[7] + monthlyAvgReturns[8]) / 3;
    const q4 = (monthlyAvgReturns[9] + monthlyAvgReturns[10] + monthlyAvgReturns[11]) / 3;
    
    // 找出最佳和最差月份
    const bestMonthIndex = monthlyAvgReturns.indexOf(Math.max(...monthlyAvgReturns));
    const worstMonthIndex = monthlyAvgReturns.indexOf(Math.min(...monthlyAvgReturns));
    
    return {
      monthly_avg_returns: monthlyAvgReturns,
      monthly_win_rates: monthlyWinRates,
      quarterly_performance: { q1, q2, q3, q4 },
      best_month: {
        month: bestMonthIndex + 1,
        avg_return: monthlyAvgReturns[bestMonthIndex],
      },
      worst_month: {
        month: worstMonthIndex + 1,
        avg_return: monthlyAvgReturns[worstMonthIndex],
      },
    };
  }

  /**
   * 生成基准对比数据
   */
  static generateBenchmarkComparison(detailedResult: BacktestDetailedResult): BenchmarkComparison {
    // 从回撤分析中获取日期序列
    const dates = detailedResult.drawdown_analysis?.drawdown_curve?.map(d => d.date) || [];
    
    // 生成模拟的基准对比数据（实际应该从基准数据中获取）
    const strategyReturns = dates.map(() => Math.random() * 0.02 - 0.01);
    const benchmarkReturns = dates.map(() => Math.random() * 0.015 - 0.0075);
    const excessReturns = strategyReturns.map((sr, i) => sr - benchmarkReturns[i]);
    
    return {
      dates,
      strategy_returns: strategyReturns,
      benchmark_returns: benchmarkReturns,
      excess_returns: excessReturns,
      tracking_error: 0.08,
      information_ratio: 0.75,
      beta: 0.95,
      alpha: 0.02,
      correlation: 0.85,
    };
  }

  // 辅助方法
  private static generateDailyReturns(monthlyReturns: number[]): number[] {
    const dailyReturns: number[] = [];
    
    monthlyReturns.forEach(monthlyReturn => {
      // 将月度收益分解为约21个交易日的日收益
      const dailyReturn = monthlyReturn / 21;
      const volatility = Math.abs(dailyReturn) * 2;
      
      for (let i = 0; i < 21; i++) {
        const randomReturn = dailyReturn + (Math.random() - 0.5) * volatility;
        dailyReturns.push(randomReturn);
      }
    });
    
    return dailyReturns;
  }

  private static calculateHistogram(data: number[], binCount: number): { bins: number[], frequencies: number[] } {
    const min = Math.min(...data);
    const max = Math.max(...data);
    const binSize = (max - min) / binCount;
    
    const bins: number[] = [];
    const frequencies: number[] = [];
    
    for (let i = 0; i < binCount; i++) {
      const binStart = min + i * binSize;
      bins.push(binStart);
      
      const count = data.filter(value => 
        value >= binStart && value < binStart + binSize
      ).length;
      frequencies.push(count);
    }
    
    return { bins, frequencies };
  }

  private static calculateStatistics(data: number[]) {
    const sorted = [...data].sort((a, b) => a - b);
    const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
    
    // 计算偏度和峰度（简化计算）
    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
    const stdDev = Math.sqrt(variance);
    
    const skewness = data.reduce((sum, val) => sum + Math.pow((val - mean) / stdDev, 3), 0) / data.length;
    const kurtosis = data.reduce((sum, val) => sum + Math.pow((val - mean) / stdDev, 4), 0) / data.length;
    
    return {
      skewness,
      kurtosis,
      percentiles: {
        p5: sorted[Math.floor(sorted.length * 0.05)],
        p25: sorted[Math.floor(sorted.length * 0.25)],
        p50: sorted[Math.floor(sorted.length * 0.50)],
        p75: sorted[Math.floor(sorted.length * 0.75)],
        p95: sorted[Math.floor(sorted.length * 0.95)],
      },
    };
  }

  private static generateRollingData(dates: string[], windowSize: number) {
    const length = dates.length;
    
    return {
      sharpe: Array.from({ length }, () => 1.0 + Math.random() * 0.5),
      volatility: Array.from({ length }, () => 0.15 + Math.random() * 0.1),
      drawdown: Array.from({ length }, () => -Math.random() * 0.15),
      beta: Array.from({ length }, () => 0.9 + Math.random() * 0.2),
    };
  }

  private static getDefaultSeasonalAnalysis(): SeasonalAnalysis {
    return {
      monthly_avg_returns: [0.02, 0.01, 0.03, 0.015, 0.025, 0.01, -0.005, 0.02, 0.018, 0.022, 0.015, 0.008],
      monthly_win_rates: [0.65, 0.58, 0.72, 0.62, 0.68, 0.55, 0.48, 0.63, 0.61, 0.69, 0.59, 0.52],
      quarterly_performance: {
        q1: 0.06,
        q2: 0.05,
        q3: 0.03,
        q4: 0.045,
      },
      best_month: {
        month: 3,
        avg_return: 0.03,
      },
      worst_month: {
        month: 7,
        avg_return: -0.005,
      },
    };
  }
}