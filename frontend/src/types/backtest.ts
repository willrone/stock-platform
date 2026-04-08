/**
 * 回测相关 API 合同类型。
 *
 * Backend source:
 * - backend/app/api/v1/backtest.py
 * - backend/app/api/v1/backtest_detailed.py
 */

export interface BacktestRequest {
  strategy_name: string;
  stock_codes: string[];
  start_date: string;
  end_date: string;
  initial_cash: number;
}

export interface BacktestResult {
  strategy_name: string;
  period: {
    start_date: string;
    end_date: string;
  };
  portfolio: {
    initial_cash: number;
    final_value: number;
    total_return: number;
    annualized_return: number;
  };
  risk_metrics: {
    max_drawdown: number;
    sharpe_ratio: number;
    volatility: number;
  };
  trading_stats: {
    total_trades: number;
    win_rate: number;
    profit_factor: number;
  };
}

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
  position_analysis:
    | {
        stock_performance: Array<{
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
        }>;
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
      }
    | Array<{
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

export interface PortfolioSnapshot {
  id: number;
  task_id: string;
  snapshot_date: string;
  portfolio_value: number;
  cash: number;
  positions_count: number;
  total_return: number;
  daily_return: number;
  positions: Record<string, unknown>;
}

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
  metadata?: Record<string, unknown>;
  executed: boolean;
  execution_reason?: string;
  created_at: string;
}

export interface SignalStatistics {
  total_signals: number;
  buy_signals: number;
  sell_signals: number;
  executed_signals: number;
  unexecuted_signals: number;
  execution_rate: number;
  avg_strength: number;
}

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
