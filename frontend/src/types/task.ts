/**
 * 任务列表接口返回的指标摘要（后端 `extract_result_summary` 产出）。
 *
 * 列表接口默认 `exclude_result=True`，因此 **`result` / `results` 大字段不存在**，
 * 只有这个精简摘要可用。前端读取回测指标必须优先读 `result_summary`。
 */
export interface TaskResultSummary {
  total_return?: number;
  annualized_return?: number;
  sharpe_ratio?: number;
  max_drawdown?: number;
  total_trades?: number;
  win_rate?: number;
  volatility?: number;
  profit_factor?: number;
  final_value?: number;
  /**
   * 净值曲线（后端可选提供）。缺失时 UI 必须优雅降级为「—」，
   * 不得用任何占位数字伪造曲线。
   */
  nav_history?: Array<{ date: string; value: number }>;
  /**
   * 年化换手次数（次/年）。后端可选提供，缺失时降级为「—」。
   */
  turnover?: number;
  [key: string]: unknown;
}

export interface Task {
  task_id: string;
  task_name: string;
  task_type?: string;
  status: 'created' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused';
  progress: number;
  stock_codes: string[];
  description?: string;
  model_id: string;
  created_at: string;
  completed_at?: string;
  error_message?: string;
  result?: any;
  backtest_results?: any;
  /** 列表接口提供的指标摘要（P0-03 的正确数据源） */
  result_summary?: TaskResultSummary | null;
  config?: {
    backtest_config?: {
      strategy_name?: string;
      strategy_config?: Record<string, any>;
      start_date?: string;
      end_date?: string;
      initial_cash?: number;
      commission_rate?: number;
      slippage_rate?: number;
    };
    prediction_config?: any;
    optimization_config?: any;
    stock_codes?: string[];
    model_id?: string;
    [key: string]: any;
  };
  results?: {
    total_stocks: number;
    successful_predictions: number;
    average_confidence: number;
    backtest_results?: any;
    predictions: Array<{
      stock_code: string;
      predicted_direction: number;
      predicted_return?: number;
      confidence_score: number;
      confidence_interval?: {
        lower: number;
        upper: number;
      };
      risk_assessment?: {
        value_at_risk: number;
        volatility: number;
        [key: string]: any;
      };
      [key: string]: any;
    }>;
  };
  optimization_info?: {
    n_trials: number;
    completed_trials: number;
    running_trials?: number;
    pruned_trials?: number;
    failed_trials?: number;
  };
}

export interface TaskListResponse {
  tasks: Task[];
  total: number;
  limit: number;
  offset: number;
}

export interface CreateTaskRequest {
  task_name: string;
  task_type?: 'prediction' | 'backtest';
  stock_codes: string[];
  model_id?: string;
  prediction_config?: {
    horizon?: 'intraday' | 'short_term' | 'medium_term';
    confidence_level?: number;
    risk_assessment?: boolean;
  };
  backtest_config?: {
    strategy_name?: string;
    backtest_mode?: 'strategy' | 'model';
    model_id?: string;
    start_date: string;
    end_date: string;
    initial_cash?: number;
    commission_rate?: number;
    slippage_rate?: number;
    strategy_config?: Record<string, any>;
    enable_performance_profiling?: boolean;
  };
}

export interface PredictionResult {
  stock_code: string;
  predicted_direction: number;
  predicted_return: number;
  confidence_score: number;
  confidence_interval: {
    lower: number;
    upper: number;
  };
  risk_assessment: {
    value_at_risk: number;
    volatility: number;
    max_drawdown: number;
    sharpe_ratio: number;
  };
  technical_indicators?: {
    [key: string]: number | string;
  };
}
