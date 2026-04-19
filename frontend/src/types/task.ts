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
