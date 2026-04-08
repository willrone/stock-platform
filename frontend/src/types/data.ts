/**
 * 数据/监控/预测相关 API 合同类型。
 *
 * Backend source:
 * - backend/app/api/v1/data.py
 * - backend/app/api/v1/stocks.py
 * - backend/app/api/v1/signals.py
 * - backend/app/api/v1/system.py
 * - backend/app/api/v1/monitoring.py
 */

export interface StockDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  adj_close?: number;
}

export interface StockIndicators {
  ma_5?: number;
  ma_10?: number;
  ma_20?: number;
  ma_60?: number;
  rsi?: number;
  macd?: number;
  macd_signal?: number;
  bb_upper?: number;
  bb_lower?: number;
}

export interface StockData {
  stock_code: string;
  data: StockDataPoint[];
  indicators?: StockIndicators;
  last_updated: string;
}

export interface StockDataRequest {
  stock_code: string;
  start_date: string;
  end_date: string;
}

export interface TechnicalIndicators {
  stock_code: string;
  indicators: Required<StockIndicators>;
  calculation_date: string;
}

export interface PredictionRequest {
  stock_codes: string[];
  model_id: string;
  horizon: 'intraday' | 'short_term' | 'medium_term';
  confidence_level: number;
}

export interface PredictionResponse {
  predictions: Array<{
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
    };
  }>;
  model_id: string;
  horizon: string;
}

export interface LatestSignalItem {
  stock_code: string;
  latest_signal: 'BUY' | 'SELL' | 'HOLD';
  signal_date: string | null;
  strength: number;
  price: number | null;
  reason: string | null;
}

export interface LatestSignalsResponse {
  strategy_name: string;
  days: number;
  source: 'local' | 'remote';
  pagination: {
    total: number;
    limit: number;
    offset: number;
  };
  signals: LatestSignalItem[];
  failures?: string[];
}

export interface MultiLatestSignalPerStrategyItem {
  latest_signal: 'BUY' | 'SELL' | 'HOLD';
  signal_date: string | null;
  strength: number;
  price: number | null;
  reason: string | null;
}

export interface MultiLatestSignalRow {
  stock_code: string;
  stock_name?: string | null;
  per_strategy: {
    [strategyName: string]: MultiLatestSignalPerStrategyItem | null;
  };
}

export interface MultiLatestSignalsResponse {
  strategy_names: string[];
  days: number;
  source: 'local' | 'remote';
  pagination: {
    total: number;
    limit: number;
    offset: number;
  };
  signals: MultiLatestSignalRow[];
  failures?: string[];
}

export interface SignalEvent {
  timestamp: string;
  signal: 'BUY' | 'SELL';
  strength: number;
  price: number;
  reason: string;
  metadata?: Record<string, unknown>;
}

export interface SignalHistoryResponse {
  stock_code: string;
  strategy_name: string;
  days: number;
  events: SignalEvent[];
}

export interface MultiSignalHistoryResponse {
  stock_code: string;
  strategy_names: string[];
  days: number;
  events_by_strategy: Record<string, SignalEvent[]>;
}

export interface SystemStatus {
  api_server: { status: string; uptime: string };
  data_service: { status: string; last_update: string };
  prediction_engine: { status: string; active_models: number };
  task_manager: { status: string; running_tasks: number };
  database: { status: string; connection: string };
  remote_data_service: { status: string; url: string };
}

export interface DataServiceStatus {
  service_url: string;
  is_connected: boolean;
  last_check: string;
  response_time: number;
  error_message?: string;
}

export interface SyncProgress {
  sync_id: string;
  total_stocks: number;
  completed_stocks: number;
  failed_stocks: number;
  current_stock: string | null;
  progress_percentage: number;
  estimated_remaining_time_seconds: number | null;
  start_time: string;
  status: string;
  last_update: string;
}

export interface SystemHealth {
  overall_healthy: boolean;
  services: Record<
    string,
    {
      healthy: boolean;
      response_time_ms: number;
      last_check: string;
      error_message: string | null;
    }
  >;
  check_time: string;
}

export interface PerformanceMetrics {
  services?: Record<string, unknown>;
  summary?: {
    total_services: number;
    avg_response_time: number;
    total_requests: number;
    total_errors: number;
  };
}
