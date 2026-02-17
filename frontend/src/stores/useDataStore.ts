/**
 * 数据管理状态
 *
 * 管理股票数据和系统状态，包括：
 * - 股票数据缓存
 * - 系统状态监控
 * - 模型信息
 * - 数据服务状态
 * - 同步进度跟踪
 * - 监控数据缓存
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

export interface StockData {
  stock_code: string;
  data: Array<{
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
    adj_close?: number;
  }>;
  indicators?: {
    ma_5?: number;
    ma_10?: number;
    ma_20?: number;
    ma_60?: number;
    rsi?: number;
    macd?: number;
    macd_signal?: number;
    bb_upper?: number;
    bb_lower?: number;
  };
  last_updated: string;
}

export interface Model {
  model_id: string;
  model_name: string;
  model_type: string;
  version: string;
  accuracy: number;
  created_at: string;
  status: 'active' | 'inactive' | 'training' | 'ready' | 'failed';
  description?: string;
  training_progress?: number;
  training_stage?: string;
  performance_metrics?: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    sharpe_ratio: number;
    max_drawdown: number;
  };
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

interface DataState {
  // 股票数据
  stockDataCache: Map<string, StockData>;

  // 模型信息
  models: Model[];
  selectedModel: Model | null;

  // 系统状态
  systemStatus: SystemStatus | null;
  dataServiceStatus: DataServiceStatus | null;

  // 监控数据
  systemHealth: SystemHealth | null;
  performanceMetrics: PerformanceMetrics | null;

  // 同步状态
  activeSyncs: Map<string, SyncProgress>;

  // 加载状态
  loadingStockData: boolean;
  loadingModels: boolean;
  loadingSystemStatus: boolean;
  loadingHealth: boolean;
  loadingMetrics: boolean;

  // Actions
  setStockData: (stockCode: string, data: StockData) => void;
  getStockData: (stockCode: string) => StockData | undefined;
  setModels: (models: Model[]) => void;
  setSelectedModel: (model: Model | null) => void;
  setSystemStatus: (status: SystemStatus) => void;
  setDataServiceStatus: (status: DataServiceStatus) => void;
  setSystemHealth: (health: SystemHealth) => void;
  setPerformanceMetrics: (metrics: PerformanceMetrics) => void;
  setSyncProgress: (syncId: string, progress: SyncProgress) => void;
  removeSyncProgress: (syncId: string) => void;
  setLoadingStockData: (loading: boolean) => void;
  setLoadingModels: (loading: boolean) => void;
  setLoadingSystemStatus: (loading: boolean) => void;
  setLoadingHealth: (loading: boolean) => void;
  setLoadingMetrics: (loading: boolean) => void;
  clearStockDataCache: () => void;
}

export const useDataStore = create<DataState>()(
  devtools(
    (set, get) => ({
      // 初始状态
      stockDataCache: new Map(),
      models: [],
      selectedModel: null,
      systemStatus: null,
      dataServiceStatus: null,
      systemHealth: null,
      performanceMetrics: null,
      activeSyncs: new Map(),
      loadingStockData: false,
      loadingModels: false,
      loadingSystemStatus: false,
      loadingHealth: false,
      loadingMetrics: false,

      // Actions
      setStockData: (stockCode, data) =>
        set(
          state => {
            const newCache = new Map(state.stockDataCache);
            newCache.set(stockCode, data);
            return { stockDataCache: newCache };
          },
          false,
          'setStockData'
        ),

      getStockData: stockCode => {
        return get().stockDataCache.get(stockCode);
      },

      setModels: models =>
        set(
          {
            models,
            loadingModels: false,
          },
          false,
          'setModels'
        ),

      setSelectedModel: selectedModel =>
        set(
          {
            selectedModel,
          },
          false,
          'setSelectedModel'
        ),

      setSystemStatus: systemStatus =>
        set(
          {
            systemStatus,
            loadingSystemStatus: false,
          },
          false,
          'setSystemStatus'
        ),

      setDataServiceStatus: dataServiceStatus =>
        set(
          {
            dataServiceStatus,
          },
          false,
          'setDataServiceStatus'
        ),

      setSystemHealth: systemHealth =>
        set(
          {
            systemHealth,
            loadingHealth: false,
          },
          false,
          'setSystemHealth'
        ),

      setPerformanceMetrics: performanceMetrics =>
        set(
          {
            performanceMetrics,
            loadingMetrics: false,
          },
          false,
          'setPerformanceMetrics'
        ),

      setSyncProgress: (syncId, progress) =>
        set(
          state => {
            const newSyncs = new Map(state.activeSyncs);
            newSyncs.set(syncId, progress);
            return { activeSyncs: newSyncs };
          },
          false,
          'setSyncProgress'
        ),

      removeSyncProgress: syncId =>
        set(
          state => {
            const newSyncs = new Map(state.activeSyncs);
            newSyncs.delete(syncId);
            return { activeSyncs: newSyncs };
          },
          false,
          'removeSyncProgress'
        ),

      setLoadingStockData: loadingStockData =>
        set(
          {
            loadingStockData,
          },
          false,
          'setLoadingStockData'
        ),

      setLoadingModels: loadingModels =>
        set(
          {
            loadingModels,
          },
          false,
          'setLoadingModels'
        ),

      setLoadingSystemStatus: loadingSystemStatus =>
        set(
          {
            loadingSystemStatus,
          },
          false,
          'setLoadingSystemStatus'
        ),

      setLoadingHealth: loadingHealth =>
        set(
          {
            loadingHealth,
          },
          false,
          'setLoadingHealth'
        ),

      setLoadingMetrics: loadingMetrics =>
        set(
          {
            loadingMetrics,
          },
          false,
          'setLoadingMetrics'
        ),

      clearStockDataCache: () =>
        set(
          {
            stockDataCache: new Map(),
          },
          false,
          'clearStockDataCache'
        ),
    }),
    {
      name: 'data-store',
    }
  )
);
