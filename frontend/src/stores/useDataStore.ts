/**
 * 数据管理状态
 * 
 * 管理股票数据和系统状态，包括：
 * - 股票数据缓存
 * - 系统状态监控
 * - 模型信息
 * - 数据服务状态
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
  status: 'active' | 'inactive' | 'training';
  description?: string;
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

interface DataState {
  // 股票数据
  stockDataCache: Map<string, StockData>;
  
  // 模型信息
  models: Model[];
  selectedModel: Model | null;
  
  // 系统状态
  systemStatus: SystemStatus | null;
  
  // 加载状态
  loadingStockData: boolean;
  loadingModels: boolean;
  loadingSystemStatus: boolean;
  
  // Actions
  setStockData: (stockCode: string, data: StockData) => void;
  getStockData: (stockCode: string) => StockData | undefined;
  setModels: (models: Model[]) => void;
  setSelectedModel: (model: Model | null) => void;
  setSystemStatus: (status: SystemStatus) => void;
  setLoadingStockData: (loading: boolean) => void;
  setLoadingModels: (loading: boolean) => void;
  setLoadingSystemStatus: (loading: boolean) => void;
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
      loadingStockData: false,
      loadingModels: false,
      loadingSystemStatus: false,

      // Actions
      setStockData: (stockCode, data) => set((state) => {
        const newCache = new Map(state.stockDataCache);
        newCache.set(stockCode, data);
        return { stockDataCache: newCache };
      }, false, 'setStockData'),

      getStockData: (stockCode) => {
        return get().stockDataCache.get(stockCode);
      },

      setModels: (models) => set({
        models,
        loadingModels: false,
      }, false, 'setModels'),

      setSelectedModel: (selectedModel) => set({
        selectedModel,
      }, false, 'setSelectedModel'),

      setSystemStatus: (systemStatus) => set({
        systemStatus,
        loadingSystemStatus: false,
      }, false, 'setSystemStatus'),

      setLoadingStockData: (loadingStockData) => set({
        loadingStockData,
      }, false, 'setLoadingStockData'),

      setLoadingModels: (loadingModels) => set({
        loadingModels,
      }, false, 'setLoadingModels'),

      setLoadingSystemStatus: (loadingSystemStatus) => set({
        loadingSystemStatus,
      }, false, 'setLoadingSystemStatus'),

      clearStockDataCache: () => set({
        stockDataCache: new Map(),
      }, false, 'clearStockDataCache'),
    }),
    {
      name: 'data-store',
    }
  )
);