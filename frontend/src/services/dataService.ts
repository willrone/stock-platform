/**
 * 数据服务
 * 
 * 处理股票数据和系统状态相关的API调用，包括：
 * - 股票数据获取
 * - 技术指标计算
 * - 模型管理
 * - 系统状态监控
 */

import { apiRequest } from './api';
import { StockData, Model, SystemStatus } from '../stores/useDataStore';

// 股票数据请求参数
export interface StockDataRequest {
  stock_code: string;
  start_date: string;
  end_date: string;
}

// 技术指标响应
export interface TechnicalIndicators {
  stock_code: string;
  indicators: {
    ma_5: number;
    ma_10: number;
    ma_20: number;
    ma_60: number;
    rsi: number;
    macd: number;
    macd_signal: number;
    bb_upper: number;
    bb_lower: number;
  };
  calculation_date: string;
}

// 预测请求
export interface PredictionRequest {
  stock_codes: string[];
  model_id: string;
  horizon: 'intraday' | 'short_term' | 'medium_term';
  confidence_level: number;
}

// 预测响应
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

// 回测请求
export interface BacktestRequest {
  strategy_name: string;
  stock_codes: string[];
  start_date: string;
  end_date: string;
  initial_cash: number;
}

// 回测结果
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

// 策略信号 - 最新信号
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

// 多策略 - 最新信号：每只股票的多策略结果
export interface MultiLatestSignalPerStrategyItem {
  latest_signal: 'BUY' | 'SELL' | 'HOLD';
  signal_date: string | null;
  strength: number;
  price: number | null;
  reason: string | null;
}

export interface MultiLatestSignalRow {
  stock_code: string;
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

// 策略信号 - 历史事件
export interface SignalEvent {
  timestamp: string;
  signal: 'BUY' | 'SELL';
  strength: number;
  price: number;
  reason: string;
  metadata?: Record<string, any>;
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

// 数据服务类
export class DataService {
  /**
   * 获取股票历史数据
   */
  static async getStockData(
    stockCode: string,
    startDate: string,
    endDate: string
  ): Promise<StockData> {
    const params = {
      stock_code: stockCode,
      start_date: startDate,
      end_date: endDate,
    };
    
    try {
      const response = await apiRequest.get<any>('/stocks/data', params);
      
      console.log(`[DataService] getStockData 响应:`, response);
      
      // 转换为标准格式
      // response 是后端返回的 response_data: { stock_code, start_date, end_date, data_points, data: [...] }
      // 如果 response 为 null（后端返回 success=False 但 data 不为 null），则尝试从其他地方获取
      let dataArray: any[] = [];
      
      if (response) {
        if (Array.isArray(response.data)) {
          dataArray = response.data;
        } else if (Array.isArray(response)) {
          dataArray = response;
        }
      }
      
      console.log(`[DataService] 解析后的数据数组长度: ${dataArray.length}`, {
        responseType: typeof response,
        responseKeys: response ? Object.keys(response) : [],
        hasData: !!response?.data,
        dataIsArray: Array.isArray(response?.data)
      });
      
      return {
        stock_code: stockCode,
        data: dataArray,
        last_updated: new Date().toISOString(),
      };
    } catch (error: any) {
      console.error(`[DataService] getStockData 失败:`, error);
      console.error(`[DataService] 错误详情:`, {
        message: error?.message,
        status: error?.status,
        response: error?.response
      });
      // 返回空数据而不是抛出错误，让组件可以处理
      return {
        stock_code: stockCode,
        data: [],
        last_updated: new Date().toISOString(),
      };
    }
  }

  /**
   * 获取技术指标
   */
  static async getTechnicalIndicators(
    stockCode: string,
    startDate?: string,
    endDate?: string
  ): Promise<TechnicalIndicators> {
    const params: any = {};
    if (startDate) params.start_date = startDate;
    if (endDate) params.end_date = endDate;
    
    return apiRequest.get<TechnicalIndicators>(
      `/stocks/${stockCode}/indicators`,
      params
    );
  }

  /**
   * 批量获取股票数据
   */
  static async getBatchStockData(
    stockCodes: string[],
    startDate: string,
    endDate: string
  ): Promise<StockData[]> {
    const promises = stockCodes.map(code =>
      this.getStockData(code, startDate, endDate)
    );
    
    return Promise.all(promises);
  }

  /**
   * 搜索股票
   */
  static async searchStocks(keyword: string): Promise<Array<{
    code: string;
    name: string;
    market: string;
  }>> {
    const response = await apiRequest.get<{
      stocks: Array<{
        code: string;
        name: string;
        market: string;
      }>;
      total: number;
    }>('/stocks/search', { keyword });
    return response?.stocks ?? [];
  }

  /**
   * 获取热门股票
   */
  static async getPopularStocks(): Promise<Array<{
    code: string;
    name: string;
    market?: string;
    change_percent: number;
    volume: number;
  }>> {
    try {
      console.log('[DataService] 开始调用 /stocks/popular API...');
      const response = await apiRequest.get<{
        stocks: Array<{
          code: string;
          name: string;
          market?: string;
          change_percent: number;
          volume: number;
        }>;
        total: number;
      }>('/stocks/popular');
      
      console.log('[DataService] API响应:', response);
      console.log('[DataService] response类型:', typeof response);
      console.log('[DataService] response是否为数组:', Array.isArray(response));
      console.log('[DataService] response.stocks:', response?.stocks);
      
      if (!response) {
        console.warn('[DataService] 响应为空');
        return [];
      }
      
      // 处理响应可能是数组的情况
      if (Array.isArray(response)) {
        console.log('[DataService] 响应是数组，直接返回');
        return response;
      }
      
      // 处理响应是对象的情况
      if (response && typeof response === 'object' && 'stocks' in response) {
        const stocks = response.stocks;
        if (!stocks || !Array.isArray(stocks)) {
          console.warn('[DataService] stocks字段不存在或不是数组:', stocks);
          return [];
        }
        console.log(`[DataService] 成功获取 ${stocks.length} 只热门股票`);
        return stocks;
      }
      
      console.warn('[DataService] 响应格式不符合预期:', response);
      return [];
    } catch (error) {
      console.error('[DataService] 获取热门股票列表失败:', error);
      console.error('[DataService] 错误详情:', error instanceof Error ? error.message : String(error));
      throw error;
    }
  }

  /**
   * 创建预测
   */
  static async createPrediction(request: PredictionRequest): Promise<PredictionResponse> {
    return apiRequest.post<PredictionResponse>('/predictions', request);
  }

  /**
   * 获取预测结果
   */
  static async getPredictionResult(predictionId: string): Promise<PredictionResponse> {
    return apiRequest.get<PredictionResponse>(`/predictions/${predictionId}`);
  }

  /**
   * 运行回测
   */
  static async runBacktest(request: BacktestRequest): Promise<BacktestResult> {
    return apiRequest.post<BacktestResult>('/backtest', request);
  }

  /**
   * 获取可用策略列表（用于策略信号页）
   */
  static async getAvailableStrategies(): Promise<
    Array<{
      key: string;
      name: string;
      description?: string;
      category?: string;
      parameters?: Record<string, any>;
    }>
  > {
    const resp = await apiRequest.get<any>('/backtest/strategies');
    // 后端返回的是 { key, name, ... } 数组
    return Array.isArray(resp) ? resp : (resp ?? []);
  }

  /**
   * 获取全市场（分页）最新信号
   */
  static async getLatestSignals(params: {
    strategy_name: string;
    days?: number;
    source?: 'local' | 'remote';
    limit?: number;
    offset?: number;
  }): Promise<LatestSignalsResponse> {
    return apiRequest.get<LatestSignalsResponse>('/signals/latest', params);
  }

  /**
   * 获取全市场（分页）多策略最新信号
   */
  static async getLatestSignalsMulti(params: {
    strategy_names: string[];
    days?: number;
    source?: 'local' | 'remote';
    limit?: number;
    offset?: number;
  }): Promise<MultiLatestSignalsResponse> {
    return apiRequest.get<MultiLatestSignalsResponse>('/signals/latest-multi', params);
  }

  /**
   * 获取单只股票信号历史（近N个交易日 BUY/SELL 事件）
   */
  static async getSignalHistory(params: {
    stock_code: string;
    strategy_name: string;
    days?: number;
  }): Promise<SignalHistoryResponse> {
    return apiRequest.get<SignalHistoryResponse>('/signals/history', params);
  }

  /**
   * 获取单只股票多策略信号历史（近N个交易日 BUY/SELL 事件）
   */
  static async getSignalHistoryMulti(params: {
    stock_code: string;
    strategy_names: string[];
    days?: number;
  }): Promise<MultiSignalHistoryResponse> {
    return apiRequest.get<MultiSignalHistoryResponse>('/signals/history-multi', params);
  }

  /**
   * 获取模型列表
   */
  static async getModels(): Promise<{ models: Model[] }> {
    return apiRequest.get<{ models: Model[] }>('/models');
  }

  /**
   * 获取模型详情
   */
  static async getModelDetail(modelId: string): Promise<Model> {
    return apiRequest.get<Model>(`/models/${modelId}`);
  }

  /**
   * 删除模型
   */
  static async deleteModel(modelId: string): Promise<void> {
    return apiRequest.delete(`/models/${modelId}`);
  }

  /**
   * 获取模型训练报告
   */
  static async getTrainingReport(modelId: string): Promise<any> {
    const response = await apiRequest.get(`/models/${modelId}/evaluation-report`);
    return response.data || response;
  }

  /**
   * 获取可用特征列表
   */
  static async getAvailableFeatures(params?: {
    stock_code?: string;
    start_date?: string;
    end_date?: string;
  }): Promise<{
    features: string[];
    feature_count: number;
    feature_categories: {
      base_features: string[];
      indicator_features: string[];
      fundamental_features: string[];
      alpha_features: string[];
    };
    source: string;
  }> {
    const queryParams = new URLSearchParams();
    if (params?.stock_code) queryParams.append('stock_code', params.stock_code);
    if (params?.start_date) queryParams.append('start_date', params.start_date);
    if (params?.end_date) queryParams.append('end_date', params.end_date);
    
    const url = `/models/available-features${queryParams.toString() ? '?' + queryParams.toString() : ''}`;
    const response = await apiRequest.get<{
      success: boolean;
      message: string;
      data: {
        features: string[];
        feature_count: number;
        feature_categories: {
          base_features: string[];
          indicator_features: string[];
          fundamental_features: string[];
          alpha_features: string[];
        };
        source: string;
      };
    }>(url);
    
    // 处理响应格式：可能是 { data: {...} } 或直接是 { success, data, ... }
    if (response.data) {
      return response.data;
    } else if ((response as any).features) {
      // 如果响应直接包含features字段
      return response as any;
    } else {
      // 如果都没有，返回空数据
      return {
        features: [],
        feature_count: 0,
        feature_categories: {
          base_features: [],
          indicator_features: [],
          fundamental_features: [],
          alpha_features: [],
        },
        source: 'error'
      };
    }
  }

  /**
   * 创建模型训练任务
   */
  static async createModel(request: {
    model_name: string;
    model_type?: string;
    stock_codes: string[];
    start_date: string;
    end_date: string;
    hyperparameters?: Record<string, any>;
    selected_features?: string[];
    description?: string;
    parent_model_id?: string;
    enable_hyperparameter_tuning?: boolean;
    hyperparameter_search_strategy?: string;
    hyperparameter_search_trials?: number;
  }): Promise<{
    model_id: string;
    model_name: string;
    status: string;
  }> {
    return apiRequest.post('/models/train', request);
  }

  /**
   * 获取系统状态
   */
  static async getSystemStatus(): Promise<SystemStatus> {
    return apiRequest.get<SystemStatus>('/system/status');
  }

  /**
   * 获取API版本信息
   */
  static async getApiVersion(): Promise<{
    version: string;
    release_date: string;
    api_name: string;
    description: string;
    features: string[];
  }> {
    return apiRequest.get('/version');
  }

  /**
   * 健康检查
   */
  static async healthCheck(): Promise<{
    status: string;
    version: string;
  }> {
    return apiRequest.get('/health');
  }

  // 数据管理相关方法

  /**
   * 获取数据服务状态
   */
  static async getDataServiceStatus(): Promise<{
    service_url: string;
    is_connected: boolean;
    last_check: string;
    response_time: number;
    error_message?: string;
  }> {
    return apiRequest.get('/data/status');
  }

  /**
   * 获取本地数据文件列表
   */
  static async getLocalDataFiles(params?: {
    stock_code?: string;
    limit?: number;
    offset?: number;
  }): Promise<{
    files: Array<{
      stock_code: string;
      file_path: string;
      file_size: number;
      last_updated: string;
      record_count: number;
      date_range: {
        start: string;
        end: string;
      };
    }>;
    total: number;
    limit: number;
    offset: number;
  }> {
    return apiRequest.get('/data/files', params);
  }

  /**
   * 获取数据统计信息
   */
  static async getDataStatistics(): Promise<{
    total_files: number;
    total_size: number;
    total_records: number;
    stock_count: number;
    last_sync: string;
    date_range: {
      start: string;
      end: string;
    };
  }> {
    return apiRequest.get('/data/stats');
  }

  /**
   * 同步数据
   */
  static async syncDataFromRemote(params: {
    stock_codes: string[];
    start_date?: string;
    end_date?: string;
    force_update?: boolean;
  }): Promise<{
    success: boolean;
    synced_stocks: string[];
    failed_stocks: string[];
    total_records: number;
    sync_duration: string;
    message: string;
  }> {
    return apiRequest.post('/data/sync', params);
  }

  /**
   * 获取远端服务的股票列表
   */
  static async getRemoteStockList(): Promise<{
    stocks: Array<{
      ts_code: string;
      name?: string;
      data_range?: {
        start_date: string;
        end_date: string;
        total_days: number;
      };
      last_update?: string;
      status?: string;
    }>;
    stock_codes: string[];
    total_stocks: number;
  }> {
    return apiRequest.get('/data/remote/stocks');
  }

  /**
   * 获取本地股票列表（快速版，仅用于选择股票）
   */
  static async getLocalStockList(): Promise<{
    stocks: Array<{
      ts_code: string;
      name?: string;
      data_range?: {
        start_date: string;
        end_date: string;
        total_days: number;
      };
      file_count?: number;
      total_size?: number;
      record_count?: number;
    }>;
    stock_codes: string[];
    total_stocks: number;
  }> {
    // 使用快速接口，只获取股票代码和名称
    return apiRequest.get('/data/local/stocks/simple');
  }

  /**
   * 获取本地股票列表（详细版，包含所有信息）
   */
  static async getLocalStockListDetailed(): Promise<{
    stocks: Array<{
      ts_code: string;
      name?: string;
      data_range?: {
        start_date: string;
        end_date: string;
        total_days: number;
      };
      file_count?: number;
      total_size?: number;
      record_count?: number;
    }>;
    stock_codes: string[];
    total_stocks: number;
  }> {
    return apiRequest.get('/data/local/stocks');
  }

  /**
   * 删除数据文件
   */
  static async deleteDataFiles(filePaths: string[]): Promise<{
    success: boolean;
    deleted_files: string[];
    failed_files: Array<{
      file_path: string;
      error: string;
    }>;
    total_deleted: number;
    freed_space_bytes: number;
    freed_space_mb: number;
    message: string;
  }> {
    // 将文件路径作为查询参数传递
    const params = new URLSearchParams();
    filePaths.forEach(path => params.append('file_paths', path));
    
    return apiRequest.delete(`/data/files?${params.toString()}`);
  }

  /**
   * 获取同步进度
   */
  static async getSyncProgress(syncId: string): Promise<{
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
  }> {
    return apiRequest.get(`/data/sync/${syncId}/progress`);
  }

  /**
   * 获取同步历史
   */
  static async getSyncHistory(limit: number = 50): Promise<{
    history: Array<{
      sync_id: string;
      request: {
        stock_codes: string[];
        start_date: string | null;
        end_date: string | null;
        force_update: boolean;
        sync_mode: string;
        max_concurrent: number;
        retry_count: number;
      };
      result: {
        success: boolean;
        total_stocks: number;
        success_count: number;
        failure_count: number;
        total_records: number;
        message: string;
      };
      created_at: string;
    }>;
    total: number;
    limit: number;
  }> {
    return apiRequest.get('/data/sync/history', { limit });
  }

  /**
   * 重试失败的同步
   */
  static async retrySyncFailed(syncId: string): Promise<{
    sync_id: string;
    retried_stocks: string[];
    retry_results: Array<{
      stock_code: string;
      success: boolean;
      records_synced: number;
      error_message: string | null;
    }>;
    success: boolean;
    message: string;
  }> {
    return apiRequest.post(`/data/sync/${syncId}/retry`);
  }

  /**
   * 获取系统健康状态
   */
  static async getSystemHealth(): Promise<{
    overall_healthy: boolean;
    services: Record<string, {
      healthy: boolean;
      response_time_ms: number;
      last_check: string;
      error_message: string | null;
    }>;
    check_time: string;
  }> {
    return apiRequest.get('/monitoring/health');
  }

  /**
   * 获取性能指标
   */
  static async getPerformanceMetrics(serviceName?: string): Promise<{
    services?: Record<string, any>;
    summary?: {
      total_services: number;
      avg_response_time: number;
      total_requests: number;
      total_errors: number;
    };
  }> {
    const params = serviceName ? { service_name: serviceName } : {};
    return apiRequest.get('/monitoring/metrics', params);
  }

  /**
   * 获取系统概览
   */
  static async getSystemOverview(): Promise<any> {
    return apiRequest.get('/monitoring/overview');
  }

  /**
   * 获取错误统计
   */
  static async getErrorStatistics(hours: number = 24): Promise<{
    time_range_hours: number;
    total_error_types: number;
    total_errors: number;
    error_statistics: Array<{
      error_type: string;
      count: number;
      last_occurrence: string;
      sample_message: string;
    }>;
  }> {
    return apiRequest.get('/monitoring/errors', { hours });
  }

  /**
   * 获取数据质量检查结果
   */
  static async getDataQuality(): Promise<any> {
    return apiRequest.get('/monitoring/quality');
  }

  /**
   * 获取异常检测结果
   */
  static async getAnomalies(): Promise<{
    total_anomalies: number;
    by_severity: {
      high: number;
      medium: number;
      low: number;
    };
    anomalies: Array<any>;
    detection_time: string;
  }> {
    return apiRequest.get('/monitoring/anomalies');
  }

  /**
   * 同步远端数据（通过SFTP）
   */
  static async syncRemoteData(stockCodes?: string[]): Promise<{
    success: boolean;
    total_files: number;
    synced_files: number;
    failed_files: string[];
    total_size: number;
    total_size_mb: number;
    message: string;
  }> {
    return apiRequest.post('/data/sync/remote', {
      stock_codes: stockCodes || null,
    });
  }
}
