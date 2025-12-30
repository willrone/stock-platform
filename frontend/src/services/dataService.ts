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
    
    const response = await apiRequest.get<any>('/stocks/data', params);
    
    // 转换为标准格式
    return {
      stock_code: stockCode,
      data: response.data || [],
      last_updated: new Date().toISOString(),
    };
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
    return apiRequest.get('/stocks/search', { keyword });
  }

  /**
   * 获取热门股票
   */
  static async getPopularStocks(): Promise<Array<{
    code: string;
    name: string;
    change_percent: number;
    volume: number;
  }>> {
    return apiRequest.get('/stocks/popular');
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
   * 删除数据文件
   */
  static async deleteDataFiles(stockCodes: string[]): Promise<{
    deleted_files: string[];
    failed_files: string[];
    total_deleted: number;
  }> {
    // 将股票代码作为查询参数传递
    const params = new URLSearchParams();
    stockCodes.forEach(code => params.append('stock_codes', code));
    
    return apiRequest.delete(`/data/files?${params.toString()}`);
  }
}