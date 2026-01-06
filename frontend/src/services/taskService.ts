/**
 * 任务管理服务
 * 
 * 处理与任务相关的API调用，包括：
 * - 任务创建
 * - 任务查询
 * - 任务状态更新
 * - 任务结果获取
 */

import { apiRequest } from './api';
import { Task } from '../stores/useTaskStore';

// 任务创建请求
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
    start_date: string;
    end_date: string;
    initial_cash?: number;
    commission_rate?: number;
    slippage_rate?: number;
    strategy_config?: Record<string, any>;
  };
}

// 任务列表响应
export interface TaskListResponse {
  tasks: Task[];
  total: number;
  limit: number;
  offset: number;
}

// 预测结果
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

// 任务服务类
export class TaskService {
  /**
   * 创建新的预测任务
   */
  static async createTask(request: CreateTaskRequest): Promise<Task> {
    return apiRequest.post<Task>('/tasks', request);
  }

  /**
   * 获取任务列表
   */
  static async getTasks(
    status?: string,
    limit: number = 20,
    offset: number = 0
  ): Promise<TaskListResponse> {
    const params: any = { limit, offset };
    if (status) {
      params.status = status;
    }
    
    return apiRequest.get<TaskListResponse>('/tasks', params);
  }

  /**
   * 获取任务详情
   */
  static async getTaskDetail(taskId: string): Promise<Task> {
    return apiRequest.get<Task>(`/tasks/${taskId}`);
  }

  /**
   * 删除任务
   */
  static async deleteTask(taskId: string): Promise<void> {
    return apiRequest.delete(`/tasks/${taskId}`);
  }

  /**
   * 重新运行任务
   */
  static async retryTask(taskId: string): Promise<Task> {
    return apiRequest.post<Task>(`/tasks/${taskId}/retry`);
  }

  /**
   * 停止运行中的任务
   */
  static async stopTask(taskId: string): Promise<Task> {
    return apiRequest.post<Task>(`/tasks/${taskId}/stop`);
  }

  /**
   * 获取任务的预测结果
   */
  static async getTaskResults(taskId: string): Promise<PredictionResult[]> {
    const task = await this.getTaskDetail(taskId);
    // 转换数据格式以匹配PredictionResult接口
    return task.results?.predictions?.map(pred => ({
      stock_code: pred.stock_code,
      predicted_direction: pred.predicted_direction,
      predicted_return: pred.predicted_return || 0,
      confidence_score: pred.confidence_score,
      confidence_interval: pred.confidence_interval || { lower: 0, upper: 0 },
      risk_assessment: {
        value_at_risk: pred.risk_assessment?.value_at_risk || 0,
        volatility: pred.risk_assessment?.volatility || 0,
        max_drawdown: pred.risk_assessment?.max_drawdown || 0,
        sharpe_ratio: pred.risk_assessment?.sharpe_ratio || 0,
      },
    })) || [];
  }

  /**
   * 获取预测任务的历史预测序列
   */
  static async getPredictionSeries(
    taskId: string,
    stockCode: string,
    lookbackDays?: number
  ): Promise<{
    stock_code: string;
    series: Array<{ date: string; actual: number; predicted: number }>;
  }> {
    const params: Record<string, string | number> = { stock_code: stockCode };
    if (lookbackDays !== undefined && lookbackDays > 0) {
      params.lookback_days = lookbackDays;
    }
    return apiRequest.get(`/tasks/${taskId}/prediction-series`, params);
  }

  /**
   * 导出任务结果
   */
  static async exportTaskResults(
    taskId: string,
    format: 'csv' | 'excel' | 'json' = 'csv'
  ): Promise<Blob> {
    const response = await apiRequest.get(
      `/tasks/${taskId}/export`,
      { format },
    );
    return new Blob([response], { 
      type: format === 'json' ? 'application/json' : 'text/csv' 
    });
  }

  /**
   * 批量删除任务
   */
  static async batchDeleteTasks(taskIds: string[]): Promise<void> {
    return apiRequest.post('/tasks/batch-delete', { task_ids: taskIds });
  }

  /**
   * 获取任务统计信息
   */
  static async getTaskStats(): Promise<{
    total: number;
    completed: number;
    running: number;
    failed: number;
    success_rate: number;
  }> {
    return apiRequest.get('/tasks/stats');
  }
}
