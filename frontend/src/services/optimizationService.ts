/**
 * 超参优化服务
 */

// StandardResponse 类型定义
interface StandardResponse<T = any> {
  success: boolean;
  message: string;
  data: T;
  timestamp?: string;
}

export interface ParamSpaceConfig {
  type: 'int' | 'float' | 'categorical';
  low?: number;
  high?: number;
  choices?: any[];
  default?: any;
  enabled?: boolean;
  log?: boolean;
}

export interface ObjectiveConfig {
  objective_metric: string | string[];
  direction: 'maximize' | 'minimize';
  objective_weights?: Record<string, number>;
}

export interface CreateOptimizationTaskRequest {
  task_name: string;
  strategy_name: string;
  stock_codes: string[];
  start_date: string;
  end_date: string;
  param_space: Record<string, ParamSpaceConfig>;
  objective_config: ObjectiveConfig;
  n_trials: number;
  optimization_method: string;
  timeout?: number;
  backtest_config?: Record<string, any>;
}

export interface OptimizationTask {
  task_id: string;
  task_name: string;
  task_type: string;
  status: string;
  progress: number;
  strategy_name: string;
  n_trials: number;
  created_at?: string;
  completed_at?: string;
  error_message?: string;
  best_score?: number;
  best_trial_number?: number;
}

export interface OptimizationStatus {
  task_id: string;
  status: string;
  progress: number;
  n_trials: number;
  completed_trials: number;
  running_trials: number;
  pruned_trials: number;
  failed_trials: number;
  best_score?: number;
  best_trial_number?: number;
  best_params?: Record<string, any>;
}

export interface OptimizationResult {
  success: boolean;
  strategy_name: string;
  best_params?: Record<string, any>;
  best_score?: number;
  best_trial_number?: number;
  objective_metric: string | string[];
  n_trials: number;
  completed_trials: number;
  running_trials: number;
  pruned_trials: number;
  failed_trials: number;
  optimization_history: Array<{
    trial_number: number;
    params: Record<string, any>;
    state: string;
    score?: number;
    objectives?: number[];
    duration_seconds?: number;
    timestamp?: string;
  }>;
  param_importance?: Record<string, number>;
  pareto_front?: Array<{
    trial_number: number;
    params: Record<string, any>;
    objectives: number[];
  }>;
  optimization_metadata?: {
    method: string;
    direction: string;
    duration_seconds?: number;
    start_time?: string;
    end_time?: string;
    data_period?: {
      start_date: string;
      end_date: string;
    };
  };
}

// API 基础 URL - 使用相对路径，通过Next.js代理转发
// 不再需要绝对URL，直接使用相对路径即可
const API_BASE_URL = '';

export class OptimizationService {
  /**
   * 创建超参优化任务
   */
  static async createTask(
    request: CreateOptimizationTaskRequest
  ): Promise<OptimizationTask> {
    try {
      // 构建完整的 API URL（使用相对路径，通过Next.js代理转发）
      const url = `/api/v1/optimization/tasks`;
      console.log('[OptimizationService] 创建任务请求 URL:', url);
      console.log('[OptimizationService] 请求数据:', JSON.stringify(request, null, 2));
      
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      console.log('[OptimizationService] 响应状态:', response.status, response.statusText);
      console.log('[OptimizationService] 响应 URL:', response.url);
      console.log('[OptimizationService] 响应头:', Object.fromEntries(response.headers.entries()));

      // 如果是 404，检查 URL 是否正确
      if (response.status === 404) {
        console.error('[OptimizationService] 404 错误 - 请求的 URL 不存在');
        console.error('[OptimizationService] 实际请求 URL:', url);
        console.error('[OptimizationService] 请求失败，URL:', url);
        console.error('[OptimizationService] 响应 URL:', response.url);
        console.error('[OptimizationService] 响应状态:', response.status, response.statusText);
        
        // 尝试获取错误详情
        try {
          const errorText = await response.text();
          console.error('[OptimizationService] 错误响应内容:', errorText);
        } catch (e) {
          console.error('[OptimizationService] 无法读取错误响应:', e);
        }
        
        throw new Error(`API 端点不存在 (404): ${url}。请检查：1) 后端服务是否正常运行；2) 路由是否正确注册；3) Next.js代理配置是否正确。`);
      }

      if (!response.ok) {
        let errorMessage = '创建优化任务失败';
        let errorData: any = null;
        
        try {
          const contentType = response.headers.get('content-type');
          if (contentType && contentType.includes('application/json')) {
            errorData = await response.json();
            console.error('[OptimizationService] 错误响应数据:', errorData);
            
            if (errorData.detail) {
              // 处理 Pydantic 验证错误
              if (Array.isArray(errorData.detail)) {
                errorMessage = errorData.detail.map((e: any) => {
                  const loc = e.loc ? e.loc.join('.') : '';
                  return `${loc}: ${e.msg || e.message}`;
                }).join(', ');
              } else if (typeof errorData.detail === 'string') {
                errorMessage = errorData.detail;
              } else {
                errorMessage = JSON.stringify(errorData.detail);
              }
            } else if (errorData.message) {
              errorMessage = errorData.message;
            } else if (errorData.error) {
              errorMessage = errorData.error;
            }
          } else {
            const text = await response.text();
            console.error('[OptimizationService] 错误响应文本:', text);
            errorMessage = text || `HTTP ${response.status}: ${response.statusText}`;
          }
        } catch (e) {
          console.error('[OptimizationService] 解析错误响应失败:', e);
          errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }
        
        throw new Error(errorMessage);
      }

      const result: StandardResponse<OptimizationTask> = await response.json();
      console.log('[OptimizationService] 成功响应:', result);
      
      if (!result.success) {
        throw new Error(result.message || '创建优化任务失败');
      }
      
      if (!result.data) {
        throw new Error('服务器返回数据为空');
      }
      
      return result.data;
    } catch (error) {
      console.error('[OptimizationService] 创建优化任务错误:', error);
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('创建优化任务失败: ' + String(error));
    }
  }

  /**
   * 获取优化任务列表
   */
  static async getTasks(
    status?: string,
    limit: number = 20,
    offset: number = 0
  ): Promise<{ tasks: OptimizationTask[]; total: number }> {
    const params = new URLSearchParams({
      limit: limit.toString(),
      offset: offset.toString(),
    });
    if (status) {
      params.append('status', status);
    }

    const response = await fetch(
      `/api/v1/optimization/tasks?${params.toString()}`
    );

    if (!response.ok) {
      throw new Error('获取优化任务列表失败');
    }

    const result: StandardResponse<{ tasks: OptimizationTask[]; total: number }> =
      await response.json();
    return result.data;
  }

  /**
   * 获取优化任务详情
   */
  static async getTask(taskId: string): Promise<OptimizationTask & { result?: OptimizationResult }> {
    const response = await fetch(
      `/api/v1/optimization/tasks/${taskId}`
    );

    if (!response.ok) {
      throw new Error('获取优化任务详情失败');
    }

    const result: StandardResponse<OptimizationTask & { result?: OptimizationResult }> =
      await response.json();
    return result.data;
  }

  /**
   * 获取优化任务状态
   */
  static async getStatus(taskId: string): Promise<OptimizationStatus> {
    const response = await fetch(
      `/api/v1/optimization/tasks/${taskId}/status`
    );

    if (!response.ok) {
      throw new Error('获取优化状态失败');
    }

    const result: StandardResponse<OptimizationStatus> = await response.json();
    return result.data;
  }

  /**
   * 获取参数重要性
   */
  static async getParamImportance(
    taskId: string
  ): Promise<Record<string, number>> {
    const response = await fetch(
      `/api/v1/optimization/tasks/${taskId}/param-importance`
    );

    if (!response.ok) {
      throw new Error('获取参数重要性失败');
    }

    const result: StandardResponse<Record<string, number>> = await response.json();
    return result.data;
  }

  /**
   * 获取帕累托前沿
   */
  static async getParetoFront(
    taskId: string
  ): Promise<
    Array<{
      trial_number: number;
      params: Record<string, any>;
      objectives: number[];
    }>
  > {
    const response = await fetch(
      `/api/v1/optimization/tasks/${taskId}/pareto-front`
    );

    if (!response.ok) {
      throw new Error('获取帕累托前沿失败');
    }

    const result: StandardResponse<
      Array<{
        trial_number: number;
        params: Record<string, any>;
        objectives: number[];
      }>
    > = await response.json();
    return result.data;
  }
}

