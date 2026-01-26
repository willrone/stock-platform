/**
 * 策略配置管理服务
 *
 * 处理策略配置的API调用，包括：
 * - 获取配置列表
 * - 保存配置
 * - 加载配置
 * - 删除配置
 */

import { apiRequest } from './api';

// 策略配置接口
export interface StrategyConfig {
  config_id: string;
  config_name: string;
  strategy_name: string;
  parameters: Record<string, any>;
  description?: string;
  user_id?: string;
  created_at: string;
  updated_at: string;
}

// 创建配置请求
export interface CreateStrategyConfigRequest {
  config_name: string;
  strategy_name: string;
  parameters: Record<string, any>;
  description?: string;
  user_id?: string;
}

// 更新配置请求
export interface UpdateStrategyConfigRequest {
  config_name?: string;
  parameters?: Record<string, any>;
  description?: string;
}

// 配置列表响应
export interface StrategyConfigListResponse {
  configs: StrategyConfig[];
  total_count: number;
}

// 策略配置服务类
export class StrategyConfigService {
  /**
   * 获取策略配置列表
   */
  static async getConfigs(
    strategyName?: string,
    userId?: string
  ): Promise<StrategyConfigListResponse> {
    const params: any = {};
    if (strategyName) {
      params.strategy_name = strategyName;
    }
    if (userId) {
      params.user_id = userId;
    }

    const response = await apiRequest.get<{ configs: StrategyConfig[]; total_count: number }>(
      '/strategy-configs',
      params
    );

    return {
      configs: response.configs || [],
      total_count: response.total_count || 0,
    };
  }

  /**
   * 获取特定配置详情
   */
  static async getConfig(configId: string): Promise<StrategyConfig> {
    return apiRequest.get<StrategyConfig>(`/strategy-configs/${configId}`);
  }

  /**
   * 保存新配置
   */
  static async createConfig(request: CreateStrategyConfigRequest): Promise<StrategyConfig> {
    return apiRequest.post<StrategyConfig>('/strategy-configs', request);
  }

  /**
   * 更新配置
   */
  static async updateConfig(
    configId: string,
    request: UpdateStrategyConfigRequest
  ): Promise<StrategyConfig> {
    return apiRequest.put<StrategyConfig>(`/strategy-configs/${configId}`, request);
  }

  /**
   * 删除配置
   */
  static async deleteConfig(configId: string): Promise<void> {
    return apiRequest.delete(`/strategy-configs/${configId}`);
  }
}
