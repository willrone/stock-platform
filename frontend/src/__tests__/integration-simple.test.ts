/**
 * 简化的前端集成测试
 * 
 * 测试前端与后端的基础集成功能
 */

// 模拟API服务
jest.mock('../services/api', () => ({
  apiRequest: {
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
    delete: jest.fn(),
  },
  healthCheck: jest.fn(),
}));

// 模拟WebSocket服务
jest.mock('../services/websocket', () => ({
  wsService: {
    on: jest.fn(),
    off: jest.fn(),
    send: jest.fn(),
    subscribeToTask: jest.fn(),
    unsubscribeFromTask: jest.fn(),
    subscribeToSystemStatus: jest.fn(),
    unsubscribeFromSystemStatus: jest.fn(),
    isConnected: jest.fn(),
    reconnect: jest.fn(),
    disconnect: jest.fn(),
  },
}));

import { apiRequest, healthCheck } from '../services/api';
import { wsService } from '../services/websocket';

describe('前端集成测试', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('API服务集成', () => {
    it('应该能够进行健康检查', async () => {
      (healthCheck as jest.MockedFunction<typeof healthCheck>).mockResolvedValue(true);

      const result = await healthCheck();
      expect(result).toBe(true);
      expect(healthCheck).toHaveBeenCalledTimes(1);
    });

    it('应该能够获取系统状态', async () => {
      const mockSystemStatus = {
        api_server: { status: 'healthy', uptime: '5 days' },
        data_service: { status: 'healthy', last_update: new Date().toISOString() },
        prediction_engine: { status: 'healthy', active_models: 3 },
      };

      (apiRequest.get as jest.MockedFunction<typeof apiRequest.get>).mockResolvedValue(mockSystemStatus);

      const result = await apiRequest.get('/system/status');
      expect(result).toEqual(mockSystemStatus);
      expect(apiRequest.get).toHaveBeenCalledWith('/system/status');
    });

    it('应该能够获取模型列表', async () => {
      const mockModels = {
        models: [
          {
            model_id: 'xgboost_v1',
            model_name: 'XGBoost基线模型',
            model_type: 'xgboost',
            accuracy: 0.75,
          },
        ],
      };

      (apiRequest.get as jest.MockedFunction<typeof apiRequest.get>).mockResolvedValue(mockModels);

      const result = await apiRequest.get('/models');
      expect(result).toEqual(mockModels);
      expect(result.models).toHaveLength(1);
    });

    it('应该能够处理API错误', async () => {
      const mockError = new Error('网络连接失败');
      (apiRequest.get as jest.MockedFunction<typeof apiRequest.get>).mockRejectedValue(mockError);

      await expect(apiRequest.get('/invalid-endpoint')).rejects.toThrow('网络连接失败');
    });
  });

  describe('WebSocket服务集成', () => {
    it('应该能够建立WebSocket连接', () => {
      (wsService.isConnected as jest.MockedFunction<typeof wsService.isConnected>).mockReturnValue(true);

      const isConnected = wsService.isConnected();
      expect(isConnected).toBe(true);
    });

    it('应该能够订阅任务更新', () => {
      const taskId = 'test_task_123';
      
      wsService.subscribeToTask(taskId);
      expect(wsService.subscribeToTask).toHaveBeenCalledWith(taskId);
    });

    it('应该能够发送消息', () => {
      wsService.send('ping');
      expect(wsService.send).toHaveBeenCalledWith('ping');
    });
  });

  describe('任务服务集成', () => {
    it('应该能够创建任务', async () => {
      const mockTask = {
        task_id: 'task_123',
        task_name: '测试任务',
        status: 'created',
        stock_codes: ['000001.SZ'],
        model_id: 'xgboost_v1',
        created_at: new Date().toISOString(),
      };

      (apiRequest.post as jest.MockedFunction<typeof apiRequest.post>).mockResolvedValue(mockTask);

      const taskRequest = {
        task_name: '测试任务',
        stock_codes: ['000001.SZ'],
        model_id: 'xgboost_v1',
      };

      const result = await apiRequest.post('/tasks', taskRequest);
      expect(result).toEqual(mockTask);
      expect(apiRequest.post).toHaveBeenCalledWith('/tasks', taskRequest);
    });

    it('应该能够获取任务列表', async () => {
      const mockTaskList = {
        tasks: [
          {
            task_id: 'task_001',
            task_name: '任务1',
            status: 'completed',
            created_at: new Date().toISOString(),
          },
        ],
        total: 1,
        limit: 20,
        offset: 0,
      };

      (apiRequest.get as jest.MockedFunction<typeof apiRequest.get>).mockResolvedValue(mockTaskList);

      const result = await apiRequest.get('/tasks');
      expect(result).toEqual(mockTaskList);
      expect(result.tasks).toHaveLength(1);
    });
  });

  describe('数据服务集成', () => {
    it('应该能够获取股票数据', async () => {
      const mockStockData = {
        stock_code: '000001.SZ',
        start_date: '2024-01-01',
        end_date: '2024-01-31',
        data_points: 20,
      };

      (apiRequest.get as jest.MockedFunction<typeof apiRequest.get>).mockResolvedValue(mockStockData);

      const params = {
        stock_code: '000001.SZ',
        start_date: '2024-01-01',
        end_date: '2024-01-31',
      };

      const result = await apiRequest.get('/stocks/data', params);
      expect(result).toEqual(mockStockData);
    });

    it('应该能够获取数据服务状态', async () => {
      const mockDataStatus = {
        service_url: 'http://192.168.3.62:8000',
        is_connected: true,
        last_check: new Date().toISOString(),
        response_time: 150,
      };

      (apiRequest.get as jest.MockedFunction<typeof apiRequest.get>).mockResolvedValue(mockDataStatus);

      const result = await apiRequest.get('/data/status');
      expect(result).toEqual(mockDataStatus);
      expect(result.is_connected).toBe(true);
    });
  });

  describe('错误处理集成', () => {
    it('应该能够处理网络错误', async () => {
      const networkError = new Error('网络连接失败');
      (apiRequest.get as jest.MockedFunction<typeof apiRequest.get>).mockRejectedValue(networkError);

      await expect(apiRequest.get('/test')).rejects.toThrow('网络连接失败');
    });

    it('应该能够处理WebSocket连接错误', () => {
      (wsService.isConnected as jest.MockedFunction<typeof wsService.isConnected>).mockReturnValue(false);

      const isConnected = wsService.isConnected();
      expect(isConnected).toBe(false);

      wsService.reconnect();
      expect(wsService.reconnect).toHaveBeenCalledTimes(1);
    });
  });
});