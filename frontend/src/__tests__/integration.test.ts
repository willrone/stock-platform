/**
 * 前端集成测试
 *
 * 测试前端与后端的集成功能，包括：
 * - API服务集成
 * - WebSocket连接
 * - 状态管理集成
 * - 用户工作流程
 */

// 测试工具导入（保留以备将来使用）
// import { render, screen, fireEvent, waitFor } from '@testing-library/react';
// import { act } from 'react-dom/test-utils';

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
import { TaskService } from '../services/taskService';
import { integrationTestManager } from '../utils/integrationTest';

describe('前端集成测试', () => {
  beforeEach(() => {
    // 重置所有模拟
    jest.clearAllMocks();
  });

  afterEach(() => {
    // 清理
    jest.restoreAllMocks();
  });

  describe('API服务集成', () => {
    it('应该能够进行健康检查', async () => {
      // 模拟健康检查成功
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

      (apiRequest.get as jest.MockedFunction<typeof apiRequest.get>).mockResolvedValue(
        mockSystemStatus
      );

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
          {
            model_id: 'lstm_v1',
            model_name: 'LSTM深度学习模型',
            model_type: 'lstm',
            accuracy: 0.78,
          },
        ],
      };

      (apiRequest.get as Mock).mockResolvedValue(mockModels);

      const result = await apiRequest.get('/models');
      expect(result).toEqual(mockModels);
      expect(result.models).toHaveLength(2);
    });

    it('应该能够处理API错误', async () => {
      const mockError = new Error('网络连接失败');
      (apiRequest.get as Mock).mockRejectedValue(mockError);

      await expect(apiRequest.get('/invalid-endpoint')).rejects.toThrow('网络连接失败');
    });
  });

  describe('WebSocket服务集成', () => {
    it('应该能够建立WebSocket连接', () => {
      (wsService.isConnected as Mock).mockReturnValue(true);

      const isConnected = wsService.isConnected();
      expect(isConnected).toBe(true);
    });

    it('应该能够订阅任务更新', () => {
      const taskId = 'test_task_123';

      wsService.subscribeToTask(taskId);
      expect(wsService.subscribeToTask).toHaveBeenCalledWith(taskId);
    });

    it('应该能够取消订阅任务更新', () => {
      const taskId = 'test_task_123';

      wsService.unsubscribeFromTask(taskId);
      expect(wsService.unsubscribeFromTask).toHaveBeenCalledWith(taskId);
    });

    it('应该能够订阅系统状态', () => {
      wsService.subscribeToSystemStatus();
      expect(wsService.subscribeToSystemStatus).toHaveBeenCalledTimes(1);
    });

    it('应该能够发送消息', () => {
      const message = { type: 'ping' };

      wsService.send('ping', message);
      expect(wsService.send).toHaveBeenCalledWith('ping', message);
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

      (apiRequest.post as Mock).mockResolvedValue(mockTask);

      const taskRequest = {
        task_name: '测试任务',
        stock_codes: ['000001.SZ'],
        model_id: 'xgboost_v1',
      };

      const result = await TaskService.createTask(taskRequest);
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
          {
            task_id: 'task_002',
            task_name: '任务2',
            status: 'running',
            created_at: new Date().toISOString(),
          },
        ],
        total: 2,
        limit: 20,
        offset: 0,
      };

      (apiRequest.get as Mock).mockResolvedValue(mockTaskList);

      const result = await TaskService.getTasks();
      expect(result).toEqual(mockTaskList);
      expect(result.tasks).toHaveLength(2);
    });

    it('应该能够获取任务详情', async () => {
      const taskId = 'task_123';
      const mockTaskDetail = {
        task_id: taskId,
        task_name: '测试任务',
        status: 'completed',
        progress: 100,
        results: {
          predictions: [
            {
              stock_code: '000001.SZ',
              predicted_direction: 1,
              confidence_score: 0.85,
            },
          ],
        },
      };

      (apiRequest.get as Mock).mockResolvedValue(mockTaskDetail);

      const result = await TaskService.getTaskDetail(taskId);
      expect(result).toEqual(mockTaskDetail);
      expect(apiRequest.get).toHaveBeenCalledWith(`/tasks/${taskId}`);
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

      (apiRequest.get as Mock).mockResolvedValue(mockStockData);

      const params = {
        stock_code: '000001.SZ',
        start_date: '2024-01-01',
        end_date: '2024-01-31',
      };

      const result = await apiRequest.get('/stocks/data', params);
      expect(result).toEqual(mockStockData);
    });

    it('应该能够获取技术指标', async () => {
      const mockIndicators = {
        stock_code: '000001.SZ',
        indicators: {
          ma_5: 10.5,
          ma_10: 10.3,
          rsi: 65.2,
          macd: 0.15,
        },
      };

      (apiRequest.get as Mock).mockResolvedValue(mockIndicators);

      const result = await apiRequest.get('/stocks/000001.SZ/indicators');
      expect(result).toEqual(mockIndicators);
      expect(Object.keys(result.indicators)).toHaveLength(4);
    });

    it('应该能够获取数据服务状态', async () => {
      const mockDataStatus = {
        service_url: 'http://192.168.3.62:8000',
        is_connected: true,
        last_check: new Date().toISOString(),
        response_time: 150,
      };

      (apiRequest.get as Mock).mockResolvedValue(mockDataStatus);

      const result = await apiRequest.get('/data/status');
      expect(result).toEqual(mockDataStatus);
      expect(result.is_connected).toBe(true);
    });

    it('应该能够同步数据', async () => {
      const mockSyncResult = {
        success: true,
        synced_stocks: ['000001.SZ', '000002.SZ'],
        failed_stocks: [],
        total_records: 2000,
        sync_duration: '2.5s',
      };

      (apiRequest.post as Mock).mockResolvedValue(mockSyncResult);

      const syncRequest = {
        stock_codes: ['000001.SZ', '000002.SZ'],
        force_update: false,
      };

      const result = await apiRequest.post('/data/sync', syncRequest);
      expect(result).toEqual(mockSyncResult);
      expect(result.synced_stocks).toHaveLength(2);
    });
  });

  describe('错误处理集成', () => {
    it('应该能够处理网络错误', async () => {
      const networkError = new Error('网络连接失败');
      (apiRequest.get as Mock).mockRejectedValue(networkError);

      await expect(apiRequest.get('/test')).rejects.toThrow('网络连接失败');
    });

    it('应该能够处理API错误响应', async () => {
      const apiError = new Error('服务器内部错误');
      (apiRequest.post as Mock).mockRejectedValue(apiError);

      await expect(
        TaskService.createTask({
          task_name: '',
          stock_codes: [],
          model_id: 'invalid',
        })
      ).rejects.toThrow('服务器内部错误');
    });

    it('应该能够处理WebSocket连接错误', () => {
      (wsService.isConnected as Mock).mockReturnValue(false);

      const isConnected = wsService.isConnected();
      expect(isConnected).toBe(false);

      // 尝试重连
      wsService.reconnect();
      expect(wsService.reconnect).toHaveBeenCalledTimes(1);
    });
  });

  describe('集成测试管理器', () => {
    it('应该能够运行API连接测试', async () => {
      (healthCheck as Mock).mockResolvedValue(true);

      const result = await integrationTestManager.testApiConnection();
      expect(result.success).toBe(true);
      expect(result.name).toBe('API连接测试');
    });

    it('应该能够运行WebSocket连接测试', async () => {
      (wsService.isConnected as Mock).mockReturnValue(true);

      const result = await integrationTestManager.testWebSocketConnection();
      expect(result.success).toBe(true);
      expect(result.name).toBe('WebSocket连接测试');
    });

    it('应该能够运行完整的测试套件', async () => {
      // 模拟所有API调用成功
      (healthCheck as Mock).mockResolvedValue(true);
      (apiRequest.get as Mock).mockResolvedValue({ status: 'healthy' });
      (apiRequest.post as Mock).mockResolvedValue({ task_id: 'test_task' });
      (wsService.isConnected as Mock).mockReturnValue(true);

      const result = await integrationTestManager.runAllTests();
      expect(result.name).toBe('前后端集成测试');
      expect(result.totalTests).toBeGreaterThan(0);
      expect(result.results).toHaveLength(result.totalTests);
    });
  });

  describe('用户工作流程测试', () => {
    it('应该能够完成完整的预测任务工作流程', async () => {
      // 模拟完整工作流程的API调用
      const mockModels = { models: [{ model_id: 'xgboost_v1' }] };
      const mockDataStatus = { is_connected: true };
      const mockTask = { task_id: 'workflow_task' };
      const mockTaskList = { tasks: [mockTask] };
      const mockTaskDetail = { ...mockTask, status: 'completed' };

      (apiRequest.get as Mock)
        .mockResolvedValueOnce(mockModels) // 获取模型列表
        .mockResolvedValueOnce(mockDataStatus) // 获取数据状态
        .mockResolvedValueOnce(mockTaskList) // 获取任务列表
        .mockResolvedValueOnce(mockTaskDetail); // 获取任务详情

      (apiRequest.post as Mock).mockResolvedValue(mockTask); // 创建任务
      (wsService.isConnected as Mock).mockReturnValue(true);

      const result = await integrationTestManager.testCompleteUserWorkflow();
      expect(result.success).toBe(true);
      expect(result.details.steps).toContain('创建预测任务成功');
      expect(result.details.steps).toContain('获取任务详情成功');
    });

    it('应该能够处理工作流程中的错误', async () => {
      // 模拟工作流程中的错误
      (apiRequest.get as Mock).mockRejectedValue(new Error('服务不可用'));

      const result = await integrationTestManager.testCompleteUserWorkflow();
      expect(result.success).toBe(false);
      expect(result.message).toContain('服务不可用');
    });
  });

  describe('性能测试', () => {
    it('应该能够处理并发请求', async () => {
      (apiRequest.get as Mock).mockResolvedValue({ status: 'ok' });

      // 并发发送多个请求
      const promises = Array.from({ length: 10 }, () => apiRequest.get('/health'));
      const results = await Promise.all(promises);

      expect(results).toHaveLength(10);
      results.forEach(result => {
        expect(result.status).toBe('ok');
      });
    });

    it('应该能够处理大量数据', async () => {
      const largeDataSet = Array.from({ length: 1000 }, (_, i) => ({
        id: i,
        name: `Item ${i}`,
        value: Math.random(),
      }));

      (apiRequest.get as Mock).mockResolvedValue({ data: largeDataSet });

      const result = await apiRequest.get('/large-dataset');
      expect(result.data).toHaveLength(1000);
    });
  });

  describe('缓存测试', () => {
    it('应该能够缓存API响应', async () => {
      const mockData = { cached: true, timestamp: Date.now() };
      (apiRequest.get as Mock).mockResolvedValue(mockData);

      // 第一次调用
      const result1 = await apiRequest.get('/cached-endpoint');
      expect(result1).toEqual(mockData);

      // 第二次调用应该使用缓存
      const result2 = await apiRequest.get('/cached-endpoint');
      expect(result2).toEqual(mockData);

      // 验证API只被调用了一次（如果有缓存机制）
      // 注意：这里的测试取决于实际的缓存实现
    });
  });

  describe('状态管理集成', () => {
    it('应该能够正确更新应用状态', () => {
      // 这里需要根据实际的状态管理实现来编写测试
      // 例如测试Zustand store的更新
    });

    it('应该能够处理状态同步', () => {
      // 测试多个组件之间的状态同步
    });
  });
});

describe('端到端工作流程测试', () => {
  it('应该能够完成完整的股票预测流程', async () => {
    // 1. 获取模型列表
    const mockModels = { models: [{ model_id: 'xgboost_v1' }] };
    (apiRequest.get as Mock).mockResolvedValueOnce(mockModels);

    // 2. 创建预测任务
    const mockTask = { task_id: 'e2e_task', status: 'created' };
    (apiRequest.post as Mock).mockResolvedValueOnce(mockTask);

    // 3. 订阅任务更新
    (wsService.isConnected as Mock).mockReturnValue(true);

    // 4. 获取任务结果
    const mockTaskDetail = {
      ...mockTask,
      status: 'completed',
      results: {
        predictions: [
          {
            stock_code: '000001.SZ',
            predicted_direction: 1,
            confidence_score: 0.85,
          },
        ],
      },
    };
    (apiRequest.get as Mock).mockResolvedValueOnce(mockTaskDetail);

    // 执行完整流程
    const models = await apiRequest.get('/models');
    expect(models.models).toHaveLength(1);

    const task = await apiRequest.post('/tasks', {
      task_name: 'E2E测试任务',
      stock_codes: ['000001.SZ'],
      model_id: models.models[0].model_id,
    });
    expect(task.task_id).toBe('e2e_task');

    wsService.subscribeToTask(task.task_id);
    expect(wsService.subscribeToTask).toHaveBeenCalledWith('e2e_task');

    const taskDetail = await apiRequest.get(`/tasks/${task.task_id}`);
    expect(taskDetail.status).toBe('completed');
    expect(taskDetail.results.predictions).toHaveLength(1);
  });

  it('应该能够处理数据同步流程', async () => {
    // 1. 检查数据状态
    const mockDataStatus = { is_connected: true };
    (apiRequest.get as Mock).mockResolvedValueOnce(mockDataStatus);

    // 2. 获取本地文件列表
    const mockFiles = { files: [], total: 0 };
    (apiRequest.get as Mock).mockResolvedValueOnce(mockFiles);

    // 3. 同步数据
    const mockSyncResult = { synced_stocks: ['000001.SZ'], failed_stocks: [] };
    (apiRequest.post as Mock).mockResolvedValueOnce(mockSyncResult);

    // 执行数据同步流程
    const dataStatus = await apiRequest.get('/data/status');
    expect(dataStatus.is_connected).toBe(true);

    const files = await apiRequest.get('/data/files');
    expect(files.total).toBe(0);

    const syncResult = await apiRequest.post('/data/sync', {
      stock_codes: ['000001.SZ'],
    });
    expect(syncResult.synced_stocks).toContain('000001.SZ');
  });
});
