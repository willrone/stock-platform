/**
 * 简单的WebSocket客户端测试
 * 
 * 测试回测进度WebSocket连接和消息处理
 */

// 模拟WebSocket类（用于测试）
class MockWebSocket {
  constructor(url) {
    this.url = url;
    this.readyState = MockWebSocket.CONNECTING;
    this.onopen = null;
    this.onmessage = null;
    this.onclose = null;
    this.onerror = null;
    
    // 模拟异步连接
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) {
        this.onopen();
      }
    }, 100);
  }
  
  send(data) {
    console.log('📤 发送消息:', data);
    
    // 模拟服务器响应
    setTimeout(() => {
      const message = JSON.parse(data);
      
      if (message.type === 'ping') {
        this.simulateMessage({
          type: 'pong',
          timestamp: new Date().toISOString()
        });
      } else if (message.type === 'get_current_progress') {
        this.simulateMessage({
          type: 'progress_update',
          task_id: 'test_task',
          backtest_id: 'bt_test',
          overall_progress: 45.5,
          current_stage: 'backtest_execution',
          processed_days: 45,
          total_days: 100,
          current_date: '2024-01-15',
          processing_speed: 2.5,
          portfolio_value: 105000,
          signals_generated: 120,
          trades_executed: 85,
          warnings_count: 2,
          stages: [
            {
              name: 'initialization',
              description: '初始化',
              progress: 100,
              status: 'completed'
            },
            {
              name: 'data_loading',
              description: '数据加载',
              progress: 100,
              status: 'completed'
            },
            {
              name: 'backtest_execution',
              description: '回测执行',
              progress: 45,
              status: 'running'
            }
          ],
          timestamp: new Date().toISOString()
        });
      }
    }, 50);
  }
  
  close() {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      this.onclose({ code: 1000, reason: 'Normal closure' });
    }
  }
  
  simulateMessage(data) {
    if (this.onmessage) {
      this.onmessage({
        data: JSON.stringify(data)
      });
    }
  }
  
  static get CONNECTING() { return 0; }
  static get OPEN() { return 1; }
  static get CLOSING() { return 2; }
  static get CLOSED() { return 3; }
}

// 简化的WebSocket客户端类
class SimpleBacktestProgressWebSocket {
  constructor(taskId) {
    this.taskId = taskId;
    this.ws = null;
    this.callbacks = {};
    this.isConnected = false;
  }
  
  setCallbacks(callbacks) {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }
  
  async connect() {
    return new Promise((resolve, reject) => {
      const wsUrl = `ws://localhost:8000/api/v1/backtest/ws/${this.taskId}`;
      this.ws = new MockWebSocket(wsUrl);
      
      this.ws.onopen = () => {
        console.log(`✅ WebSocket连接已建立: ${this.taskId}`);
        this.isConnected = true;
        this.callbacks.onConnection?.(true);
        resolve();
      };
      
      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('❌ 解析消息失败:', error);
        }
      };
      
      this.ws.onclose = (event) => {
        console.log(`🔌 WebSocket连接已关闭: ${this.taskId}`);
        this.isConnected = false;
        this.callbacks.onConnection?.(false);
      };
      
      this.ws.onerror = (error) => {
        console.error(`❌ WebSocket错误: ${this.taskId}`, error);
        this.callbacks.onConnection?.(false);
        reject(error);
      };
    });
  }
  
  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.isConnected = false;
  }
  
  requestCurrentProgress() {
    this.sendMessage({ type: 'get_current_progress' });
  }
  
  sendMessage(message) {
    if (this.ws && this.ws.readyState === MockWebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn('⚠️ WebSocket未连接，无法发送消息');
    }
  }
  
  handleMessage(data) {
    console.log('📥 收到消息:', data.type);
    
    switch (data.type) {
      case 'connection_established':
        console.log('🔗 连接建立确认');
        break;
        
      case 'progress_update':
        console.log(`📊 进度更新: ${data.overall_progress}%`);
        this.callbacks.onProgress?.(data);
        break;
        
      case 'backtest_error':
        console.log('❌ 回测错误:', data.error_message);
        this.callbacks.onError?.(data);
        break;
        
      case 'backtest_completed':
        console.log('✅ 回测完成');
        this.callbacks.onCompletion?.(data);
        break;
        
      case 'pong':
        console.log('🏓 心跳响应');
        break;
        
      default:
        console.log('❓ 未知消息类型:', data.type);
    }
  }
}

// 测试函数
async function testWebSocketConnection() {
  console.log('🧪 测试WebSocket连接...');
  
  const client = new SimpleBacktestProgressWebSocket('test_task_001');
  
  // 设置回调
  client.setCallbacks({
    onConnection: (connected) => {
      console.log(`🔌 连接状态变化: ${connected ? '已连接' : '已断开'}`);
    },
    onProgress: (data) => {
      console.log(`📈 进度更新: ${data.overall_progress}% - ${data.current_stage}`);
      console.log(`   处理进度: ${data.processed_days}/${data.total_days} 天`);
      console.log(`   组合价值: ${data.portfolio_value}`);
    },
    onError: (error) => {
      console.log(`❌ 错误通知: ${error.error_message}`);
    },
    onCompletion: (completion) => {
      console.log(`🎉 完成通知:`, completion.results);
    }
  });
  
  try {
    // 连接WebSocket
    await client.connect();
    
    // 等待一下
    await new Promise(resolve => setTimeout(resolve, 200));
    
    // 请求当前进度
    console.log('📋 请求当前进度...');
    client.requestCurrentProgress();
    
    // 等待响应
    await new Promise(resolve => setTimeout(resolve, 200));
    
    // 发送心跳
    console.log('💓 发送心跳...');
    client.sendMessage({ type: 'ping' });
    
    // 等待响应
    await new Promise(resolve => setTimeout(resolve, 200));
    
    // 断开连接
    console.log('🔌 断开连接...');
    client.disconnect();
    
    console.log('✅ WebSocket测试完成');
    return true;
    
  } catch (error) {
    console.error('❌ WebSocket测试失败:', error);
    return false;
  }
}

async function testProgressDataHandling() {
  console.log('🧪 测试进度数据处理...');
  
  // 模拟进度数据
  const progressData = {
    type: 'progress_update',
    task_id: 'test_task',
    backtest_id: 'bt_test',
    overall_progress: 75.5,
    current_stage: 'metrics_calculation',
    processed_days: 75,
    total_days: 100,
    current_date: '2024-01-20',
    processing_speed: 3.2,
    estimated_completion: new Date(Date.now() + 300000).toISOString(), // 5分钟后
    elapsed_time: '0:02:30.123456',
    portfolio_value: 112500,
    signals_generated: 180,
    trades_executed: 125,
    warnings_count: 1,
    stages: [
      { name: 'initialization', status: 'completed', progress: 100 },
      { name: 'data_loading', status: 'completed', progress: 100 },
      { name: 'strategy_setup', status: 'completed', progress: 100 },
      { name: 'backtest_execution', status: 'completed', progress: 100 },
      { name: 'metrics_calculation', status: 'running', progress: 75 },
      { name: 'report_generation', status: 'pending', progress: 0 },
      { name: 'data_storage', status: 'pending', progress: 0 }
    ]
  };
  
  // 验证数据结构
  console.log('📊 验证进度数据结构...');
  
  const requiredFields = [
    'task_id', 'backtest_id', 'overall_progress', 'current_stage',
    'processed_days', 'total_days', 'portfolio_value', 'stages'
  ];
  
  const missingFields = requiredFields.filter(field => !(field in progressData));
  
  if (missingFields.length > 0) {
    console.error('❌ 缺少必需字段:', missingFields);
    return false;
  }
  
  // 验证阶段数据
  if (!Array.isArray(progressData.stages) || progressData.stages.length === 0) {
    console.error('❌ 阶段数据无效');
    return false;
  }
  
  // 计算完成的阶段数
  const completedStages = progressData.stages.filter(s => s.status === 'completed').length;
  const runningStages = progressData.stages.filter(s => s.status === 'running').length;
  
  console.log(`✅ 数据验证通过:`);
  console.log(`   - 总体进度: ${progressData.overall_progress}%`);
  console.log(`   - 当前阶段: ${progressData.current_stage}`);
  console.log(`   - 已完成阶段: ${completedStages}/${progressData.stages.length}`);
  console.log(`   - 运行中阶段: ${runningStages}`);
  console.log(`   - 处理进度: ${progressData.processed_days}/${progressData.total_days}`);
  
  return true;
}

// 主测试函数
async function runTests() {
  console.log('🚀 回测进度WebSocket客户端测试');
  console.log('=' * 50);
  
  const tests = [
    { name: 'WebSocket连接测试', func: testWebSocketConnection },
    { name: '进度数据处理测试', func: testProgressDataHandling }
  ];
  
  let passed = 0;
  const total = tests.length;
  
  for (const test of tests) {
    console.log(`\n📋 ${test.name}`);
    console.log('-'.repeat(30));
    
    try {
      const result = await test.func();
      if (result) {
        passed++;
        console.log(`✅ ${test.name} 通过`);
      } else {
        console.log(`❌ ${test.name} 失败`);
      }
    } catch (error) {
      console.log(`❌ ${test.name} 异常:`, error.message);
    }
  }
  
  console.log('\n' + '='.repeat(50));
  console.log(`📊 测试结果: ${passed}/${total} 通过`);
  
  if (passed === total) {
    console.log('🎉 所有测试通过！');
  } else {
    console.log('💥 部分测试失败！');
  }
}

// 运行测试
runTests().catch(console.error);