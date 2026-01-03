#!/usr/bin/env node
/**
 * 真实的回测WebSocket端点测试
 * 
 * 测试后端WebSocket端点是否正常工作
 */

const WebSocket = require('ws');

// 测试配置
const WS_URL = 'ws://localhost:8000/api/v1/backtest/ws/test_task_001';
const HTTP_BASE_URL = 'http://localhost:8000/api/v1/backtest';

// 测试HTTP端点
async function testHttpEndpoints() {
  console.log('\n=== 测试HTTP端点 ===\n');
  
  try {
    // 测试统计端点
    console.log('📊 测试WebSocket统计端点...');
    const statsResponse = await fetch(`${HTTP_BASE_URL}/ws/stats`);
    const statsData = await statsResponse.json();
    console.log('✅ 统计端点响应:', JSON.stringify(statsData, null, 2));
    
    // 测试进度端点
    console.log('\n📈 测试进度HTTP端点...');
    const progressResponse = await fetch(`${HTTP_BASE_URL}/progress/test_task_001`);
    const progressData = await progressResponse.json();
    console.log(`${progressResponse.status === 404 ? '✅' : '⚠️'} 进度端点响应 (${progressResponse.status}):`, JSON.stringify(progressData, null, 2));
    
    return true;
  } catch (error) {
    console.error('❌ HTTP端点测试失败:', error.message);
    return false;
  }
}

// 测试WebSocket连接
async function testWebSocketConnection() {
  console.log('\n=== 测试WebSocket连接 ===\n');
  
  return new Promise((resolve) => {
    console.log(`🔌 连接到: ${WS_URL}`);
    
    const ws = new WebSocket(WS_URL);
    let messageCount = 0;
    let testPassed = false;
    
    // 设置超时
    const timeout = setTimeout(() => {
      console.log('⏱️ 测试超时');
      ws.close();
      resolve(testPassed);
    }, 5000);
    
    ws.on('open', () => {
      console.log('✅ WebSocket连接已建立');
      
      // 发送ping消息
      console.log('\n💓 发送ping消息...');
      ws.send(JSON.stringify({ type: 'ping' }));
    });
    
    ws.on('message', (data) => {
      messageCount++;
      const message = JSON.parse(data.toString());
      console.log(`\n📥 收到消息 #${messageCount}:`, message.type);
      console.log('   内容:', JSON.stringify(message, null, 2));
      
      // 如果收到pong，请求进度
      if (message.type === 'pong') {
        console.log('\n📋 请求当前进度...');
        ws.send(JSON.stringify({ type: 'get_current_progress' }));
      }
      
      // 如果收到进度或无进度数据，测试成功
      if (message.type === 'progress_update' || message.type === 'no_progress_data') {
        testPassed = true;
        console.log('\n✅ WebSocket功能测试通过');
        clearTimeout(timeout);
        ws.close();
      }
    });
    
    ws.on('close', (code, reason) => {
      console.log(`\n🔌 WebSocket连接已关闭: code=${code}, reason=${reason || '正常关闭'}`);
      
      if (code === 4004) {
        console.log('✅ 端点正常工作（任务不存在是预期的）');
        testPassed = true;
      }
      
      clearTimeout(timeout);
      resolve(testPassed);
    });
    
    ws.on('error', (error) => {
      console.error('❌ WebSocket错误:', error.message);
      clearTimeout(timeout);
      resolve(false);
    });
  });
}

// 主测试函数
async function main() {
  console.log('🚀 回测WebSocket端点测试');
  console.log('='.repeat(50));
  
  // 测试HTTP端点
  const httpPassed = await testHttpEndpoints();
  
  // 测试WebSocket连接
  const wsPassed = await testWebSocketConnection();
  
  // 总结
  console.log('\n' + '='.repeat(50));
  console.log('📊 测试总结:');
  console.log(`   HTTP端点: ${httpPassed ? '✅ 通过' : '❌ 失败'}`);
  console.log(`   WebSocket连接: ${wsPassed ? '✅ 通过' : '❌ 失败'}`);
  
  if (httpPassed && wsPassed) {
    console.log('\n🎉 所有测试通过！WebSocket端点工作正常。');
    process.exit(0);
  } else {
    console.log('\n💥 部分测试失败，请检查日志。');
    process.exit(1);
  }
}

// 运行测试
main().catch((error) => {
  console.error('❌ 测试执行失败:', error);
  process.exit(1);
});
