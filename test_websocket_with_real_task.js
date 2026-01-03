#!/usr/bin/env node
/**
 * 使用真实任务测试WebSocket
 */

const WebSocket = require('ws');

const TASK_ID = 'b998692c-03f0-4169-8f0e-6872b73321ef';
const WS_URL = `ws://localhost:8000/api/v1/backtest/ws/${TASK_ID}`;

console.log('🚀 测试回测WebSocket连接');
console.log(`📋 任务ID: ${TASK_ID}`);
console.log(`🔌 连接到: ${WS_URL}\n`);

const ws = new WebSocket(WS_URL);
let messageCount = 0;

ws.on('open', () => {
  console.log('✅ WebSocket连接已建立\n');
  
  // 发送ping
  setTimeout(() => {
    console.log('💓 发送ping消息...');
    ws.send(JSON.stringify({ type: 'ping' }));
  }, 500);
});

ws.on('message', (data) => {
  messageCount++;
  const message = JSON.parse(data.toString());
  
  console.log(`\n📥 收到消息 #${messageCount}: ${message.type}`);
  console.log('─'.repeat(50));
  console.log(JSON.stringify(message, null, 2));
  console.log('─'.repeat(50));
  
  // 如果收到pong，请求进度
  if (message.type === 'pong') {
    setTimeout(() => {
      console.log('\n📋 请求当前进度...');
      ws.send(JSON.stringify({ type: 'get_current_progress' }));
    }, 500);
  }
  
  // 如果收到进度更新，等待一会儿后关闭
  if (message.type === 'progress_update' || message.type === 'no_progress_data') {
    setTimeout(() => {
      console.log('\n✅ 测试完成，关闭连接...');
      ws.close();
    }, 1000);
  }
});

ws.on('close', (code, reason) => {
  console.log(`\n🔌 WebSocket连接已关闭`);
  console.log(`   Code: ${code}`);
  console.log(`   Reason: ${reason || '正常关闭'}`);
  console.log(`   收到消息数: ${messageCount}`);
  
  if (messageCount > 0) {
    console.log('\n🎉 WebSocket测试成功！');
    process.exit(0);
  } else {
    console.log('\n❌ 未收到任何消息');
    process.exit(1);
  }
});

ws.on('error', (error) => {
  console.error('\n❌ WebSocket错误:', error.message);
  process.exit(1);
});

// 超时保护
setTimeout(() => {
  console.log('\n⏱️ 测试超时');
  ws.close();
  process.exit(messageCount > 0 ? 0 : 1);
}, 10000);
