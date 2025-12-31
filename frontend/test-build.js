#!/usr/bin/env node

/**
 * 前端构建测试脚本
 * 
 * 验证所有新增的组件和页面是否能正常编译
 */

const { execSync } = require('child_process');
const path = require('path');

console.log('🚀 开始测试前端构建...\n');

try {
  // 切换到前端目录
  process.chdir(path.join(__dirname));
  
  console.log('📦 检查依赖...');
  execSync('npm list --depth=0', { stdio: 'pipe' });
  console.log('✅ 依赖检查完成\n');
  
  console.log('🔍 进行类型检查...');
  execSync('npx tsc --noEmit', { stdio: 'inherit' });
  console.log('✅ 类型检查通过\n');
  
  console.log('🏗️  尝试构建...');
  execSync('npm run build', { stdio: 'inherit' });
  console.log('✅ 构建成功\n');
  
  console.log('🎉 所有测试通过！前端适配完成。');
  
} catch (error) {
  console.error('❌ 测试失败:', error.message);
  process.exit(1);
}