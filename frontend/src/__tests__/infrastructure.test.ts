/**
 * 前端基础设施测试
 */

describe('前端项目结构测试', () => {
  test('package.json 配置正确', () => {
    const packageJson = require('../../package.json');
    
    expect(packageJson.name).toBe('stock-prediction-frontend');
    expect(packageJson.version).toBe('0.1.0');
    expect(packageJson.dependencies).toHaveProperty('next');
    expect(packageJson.dependencies).toHaveProperty('react');
    expect(packageJson.dependencies).toHaveProperty('@heroui/react');
  });

  test('TypeScript 配置存在', () => {
    const tsConfig = require('../../tsconfig.json');
    
    expect(tsConfig.compilerOptions).toBeDefined();
    expect(tsConfig.compilerOptions.strict).toBe(true);
    expect(tsConfig.compilerOptions.baseUrl).toBe('.');
    expect(tsConfig.compilerOptions.paths).toHaveProperty('@/*');
  });
});

describe('依赖包测试', () => {
  test('核心依赖可以导入', () => {
    // 测试 React 相关
    expect(() => require('react')).not.toThrow();
    expect(() => require('react-dom')).not.toThrow();
    
    // 测试 Next.js
    expect(() => require('next')).not.toThrow();
  });

  test('工具库可以导入', () => {
    expect(() => require('axios')).not.toThrow();
    expect(() => require('dayjs')).not.toThrow();
    expect(() => require('zustand')).not.toThrow();
  });

  test('图表库可以导入', () => {
    expect(() => require('echarts')).not.toThrow();
    expect(() => require('echarts-for-react')).not.toThrow();
  });
});

describe('配置文件测试', () => {
  test('Next.js 配置正确', () => {
    const nextConfig = require('../../next.config.js');
    
    expect(nextConfig.reactStrictMode).toBe(true);
    expect(nextConfig.swcMinify).toBe(true);
    expect(nextConfig.experimental).toHaveProperty('appDir');
  });

  test('API 代理配置存在', () => {
    const nextConfig = require('../../next.config.js');
    
    expect(nextConfig.rewrites).toBeDefined();
    expect(typeof nextConfig.rewrites).toBe('function');
  });

  test('项目文件结构存在', () => {
    const fs = require('fs');
    const path = require('path');
    
    // 检查关键文件存在
    expect(fs.existsSync(path.join(process.cwd(), 'src/app/page.tsx'))).toBe(true);
    expect(fs.existsSync(path.join(process.cwd(), 'src/app/layout.tsx'))).toBe(true);
    expect(fs.existsSync(path.join(process.cwd(), 'src/app/globals.css'))).toBe(true);
  });});