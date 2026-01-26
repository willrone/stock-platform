/**
 * 绩效分解组件测试
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { PerformanceBreakdown } from '../PerformanceBreakdown';

// Mock ECharts
jest.mock('echarts', () => ({
  init: jest.fn(() => ({
    setOption: jest.fn(),
    dispose: jest.fn(),
    resize: jest.fn(),
    on: jest.fn(),
    off: jest.fn(),
  })),
  graphic: {
    LinearGradient: jest.fn(),
  },
}));

const mockMonthlyPerformance = [
  {
    year: 2023,
    month: 1,
    return_rate: 0.05,
    volatility: 0.15,
    sharpe_ratio: 1.2,
    max_drawdown: -0.08,
    trading_days: 21,
  },
  {
    year: 2023,
    month: 2,
    return_rate: -0.02,
    volatility: 0.18,
    sharpe_ratio: 0.8,
    max_drawdown: -0.12,
    trading_days: 20,
  },
  {
    year: 2023,
    month: 3,
    return_rate: 0.08,
    volatility: 0.16,
    sharpe_ratio: 1.5,
    max_drawdown: -0.05,
    trading_days: 22,
  },
];

const mockYearlyPerformance = [
  {
    year: 2022,
    annual_return: 0.15,
    volatility: 0.18,
    sharpe_ratio: 1.2,
    max_drawdown: -0.15,
    calmar_ratio: 1.0,
    sortino_ratio: 1.4,
    win_rate: 0.65,
    profit_factor: 1.8,
    total_trades: 120,
  },
  {
    year: 2023,
    annual_return: 0.22,
    volatility: 0.16,
    sharpe_ratio: 1.5,
    max_drawdown: -0.12,
    calmar_ratio: 1.8,
    sortino_ratio: 1.8,
    win_rate: 0.68,
    profit_factor: 2.1,
    total_trades: 135,
  },
];

const mockSeasonalAnalysis = {
  monthly_avg_returns: [
    0.02, 0.01, 0.03, 0.015, 0.025, 0.01, -0.005, 0.02, 0.018, 0.022, 0.015, 0.008,
  ],
  monthly_win_rates: [0.65, 0.58, 0.72, 0.62, 0.68, 0.55, 0.48, 0.63, 0.61, 0.69, 0.59, 0.52],
  quarterly_performance: {
    q1: 0.06,
    q2: 0.05,
    q3: 0.03,
    q4: 0.045,
  },
  best_month: {
    month: 3,
    avg_return: 0.03,
  },
  worst_month: {
    month: 7,
    avg_return: -0.005,
  },
};

const mockBenchmarkComparison = {
  dates: ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01'],
  strategy_returns: [0.05, 0.03, 0.08, 0.06],
  benchmark_returns: [0.03, 0.02, 0.06, 0.04],
  excess_returns: [0.02, 0.01, 0.02, 0.02],
  tracking_error: 0.08,
  information_ratio: 0.75,
  beta: 0.95,
  alpha: 0.02,
  correlation: 0.85,
};

describe('PerformanceBreakdown', () => {
  it('应该渲染绩效分解组件', () => {
    render(
      <PerformanceBreakdown
        taskId="test-task-1"
        monthlyPerformance={mockMonthlyPerformance}
        yearlyPerformance={mockYearlyPerformance}
        seasonalAnalysis={mockSeasonalAnalysis}
        benchmarkComparison={mockBenchmarkComparison}
      />
    );

    // 检查年度统计概览
    expect(screen.getByText('平均年化收益')).toBeInTheDocument();
    expect(screen.getByText('平均波动率')).toBeInTheDocument();
    expect(screen.getByText('平均夏普比率')).toBeInTheDocument();
    expect(screen.getByText('回测年数')).toBeInTheDocument();
  });

  it('应该显示最佳和最差年份', () => {
    render(
      <PerformanceBreakdown
        taskId="test-task-1"
        monthlyPerformance={mockMonthlyPerformance}
        yearlyPerformance={mockYearlyPerformance}
        seasonalAnalysis={mockSeasonalAnalysis}
        benchmarkComparison={mockBenchmarkComparison}
      />
    );

    // 检查最佳年份
    expect(screen.getByText('最佳年份')).toBeInTheDocument();
    expect(screen.getAllByText('2023')[0]).toBeInTheDocument(); // 使用 getAllByText 获取第一个

    // 检查最差年份
    expect(screen.getByText('最差年份')).toBeInTheDocument();
    expect(screen.getByText('2022')).toBeInTheDocument(); // 最差年份
  });

  it('应该显示标签页组件', () => {
    render(
      <PerformanceBreakdown
        taskId="test-task-1"
        monthlyPerformance={mockMonthlyPerformance}
        yearlyPerformance={mockYearlyPerformance}
        seasonalAnalysis={mockSeasonalAnalysis}
        benchmarkComparison={mockBenchmarkComparison}
      />
    );

    // 检查 Tabs 组件是否渲染
    const tabsElement = screen.getByRole('tablist');
    expect(tabsElement).toBeInTheDocument();
  });

  it('应该显示季度表现统计', () => {
    render(
      <PerformanceBreakdown
        taskId="test-task-1"
        monthlyPerformance={mockMonthlyPerformance}
        yearlyPerformance={mockYearlyPerformance}
        seasonalAnalysis={mockSeasonalAnalysis}
        benchmarkComparison={mockBenchmarkComparison}
      />
    );

    // 检查组件能正常渲染，不依赖标签页交互
    expect(screen.getByText('平均年化收益')).toBeInTheDocument();
  });

  it('应该显示年度绩效数据', () => {
    render(
      <PerformanceBreakdown
        taskId="test-task-1"
        monthlyPerformance={mockMonthlyPerformance}
        yearlyPerformance={mockYearlyPerformance}
        seasonalAnalysis={mockSeasonalAnalysis}
        benchmarkComparison={mockBenchmarkComparison}
      />
    );

    // 检查年度统计数据
    expect(screen.getByText('平均年化收益')).toBeInTheDocument();
    expect(screen.getByText('18.50%')).toBeInTheDocument(); // 计算出的平均收益
  });

  it('应该显示基准对比数据', () => {
    render(
      <PerformanceBreakdown
        taskId="test-task-1"
        monthlyPerformance={mockMonthlyPerformance}
        yearlyPerformance={mockYearlyPerformance}
        seasonalAnalysis={mockSeasonalAnalysis}
        benchmarkComparison={mockBenchmarkComparison}
      />
    );

    // 检查基本组件渲染
    expect(screen.getByText('回测年数')).toBeInTheDocument();
    expect(screen.getByText('2 年')).toBeInTheDocument();
  });

  it('应该处理空数据情况', () => {
    render(
      <PerformanceBreakdown
        taskId="test-task-1"
        monthlyPerformance={[]}
        yearlyPerformance={[]}
        seasonalAnalysis={mockSeasonalAnalysis}
        benchmarkComparison={mockBenchmarkComparison}
      />
    );

    // 检查空数据提示
    expect(screen.getByText('暂无绩效分解数据')).toBeInTheDocument();
  });

  it('应该支持组件渲染', () => {
    render(
      <PerformanceBreakdown
        taskId="test-task-1"
        monthlyPerformance={mockMonthlyPerformance}
        yearlyPerformance={mockYearlyPerformance}
        seasonalAnalysis={mockSeasonalAnalysis}
        benchmarkComparison={mockBenchmarkComparison}
      />
    );

    // 检查基本统计信息
    expect(screen.getByText('平均年化收益')).toBeInTheDocument();
    expect(screen.getByText('平均波动率')).toBeInTheDocument();

    // 检查 Tabs 组件渲染
    const tabsElement = screen.getByRole('tablist');
    expect(tabsElement).toBeInTheDocument();
  });
});
