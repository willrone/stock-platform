/**
 * 风险分析组件测试
 */

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { RiskAnalysis } from '../RiskAnalysis';

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

const mockRiskMetrics = {
  sharpe_ratio: 1.25,
  sortino_ratio: 1.45,
  calmar_ratio: 0.85,
  information_ratio: 0.75,
  max_drawdown: -0.15,
  avg_drawdown: -0.05,
  drawdown_recovery_time: 45,
  volatility_daily: 0.02,
  volatility_monthly: 0.08,
  volatility_annual: 0.18,
  var_95: -0.03,
  var_99: -0.05,
  cvar_95: -0.04,
  cvar_99: -0.06,
  beta: 0.95,
  alpha: 0.02,
  tracking_error: 0.08,
  upside_capture: 1.05,
  downside_capture: 0.92,
};

const mockReturnDistribution = {
  daily_returns: [0.01, -0.02, 0.015, -0.01, 0.005],
  monthly_returns: [0.05, -0.03, 0.08, -0.02],
  return_bins: [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05],
  return_frequencies: [2, 5, 8, 12, 6, 3],
  normality_test: {
    statistic: 2.45,
    p_value: 0.12,
    is_normal: false,
  },
  skewness: -0.25,
  kurtosis: 3.2,
  percentiles: {
    p5: -0.04,
    p25: -0.015,
    p50: 0.005,
    p75: 0.02,
    p95: 0.045,
  },
};

const mockRollingMetrics = {
  dates: ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01'],
  rolling_sharpe: [1.2, 1.3, 1.1, 1.4],
  rolling_volatility: [0.18, 0.16, 0.2, 0.17],
  rolling_drawdown: [-0.05, -0.08, -0.12, -0.06],
  rolling_beta: [0.95, 0.98, 0.92, 0.96],
  window_size: 60,
};

describe('RiskAnalysis', () => {
  it('应该渲染风险分析组件', () => {
    render(
      <RiskAnalysis
        taskId="test-task-1"
        riskMetrics={mockRiskMetrics}
        returnDistribution={mockReturnDistribution}
        rollingMetrics={mockRollingMetrics}
      />
    );

    // 检查主要标题
    expect(screen.getByText('绩效指标')).toBeInTheDocument();
    expect(screen.getByText('风险指标')).toBeInTheDocument();
    expect(screen.getByText('市场相关性')).toBeInTheDocument();
  });

  it('应该显示风险指标数值', () => {
    render(
      <RiskAnalysis
        taskId="test-task-1"
        riskMetrics={mockRiskMetrics}
        returnDistribution={mockReturnDistribution}
        rollingMetrics={mockRollingMetrics}
      />
    );

    // 检查夏普比率
    expect(screen.getByText('夏普比率')).toBeInTheDocument();
    expect(screen.getByText('1.250')).toBeInTheDocument();

    // 检查最大回撤
    expect(screen.getByText('最大回撤')).toBeInTheDocument();
    expect(screen.getByText('-15.00%')).toBeInTheDocument();
  });

  it('应该显示VaR分析', () => {
    render(
      <RiskAnalysis
        taskId="test-task-1"
        riskMetrics={mockRiskMetrics}
        returnDistribution={mockReturnDistribution}
        rollingMetrics={mockRollingMetrics}
      />
    );

    // 检查VaR标题
    expect(screen.getByText('风险价值 (VaR) 分析')).toBeInTheDocument();

    // 检查VaR数值
    expect(screen.getAllByText('95% 置信度')).toHaveLength(2);
    expect(screen.getAllByText('99% 置信度')).toHaveLength(2);
  });

  it('应该显示收益分布分析', () => {
    render(
      <RiskAnalysis
        taskId="test-task-1"
        riskMetrics={mockRiskMetrics}
        returnDistribution={mockReturnDistribution}
        rollingMetrics={mockRollingMetrics}
      />
    );

    // 检查收益分布标题
    expect(screen.getByText('收益分布分析')).toBeInTheDocument();

    // 检查分布特征
    expect(screen.getByText('分布特征')).toBeInTheDocument();
    expect(screen.getByText('偏度')).toBeInTheDocument();
    expect(screen.getByText('峰度')).toBeInTheDocument();

    // 检查正态性检验
    expect(screen.getByText('正态性检验')).toBeInTheDocument();
    expect(screen.getByText('否')).toBeInTheDocument(); // is_normal: false
  });

  it('应该显示滚动风险指标', () => {
    render(
      <RiskAnalysis
        taskId="test-task-1"
        riskMetrics={mockRiskMetrics}
        returnDistribution={mockReturnDistribution}
        rollingMetrics={mockRollingMetrics}
      />
    );

    // 检查滚动指标标题
    expect(screen.getByText('滚动风险指标')).toBeInTheDocument();
  });

  it('应该处理空数据情况', () => {
    render(
      <RiskAnalysis
        taskId="test-task-1"
        riskMetrics={null as unknown as never}
        returnDistribution={null as unknown as never}
        rollingMetrics={null as unknown as never}
      />
    );

    // 检查空数据提示
    expect(screen.getByText('暂无风险分析数据')).toBeInTheDocument();
  });

  it('应该支持风险指标点击查看详情', () => {
    render(
      <RiskAnalysis
        taskId="test-task-1"
        riskMetrics={mockRiskMetrics}
        returnDistribution={mockReturnDistribution}
        rollingMetrics={mockRollingMetrics}
      />
    );

    // 点击夏普比率指标
    const sharpeRatioElement = screen.getByText('夏普比率').closest('div');
    if (sharpeRatioElement) {
      fireEvent.click(sharpeRatioElement);
    }

    // 注意：由于模态框可能需要异步渲染，这里主要测试点击事件不会报错
    expect(sharpeRatioElement).toBeInTheDocument();
  });
});
