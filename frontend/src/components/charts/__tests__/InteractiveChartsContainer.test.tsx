/**
 * 交互式图表容器组件测试
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import InteractiveChartsContainer from '../InteractiveChartsContainer';
import { BacktestService } from '../../../services/backtestService';

// Mock BacktestService
jest.mock('../../../services/backtestService');
const mockBacktestService = BacktestService as jest.Mocked<typeof BacktestService>;

// Mock ECharts
jest.mock('echarts', () => ({
  init: jest.fn(() => ({
    setOption: jest.fn(),
    dispose: jest.fn(),
    resize: jest.fn(),
    dispatchAction: jest.fn(),
    getDataURL: jest.fn(() => 'data:image/png;base64,mock'),
  })),
  graphic: {
    LinearGradient: jest.fn(),
  },
}));

describe('InteractiveChartsContainer', () => {
  const mockTaskId = 'test-task-id';
  const mockBacktestData = {
    portfolio_history: [
      {
        date: '2023-01-01',
        portfolio_value: 100000,
        total_return: 0,
        daily_return: 0,
      },
      {
        date: '2023-01-02',
        portfolio_value: 101000,
        total_return: 0.01,
        daily_return: 0.01,
      },
    ],
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('应该正确渲染加载状态', () => {
    // Mock API调用返回pending promise
    mockBacktestService.getChartData.mockImplementation(() => new Promise(() => {}));

    render(
      <InteractiveChartsContainer
        taskId={mockTaskId}
        backtestData={mockBacktestData}
      />
    );

    expect(screen.getByText('正在加载图表数据...')).toBeInTheDocument();
  });

  it('应该正确渲染图表标签页', async () => {
    // Mock API调用
    mockBacktestService.getChartData.mockImplementation((taskId, chartType) => {
      switch (chartType) {
        case 'equity_curve':
          return Promise.resolve({
            dates: ['2023-01-01', '2023-01-02'],
            portfolioValues: [100000, 101000],
            returns: [0, 0.01],
            dailyReturns: [0, 0.01],
          });
        case 'drawdown_curve':
          return Promise.resolve({
            dates: ['2023-01-01', '2023-01-02'],
            drawdowns: [0, -0.5],
            maxDrawdown: -0.5,
            maxDrawdownDate: '2023-01-02',
            maxDrawdownDuration: 1,
          });
        case 'monthly_heatmap':
          return Promise.resolve({
            monthlyReturns: [
              { year: 2023, month: 1, return: 0.01, date: '2023-01' },
            ],
            years: [2023],
            months: [1],
          });
        default:
          return Promise.resolve({});
      }
    });

    mockBacktestService.getBenchmarkData.mockRejectedValue(new Error('No benchmark data'));

    render(
      <InteractiveChartsContainer
        taskId={mockTaskId}
        backtestData={mockBacktestData}
      />
    );

    // 等待加载完成
    await waitFor(() => {
      expect(screen.getByText('交互式图表分析')).toBeInTheDocument();
    });

    // 检查标签页是否存在
    expect(screen.getByText('收益曲线')).toBeInTheDocument();
    expect(screen.getByText('回撤分析')).toBeInTheDocument();
    expect(screen.getByText('月度热力图')).toBeInTheDocument();
  });

  it('应该正确处理API错误', async () => {
    // Mock API调用失败
    mockBacktestService.getChartData.mockRejectedValue(new Error('API Error'));

    render(
      <InteractiveChartsContainer
        taskId={mockTaskId}
        backtestData={mockBacktestData}
      />
    );

    // 等待错误状态显示
    await waitFor(() => {
      expect(screen.getByText('图表数据加载失败')).toBeInTheDocument();
      expect(screen.getByText('API Error')).toBeInTheDocument();
    });

    // 检查重试按钮是否存在
    expect(screen.getByText('重试')).toBeInTheDocument();
  });

  it('应该正确调用API获取图表数据', async () => {
    mockBacktestService.getChartData.mockResolvedValue({});
    mockBacktestService.getBenchmarkData.mockRejectedValue(new Error('No benchmark'));

    render(
      <InteractiveChartsContainer
        taskId={mockTaskId}
        backtestData={mockBacktestData}
      />
    );

    await waitFor(() => {
      expect(mockBacktestService.getChartData).toHaveBeenCalledWith(mockTaskId, 'equity_curve', false);
      expect(mockBacktestService.getChartData).toHaveBeenCalledWith(mockTaskId, 'drawdown_curve', false);
      expect(mockBacktestService.getChartData).toHaveBeenCalledWith(mockTaskId, 'monthly_heatmap', false);
      expect(mockBacktestService.getBenchmarkData).toHaveBeenCalledWith(mockTaskId);
    });
  });
});