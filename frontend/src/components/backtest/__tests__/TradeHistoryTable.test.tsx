/**
 * 交易记录表格组件测试
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { TradeHistoryTable } from '../TradeHistoryTable';
import { BacktestService } from '../../../services/backtestService';

// Mock BacktestService
jest.mock('../../../services/backtestService');
const mockBacktestService = BacktestService as jest.Mocked<typeof BacktestService>;

// Mock data
const mockTrades = [
  {
    id: 1,
    task_id: 'test-task-1',
    trade_id: 'trade-1',
    stock_code: '000001.SZ',
    action: 'BUY' as const,
    quantity: 1000,
    price: 10.50,
    timestamp: '2024-01-01T09:30:00Z',
    commission: 5.25,
    pnl: 0,
  },
  {
    id: 2,
    task_id: 'test-task-1',
    trade_id: 'trade-2',
    stock_code: '000002.SZ',
    action: 'SELL' as const,
    quantity: 1000,
    price: 11.00,
    timestamp: '2024-01-02T15:00:00Z',
    commission: 5.50,
    pnl: 494.25,
  },
];

const mockStatistics = {
  total_trades: 2,
  buy_trades: 1,
  sell_trades: 1,
  winning_trades: 1,
  losing_trades: 0,
  win_rate: 1.0,
  avg_profit: 494.25,
  avg_loss: 0,
  profit_factor: Infinity,
  total_commission: 10.75,
  total_pnl: 494.25,
};

describe('TradeHistoryTable', () => {
  beforeEach(() => {
    mockBacktestService.getTradeRecords.mockResolvedValue({
      trades: mockTrades,
      pagination: {
        offset: 0,
        limit: 50,
        count: 2,
      },
    });

    mockBacktestService.getTradeStatistics.mockResolvedValue(mockStatistics);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('应该渲染交易记录表格', async () => {
    render(<TradeHistoryTable taskId="test-task-1" />);

    // 等待数据加载
    await waitFor(() => {
      expect(screen.getAllByText('000001.SZ')[0]).toBeInTheDocument();
    });

    // 验证表格内容
    expect(screen.getByText('买入')).toBeInTheDocument();
    expect(screen.getByText('卖出')).toBeInTheDocument();
    expect(screen.getAllByText('¥494.25')[0]).toBeInTheDocument();
  });

  it('应该显示统计信息', async () => {
    render(<TradeHistoryTable taskId="test-task-1" />);

    // 等待统计数据加载
    await waitFor(() => {
      expect(screen.getByText('2')).toBeInTheDocument(); // 总交易次数
    });

    expect(screen.getByText('100.00%')).toBeInTheDocument(); // 胜率
  });

  it('应该处理加载状态', () => {
    // Mock 延迟响应
    mockBacktestService.getTradeRecords.mockImplementation(
      () => new Promise(resolve => setTimeout(resolve, 1000))
    );

    render(<TradeHistoryTable taskId="test-task-1" />);

    expect(screen.getByText('加载交易记录中...')).toBeInTheDocument();
  });

  it('应该处理错误状态', async () => {
    mockBacktestService.getTradeRecords.mockRejectedValue(
      new Error('获取交易记录失败')
    );

    render(<TradeHistoryTable taskId="test-task-1" />);

    await waitFor(() => {
      expect(screen.getByText('获取交易记录失败')).toBeInTheDocument();
    });
  });
});