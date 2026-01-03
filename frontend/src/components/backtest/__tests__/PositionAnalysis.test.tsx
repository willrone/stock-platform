/**
 * 持仓分析组件测试
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { PositionAnalysis } from '../PositionAnalysis';

// Mock data
const mockPositionAnalysis = [
  {
    stock_code: '000001.SZ',
    stock_name: '平安银行',
    total_return: 1500.50,
    trade_count: 5,
    win_rate: 0.8,
    avg_holding_period: 15,
    winning_trades: 4,
    losing_trades: 1,
  },
  {
    stock_code: '000002.SZ',
    stock_name: '万科A',
    total_return: -800.25,
    trade_count: 3,
    win_rate: 0.33,
    avg_holding_period: 10,
    winning_trades: 1,
    losing_trades: 2,
  },
];

const mockStockCodes = ['000001.SZ', '000002.SZ'];

describe('PositionAnalysis', () => {
  it('应该渲染持仓分析组件', () => {
    render(
      <PositionAnalysis 
        positionAnalysis={mockPositionAnalysis} 
        stockCodes={mockStockCodes} 
      />
    );

    // 验证统计概览
    expect(screen.getByText('持仓股票')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument(); // 总股票数

    expect(screen.getByText('盈利股票')).toBeInTheDocument();
    expect(screen.getByText('1')).toBeInTheDocument(); // 盈利股票数

    // 验证股票代码存在
    expect(screen.getAllByText('000001.SZ')[0]).toBeInTheDocument();
    expect(screen.getAllByText('平安银行')[0]).toBeInTheDocument();
    expect(screen.getAllByText('000002.SZ')[0]).toBeInTheDocument();
    expect(screen.getAllByText('万科A')[0]).toBeInTheDocument();
  });

  it('应该显示最佳和最差表现者', () => {
    render(
      <PositionAnalysis 
        positionAnalysis={mockPositionAnalysis} 
        stockCodes={mockStockCodes} 
      />
    );

    expect(screen.getByText('最佳表现')).toBeInTheDocument();
    expect(screen.getByText('最差表现')).toBeInTheDocument();
  });

  it('应该处理空数据', () => {
    render(
      <PositionAnalysis 
        positionAnalysis={[]} 
        stockCodes={[]} 
      />
    );

    expect(screen.getByText('暂无持仓分析数据')).toBeInTheDocument();
  });

  it('应该正确计算统计信息', () => {
    render(
      <PositionAnalysis 
        positionAnalysis={mockPositionAnalysis} 
        stockCodes={mockStockCodes} 
      />
    );

    // 验证盈利股票比例
    expect(screen.getByText('(50.0%)')).toBeInTheDocument();

    // 验证总收益（1500.50 - 800.25 = 700.25）
    expect(screen.getByText('¥700.25')).toBeInTheDocument();
  });

  it('应该支持排序功能', () => {
    render(
      <PositionAnalysis 
        positionAnalysis={mockPositionAnalysis} 
        stockCodes={mockStockCodes} 
      />
    );

    // 验证排序指示器存在
    const sortHeaders = screen.getAllByText('↓');
    expect(sortHeaders.length).toBeGreaterThan(0);
  });
});