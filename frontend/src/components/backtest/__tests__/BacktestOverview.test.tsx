/**
 * 回测概览组件测试（工单#24：信号执行统计 / Top 拒绝原因）
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import BacktestOverview from '../BacktestOverview';

const theme = createTheme();

/** 最小回测数据：保证 processMetrics 不报错 */
const minimalBacktestData = {
  total_return: 0.05,
  annualized_return: 0.08,
  sharpe_ratio: 1.2,
  max_drawdown: -0.1,
  volatility: 0.15,
  total_trades: 50,
  profit_factor: 1.5,
};

/** 带 signal_execution_summary 的回测数据（工单#24） */
const backtestDataWithSignalSummary = {
  ...minimalBacktestData,
  signal_execution_summary: {
    raw_signal_count: 120,
    actionable_signal_count: 80,
    executed_signal_count: 60,
    execution_rate: 0.5,
    execution_rate_actionable: 0.75,
    top_rejection_reasons: [
      { reason: '资金不足', count: 25 },
      { reason: '已持仓同类', count: 18 },
      { reason: '信号强度不足', count: 12 },
    ],
  },
};

function wrap(ui: React.ReactElement) {
  return <ThemeProvider theme={theme}>{ui}</ThemeProvider>;
}

describe('BacktestOverview (#24 信号执行统计)', () => {
  it('无 signal_execution_summary 时不渲染信号执行统计卡片', () => {
    render(wrap(<BacktestOverview backtestData={minimalBacktestData} />));
    expect(screen.queryByText('信号执行统计')).not.toBeInTheDocument();
  });

  it('有 signal_execution_summary 时渲染信号执行统计卡片和 Top 拒绝原因', () => {
    render(wrap(<BacktestOverview backtestData={backtestDataWithSignalSummary} />));
    expect(screen.getByText('信号执行统计')).toBeInTheDocument();
    expect(screen.getByText('Top 拒绝原因')).toBeInTheDocument();
  });

  it('渲染原始信号数、可执行/实际执行、执行率', () => {
    render(wrap(<BacktestOverview backtestData={backtestDataWithSignalSummary} />));
    expect(screen.getByText('原始信号数')).toBeInTheDocument();
    expect(screen.getByText('120')).toBeInTheDocument();
    expect(screen.getByText(/可执行 \/ 实际执行/)).toBeInTheDocument();
    expect(screen.getByText('80 / 60')).toBeInTheDocument();
    expect(screen.getByText('Top 拒绝原因')).toBeInTheDocument();
  });

  it('渲染 Top 拒绝原因列表内容', () => {
    render(wrap(<BacktestOverview backtestData={backtestDataWithSignalSummary} />));
    expect(screen.getByText('资金不足')).toBeInTheDocument();
    expect(screen.getByText('已持仓同类')).toBeInTheDocument();
    expect(screen.getByText('信号强度不足')).toBeInTheDocument();
    expect(screen.getByText('(25)')).toBeInTheDocument();
    expect(screen.getByText('(18)')).toBeInTheDocument();
    expect(screen.getByText('(12)')).toBeInTheDocument();
  });

  it('仅 top_rejection_reasons 时也显示信号执行统计卡片', () => {
    const dataOnlyReasons = {
      ...minimalBacktestData,
      signal_execution_summary: {
        top_rejection_reasons: [{ reason: '测试拒绝', count: 5 }],
      },
    };
    render(wrap(<BacktestOverview backtestData={dataOnlyReasons} />));
    expect(screen.getByText('信号执行统计')).toBeInTheDocument();
    expect(screen.getByText('Top 拒绝原因')).toBeInTheDocument();
    expect(screen.getByText('测试拒绝')).toBeInTheDocument();
    expect(screen.getByText('(5)')).toBeInTheDocument();
  });
});
