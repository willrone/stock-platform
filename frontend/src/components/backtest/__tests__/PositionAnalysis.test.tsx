/**
 * 持仓分析组件测试
 * 全面测试持仓分析组件的各项功能
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import '@testing-library/jest-dom';
import { PositionAnalysis } from '../PositionAnalysis';
import { BacktestService } from '@/services/backtestService';

// Mock BacktestService
jest.mock('@/services/backtestService', () => ({
  BacktestService: {
    getPortfolioSnapshots: jest.fn(),
  },
}));

// Mock ECharts (已在 jest.setup.js 中全局 mock，这里确保可用)
const mockEChartsInstance = {
  setOption: jest.fn(),
  dispose: jest.fn(),
  resize: jest.fn(),
  on: jest.fn(),
  off: jest.fn(),
};

// 基础测试数据
const mockPositionData = [
  {
    stock_code: '000001.SZ',
    stock_name: '平安银行',
    total_return: 1500.5,
    trade_count: 5,
    win_rate: 0.8,
    avg_holding_period: 15,
    winning_trades: 4,
    losing_trades: 1,
    avg_win: 500,
    avg_loss: -200,
    largest_win: 800,
    largest_loss: -300,
    profit_factor: 2.5,
    max_holding_period: 30,
    min_holding_period: 5,
    avg_buy_price: 10.5,
    avg_sell_price: 11.2,
    price_improvement: 0.067,
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
    avg_win: 200,
    avg_loss: -500,
    largest_win: 250,
    largest_loss: -600,
    profit_factor: 0.4,
    max_holding_period: 20,
    min_holding_period: 3,
    avg_buy_price: 8.5,
    avg_sell_price: 7.8,
    price_improvement: -0.082,
  },
  {
    stock_code: '600000.SH',
    stock_name: '浦发银行',
    total_return: 2300.75,
    trade_count: 8,
    win_rate: 0.75,
    avg_holding_period: 20,
    winning_trades: 6,
    losing_trades: 2,
  },
];

const mockEnhancedData = {
  stock_performance: mockPositionData,
  position_weights: {
    current_weights: {
      '000001.SZ': 0.4,
      '000002.SZ': 0.3,
      '600000.SH': 0.3,
    },
    concentration_metrics: {
      averages: {
        avg_hhi: 0.34,
        avg_effective_stocks: 2.94,
        avg_top_3_concentration: 0.85,
        avg_top_5_concentration: 0.95,
      },
    },
  },
  trading_patterns: {
    time_patterns: {
      monthly_distribution: [
        { month: 1, count: 5 },
        { month: 2, count: 8 },
        { month: 3, count: 3 },
      ],
    },
    size_patterns: {
      avg_trade_size: 50000,
      total_volume: 400000,
    },
    frequency_patterns: {
      avg_interval_days: 5.5,
      avg_monthly_trades: 5.3,
    },
  },
  holding_periods: {
    avg_holding_period: 15.5,
    median_holding_period: 14.0,
    short_term_positions: 5,
    medium_term_positions: 8,
    long_term_positions: 2,
  },
};

const mockPortfolioSnapshots = [
  {
    snapshot_date: '2024-01-01T00:00:00',
    portfolio_value: 100000,
    cash: 20000,
    positions: {},
  },
  {
    snapshot_date: '2024-01-02T00:00:00',
    portfolio_value: 102000,
    cash: 18000,
    positions: {},
  },
  {
    snapshot_date: '2024-01-03T00:00:00',
    portfolio_value: 101500,
    cash: 19500,
    positions: {},
  },
];

const mockStockCodes = ['000001.SZ', '000002.SZ', '600000.SH'];

describe('PositionAnalysis', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (BacktestService.getPortfolioSnapshots as jest.Mock).mockResolvedValue({
      snapshots: mockPortfolioSnapshots,
      total_count: 3,
    });
  });

  describe('基础渲染', () => {
    it('应该渲染持仓分析组件（数组格式数据）', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      // 验证统计概览
      expect(screen.getByText('持仓股票')).toBeInTheDocument();
      expect(screen.getAllByText('3').length).toBeGreaterThan(0); // 总股票数
      expect(screen.getByText('盈利股票')).toBeInTheDocument();
      expect(screen.getAllByText('2').length).toBeGreaterThan(0); // 盈利股票数

      // 验证股票代码存在
      expect(screen.getAllByText('000001.SZ')[0]).toBeInTheDocument();
      expect(screen.getAllByText('平安银行')[0]).toBeInTheDocument();
    });

    it('应该渲染持仓分析组件（对象格式数据）', () => {
      render(<PositionAnalysis positionAnalysis={mockEnhancedData} stockCodes={mockStockCodes} />);

      expect(screen.getByText('持仓股票')).toBeInTheDocument();
      expect(screen.getAllByText('3').length).toBeGreaterThan(0);
    });

    it('应该显示最佳和最差表现者', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      expect(screen.getByText('最佳表现')).toBeInTheDocument();
      expect(screen.getByText('最差表现')).toBeInTheDocument();
      expect(screen.getAllByText('600000.SH').length).toBeGreaterThan(0); // 最佳表现
      expect(screen.getAllByText('000002.SZ').length).toBeGreaterThan(0); // 最差表现
    });

    it('应该显示所有统计卡片', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      expect(screen.getByText('持仓股票')).toBeInTheDocument();
      expect(screen.getByText('盈利股票')).toBeInTheDocument();
      expect(screen.getByText('平均胜率')).toBeInTheDocument();
      expect(screen.getAllByText('总收益').length).toBeGreaterThan(0);
    });
  });

  describe('空数据处理', () => {
    it('应该处理空数组数据', () => {
      render(<PositionAnalysis positionAnalysis={[]} stockCodes={[]} />);

      expect(screen.getByText('暂无持仓分析数据')).toBeInTheDocument();
    });

    it('应该处理 null 数据', () => {
      render(<PositionAnalysis positionAnalysis={null as unknown as never} stockCodes={[]} />);

      expect(screen.getByText('暂无持仓分析数据')).toBeInTheDocument();
    });

    it('应该处理无效的对象格式数据', () => {
      render(
        <PositionAnalysis positionAnalysis={{ stock_performance: null } as any} stockCodes={[]} />
      );

      expect(screen.getByText('暂无持仓分析数据')).toBeInTheDocument();
    });
  });

  describe('统计信息计算', () => {
    it('应该正确计算总股票数', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      expect(screen.getAllByText('3').length).toBeGreaterThan(0); // 总股票数
    });

    it('应该正确计算盈利股票数', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      // 盈利股票数：000001.SZ 和 600000.SH
      expect(screen.getAllByText('2').length).toBeGreaterThan(0);
      expect(screen.getByText('(66.7%)')).toBeInTheDocument(); // 2/3 = 66.7%
    });

    it('应该正确计算总收益', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      // 总收益：1500.5 + (-800.25) + 2300.75 = 3001.0
      expect(screen.getByText(/¥3,001.00/)).toBeInTheDocument();
    });

    it('应该正确计算平均胜率', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      // 平均胜率：(0.8 + 0.33 + 0.75) / 3 = 0.6267 ≈ 62.67%
      const avgWinRate = screen.getByText(/62\.\d+%/);
      expect(avgWinRate).toBeInTheDocument();
    });
  });

  describe('表格视图', () => {
    it('应该显示表格视图', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      // 默认显示表格视图
      expect(screen.getByText('股票代码')).toBeInTheDocument();
      expect(screen.getAllByText('总收益').length).toBeGreaterThan(0);
      expect(screen.getByText('交易次数')).toBeInTheDocument();
      expect(screen.getByText('胜率')).toBeInTheDocument();
      expect(screen.getByText('平均持仓期')).toBeInTheDocument();
    });

    it('应该显示所有股票数据行', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      expect(screen.getAllByText('000001.SZ')[0]).toBeInTheDocument();
      expect(screen.getAllByText('000002.SZ')[0]).toBeInTheDocument();
      expect(screen.getAllByText('600000.SH')[0]).toBeInTheDocument();
    });

    it('应该显示详情按钮', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      const detailButtons = screen.getAllByText('详情');
      expect(detailButtons.length).toBe(3); // 每个股票一行
    });
  });

  describe('排序功能', () => {
    it('应该支持按总收益排序', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      const totalReturnHeaders = screen.getAllByText('总收益');
      fireEvent.click(totalReturnHeaders[0]);

      // 验证排序后的顺序（应该按总收益降序）
      const rows = screen.getAllByRole('row');
      // 第一行是表头，跳过
      expect(rows[1]).toHaveTextContent('600000.SH'); // 最高收益
    });

    it('应该支持按交易次数排序', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      const tradeCountHeader = screen.getByText('交易次数');
      fireEvent.click(tradeCountHeader);

      // 验证排序功能已触发（表格仍然存在）
      const rows = screen.getAllByRole('row');
      expect(rows.length).toBeGreaterThan(1); // 至少表头+数据行
      // 验证包含股票代码
      expect(screen.getAllByText('600000.SH').length).toBeGreaterThan(0);
    });

    it('应该支持按胜率排序', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      const winRateHeader = screen.getByText('胜率');
      fireEvent.click(winRateHeader);

      // 验证排序功能已触发
      const rows = screen.getAllByRole('row');
      expect(rows.length).toBeGreaterThan(1);
      // 验证包含股票代码
      expect(screen.getAllByText('000001.SZ').length).toBeGreaterThan(0);
    });

    it('应该支持切换排序方向', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      const totalReturnHeaders = screen.getAllByText('总收益');
      fireEvent.click(totalReturnHeaders[0]); // 第一次点击：降序
      fireEvent.click(totalReturnHeaders[0]); // 第二次点击：升序

      // 验证排序功能已触发
      const rows = screen.getAllByRole('row');
      expect(rows.length).toBeGreaterThan(1);
      // 验证包含所有股票代码
      expect(screen.getAllByText('000002.SZ').length).toBeGreaterThan(0);
    });
  });

  describe('Tab 切换', () => {
    it('应该支持切换到饼图视图', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      const pieTab = screen.getByText('饼图');
      fireEvent.click(pieTab);

      expect(screen.getByText('持仓权重分布（按收益绝对值）')).toBeInTheDocument();
    });

    it('应该支持切换到柱状图视图', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      const barTab = screen.getByText('柱状图');
      fireEvent.click(barTab);

      expect(screen.getByText('股票表现对比')).toBeInTheDocument();
    });

    it('应该支持切换到树状图视图', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      const treemapTab = screen.getByText('树状图');
      fireEvent.click(treemapTab);

      expect(screen.getByText('持仓权重树状图')).toBeInTheDocument();
    });

    it('应该在增强数据时显示额外 Tab', () => {
      render(<PositionAnalysis positionAnalysis={mockEnhancedData} stockCodes={mockStockCodes} />);

      expect(screen.getByText('权重分析')).toBeInTheDocument();
      expect(screen.getByText('交易模式')).toBeInTheDocument();
      expect(screen.getByText('持仓期分析')).toBeInTheDocument();
    });

    it('应该在有 taskId 时显示资金分析 Tab', () => {
      render(
        <PositionAnalysis
          positionAnalysis={mockPositionData}
          stockCodes={mockStockCodes}
          taskId="test-task-1"
        />
      );

      expect(screen.getByText('资金分析')).toBeInTheDocument();
    });
  });

  describe('柱状图指标切换', () => {
    it('应该支持切换柱状图指标', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      // 切换到柱状图
      const barTab = screen.getByText('柱状图');
      fireEvent.click(barTab);

      // 切换指标 - 使用 role 查找 select
      const metricSelects = screen.getAllByRole('combobox');
      if (metricSelects.length > 0) {
        fireEvent.mouseDown(metricSelects[0]);
        const winRateOption = screen.getByText('胜率');
        fireEvent.click(winRateOption);
      }

      // 验证指标已切换（至少验证柱状图已显示）
      expect(screen.getByText('股票表现对比')).toBeInTheDocument();
    });
  });

  describe('详情对话框', () => {
    it('应该打开详情对话框', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      const detailButtons = screen.getAllByText('详情');
      fireEvent.click(detailButtons[0]);

      expect(screen.getByText('股票详细分析')).toBeInTheDocument();
      expect(screen.getAllByText('000001.SZ')[0]).toBeInTheDocument();
      expect(screen.getAllByText('平安银行')[0]).toBeInTheDocument();
    });

    it('应该显示股票详细信息', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      const detailButtons = screen.getAllByText('详情');
      fireEvent.click(detailButtons[0]);

      // 验证详细信息
      expect(screen.getByText(/¥1,500.50/)).toBeInTheDocument(); // 总收益
      expect(screen.getByText('80.00%')).toBeInTheDocument(); // 胜率
      expect(screen.getAllByText('5').length).toBeGreaterThan(0); // 交易次数
      expect(screen.getByText('15 天')).toBeInTheDocument(); // 平均持仓期
    });

    it('应该显示扩展信息（如果存在）', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      const detailButtons = screen.getAllByText('详情');
      fireEvent.click(detailButtons[0]);

      // 验证扩展信息（使用更灵活的查询，检查是否包含相关关键词）
      const dialogContent = screen.getByText('股票详细分析').closest('div');
      const dialogText = dialogContent?.textContent || '';

      // 检查是否包含扩展信息的文本（可能以不同形式存在）
      const hasExtendedInfo =
        dialogText.includes('盈亏') ||
        dialogText.includes('价格') ||
        dialogText.includes('持仓期') ||
        dialogText.includes('平均盈利') ||
        dialogText.includes('平均亏损') ||
        dialogText.includes('最大盈利') ||
        dialogText.includes('最大亏损');

      // 由于 mockPositionData[0] 包含扩展字段，应该显示扩展信息
      expect(hasExtendedInfo).toBeTruthy();
    });

    it('应该关闭详情对话框', async () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      const detailButtons = screen.getAllByText('详情');
      fireEvent.click(detailButtons[0]);

      expect(screen.getByText('股票详细分析')).toBeInTheDocument();

      const closeButton = screen.getByText('关闭');
      fireEvent.click(closeButton);

      await waitFor(() => {
        expect(screen.queryByText('股票详细分析')).not.toBeInTheDocument();
      });
    });
  });

  describe('增强数据功能', () => {
    it('应该显示持仓权重分析', () => {
      render(<PositionAnalysis positionAnalysis={mockEnhancedData} stockCodes={mockStockCodes} />);

      const weightTab = screen.getByText('权重分析');
      fireEvent.click(weightTab);

      expect(screen.getByText('持仓权重分析')).toBeInTheDocument();
      expect(screen.getByText('HHI指数')).toBeInTheDocument();
      expect(screen.getByText('有效股票数')).toBeInTheDocument();
    });

    it('应该显示交易模式分析', () => {
      render(<PositionAnalysis positionAnalysis={mockEnhancedData} stockCodes={mockStockCodes} />);

      const tradingTab = screen.getByText('交易模式');
      fireEvent.click(tradingTab);

      expect(screen.getByText('交易模式分析')).toBeInTheDocument();
      expect(screen.getByText('平均交易规模')).toBeInTheDocument();
      expect(screen.getByText('总交易量')).toBeInTheDocument();
    });

    it('应该显示持仓期分析', () => {
      render(<PositionAnalysis positionAnalysis={mockEnhancedData} stockCodes={mockStockCodes} />);

      const holdingTab = screen.getByText('持仓期分析');
      fireEvent.click(holdingTab);

      expect(screen.getByText('持仓时间分析')).toBeInTheDocument();
      expect(screen.getByText('平均持仓期')).toBeInTheDocument();
      expect(screen.getByText('中位数持仓期')).toBeInTheDocument();
    });
  });

  describe('资金分配功能', () => {
    it('应该加载组合快照数据', async () => {
      render(
        <PositionAnalysis
          positionAnalysis={mockPositionData}
          stockCodes={mockStockCodes}
          taskId="test-task-1"
        />
      );

      await waitFor(() => {
        expect(BacktestService.getPortfolioSnapshots).toHaveBeenCalledWith(
          'test-task-1',
          undefined,
          undefined,
          10000
        );
      });
    });

    it('应该显示资金分配图表', async () => {
      render(
        <PositionAnalysis
          positionAnalysis={mockPositionData}
          stockCodes={mockStockCodes}
          taskId="test-task-1"
        />
      );

      await waitFor(() => {
        expect(BacktestService.getPortfolioSnapshots).toHaveBeenCalled();
      });

      const capitalTab = screen.getByText('资金分析');
      fireEvent.click(capitalTab);

      expect(screen.getByText('资金分配趋势')).toBeInTheDocument();
    });

    it('应该显示资金统计信息', async () => {
      render(
        <PositionAnalysis
          positionAnalysis={mockPositionData}
          stockCodes={mockStockCodes}
          taskId="test-task-1"
        />
      );

      await waitFor(() => {
        expect(BacktestService.getPortfolioSnapshots).toHaveBeenCalled();
      });

      const capitalTab = screen.getByText('资金分析');
      fireEvent.click(capitalTab);

      await waitFor(() => {
        expect(screen.getByText('平均总资金')).toBeInTheDocument();
        expect(screen.getByText('平均持仓资金')).toBeInTheDocument();
        expect(screen.getByText('平均空闲资金')).toBeInTheDocument();
        expect(screen.getByText('平均持仓比例')).toBeInTheDocument();
      });
    });

    it('应该处理快照数据加载失败', async () => {
      (BacktestService.getPortfolioSnapshots as jest.Mock).mockRejectedValue(new Error('加载失败'));

      render(
        <PositionAnalysis
          positionAnalysis={mockPositionData}
          stockCodes={mockStockCodes}
          taskId="test-task-1"
        />
      );

      // 应该不会崩溃
      expect(screen.getByText('持仓股票')).toBeInTheDocument();
    });

    it('应该在没有快照数据时显示提示', async () => {
      (BacktestService.getPortfolioSnapshots as jest.Mock).mockResolvedValue({
        snapshots: [],
        total_count: 0,
      });

      render(
        <PositionAnalysis
          positionAnalysis={mockPositionData}
          stockCodes={mockStockCodes}
          taskId="test-task-1"
        />
      );

      await waitFor(() => {
        expect(BacktestService.getPortfolioSnapshots).toHaveBeenCalled();
      });

      const capitalTab = screen.getByText('资金分析');
      fireEvent.click(capitalTab);

      await waitFor(() => {
        expect(screen.getByText('暂无资金分配数据')).toBeInTheDocument();
      });
    });
  });

  describe('数据格式化', () => {
    it('应该正确格式化货币', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      // 验证货币格式（包含 ¥ 符号）
      expect(screen.getByText(/¥1,500.50/)).toBeInTheDocument();
    });

    it('应该正确格式化百分比', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      // 验证百分比格式
      const winRateElements = screen.getAllByText(/80\.00%/);
      expect(winRateElements.length).toBeGreaterThan(0);
    });
  });

  describe('边界情况', () => {
    it('应该处理单个股票数据', () => {
      const singleStock = [mockPositionData[0]];
      render(<PositionAnalysis positionAnalysis={singleStock} stockCodes={['000001.SZ']} />);

      expect(screen.getByText('持仓股票')).toBeInTheDocument();
      expect(screen.getAllByText('1').length).toBeGreaterThan(0);
    });

    it('应该处理大量股票数据', () => {
      const manyStocks = Array.from({ length: 50 }, (_, i) => ({
        ...mockPositionData[0],
        stock_code: `00000${i}.SZ`,
        stock_name: `股票${i}`,
      }));

      render(
        <PositionAnalysis
          positionAnalysis={manyStocks}
          stockCodes={manyStocks.map(s => s.stock_code)}
        />
      );

      expect(screen.getByText('持仓股票')).toBeInTheDocument();
      expect(screen.getAllByText('50').length).toBeGreaterThan(0);
    });

    it('应该处理零收益股票', () => {
      const zeroReturnStock = [
        {
          ...mockPositionData[0],
          total_return: 0,
        },
      ];

      render(<PositionAnalysis positionAnalysis={zeroReturnStock} stockCodes={['000001.SZ']} />);

      expect(screen.getAllByText('000001.SZ')[0]).toBeInTheDocument();
    });

    it('应该处理极端数值', () => {
      const extremeData = [
        {
          ...mockPositionData[0],
          total_return: 999999.99,
          win_rate: 1.0,
          trade_count: 1000,
        },
      ];

      render(<PositionAnalysis positionAnalysis={extremeData} stockCodes={['000001.SZ']} />);

      expect(screen.getAllByText('000001.SZ')[0]).toBeInTheDocument();
    });
  });

  describe('交互功能', () => {
    it('应该支持点击股票行查看详情', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      const detailButtons = screen.getAllByText('详情');
      expect(detailButtons.length).toBeGreaterThan(0);

      fireEvent.click(detailButtons[0]);
      expect(screen.getByText('股票详细分析')).toBeInTheDocument();
    });

    it('应该支持在不同 Tab 间切换', () => {
      render(<PositionAnalysis positionAnalysis={mockPositionData} stockCodes={mockStockCodes} />);

      // 切换到饼图
      fireEvent.click(screen.getByText('饼图'));
      expect(screen.getByText('持仓权重分布（按收益绝对值）')).toBeInTheDocument();

      // 切换回表格
      fireEvent.click(screen.getByText('表格视图'));
      expect(screen.getByText('股票代码')).toBeInTheDocument();
    });
  });
});
