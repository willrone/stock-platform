/**
 * 交易模式分析图表组件
 */

import React from 'react';
import { Box, Card, CardHeader, CardContent, Typography } from '@mui/material';
import { useECharts } from '@/hooks/backtest/useECharts';

interface MonthlyDistribution {
  month: number;
  count: number;
}

interface TradingPatternData {
  time_patterns?: {
    monthly_distribution?: MonthlyDistribution[];
  };
  size_patterns?: {
    avg_trade_size: number;
    total_volume: number;
  };
  frequency_patterns?: {
    avg_interval_days: number;
    avg_monthly_trades: number;
  };
}

interface TradingPatternChartProps {
  tradingPatterns: TradingPatternData;
  isActive: boolean;
}

export const TradingPatternChart: React.FC<TradingPatternChartProps> = ({
  tradingPatterns,
  isActive,
}) => {
  const chartRef = useECharts(
    tradingPatterns,
    patterns => ({
      title: {
        text: '交易模式分析',
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '15%',
        top: '15%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        data: patterns.time_patterns?.monthly_distribution?.map((m: MonthlyDistribution) => `${m.month}月`) || [],
      },
      yAxis: {
        type: 'value',
      },
      series: [
        {
          name: '交易次数',
          type: 'bar',
          data: patterns.time_patterns?.monthly_distribution?.map((m: MonthlyDistribution) => m.count) || [],
          itemStyle: {
            color: '#3b82f6',
          },
        },
      ],
    }),
    [],
    isActive
  );

  return (
    <Card>
      <CardHeader
        title={
          <Typography
            variant="h6"
            component="h3"
            sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}
          >
            交易模式分析
          </Typography>
        }
      />
      <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
        <Box ref={chartRef} sx={{ height: 400, width: '100%', overflowX: 'auto' }} />
        {/* 交易模式统计 */}
        <Box
          sx={{
            mt: 3,
            display: 'grid',
            gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
            gap: 2,
          }}
        >
          {tradingPatterns.size_patterns && (
            <>
              <Box
                sx={{
                  textAlign: 'center',
                  p: 1.5,
                  bgcolor: 'grey.50',
                  borderRadius: 1,
                  overflow: 'hidden',
                  wordBreak: 'break-word',
                }}
              >
                <Typography variant="caption" color="text.secondary">
                  平均交易规模
                </Typography>
                <Typography
                  variant="h6"
                  sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}
                >
                  ¥{(tradingPatterns.size_patterns.avg_trade_size / 10000).toFixed(2)}万
                </Typography>
              </Box>
              <Box
                sx={{
                  textAlign: 'center',
                  p: 1.5,
                  bgcolor: 'grey.50',
                  borderRadius: 1,
                  overflow: 'hidden',
                  wordBreak: 'break-word',
                }}
              >
                <Typography variant="caption" color="text.secondary">
                  总交易量
                </Typography>
                <Typography
                  variant="h6"
                  sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}
                >
                  ¥{(tradingPatterns.size_patterns.total_volume / 10000).toFixed(2)}万
                </Typography>
              </Box>
            </>
          )}
          {tradingPatterns.frequency_patterns && (
            <>
              <Box
                sx={{
                  textAlign: 'center',
                  p: 1.5,
                  bgcolor: 'grey.50',
                  borderRadius: 1,
                  overflow: 'hidden',
                  wordBreak: 'break-word',
                }}
              >
                <Typography variant="caption" color="text.secondary">
                  平均间隔
                </Typography>
                <Typography
                  variant="h6"
                  sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}
                >
                  {tradingPatterns.frequency_patterns.avg_interval_days.toFixed(1)}天
                </Typography>
              </Box>
              <Box
                sx={{
                  textAlign: 'center',
                  p: 1.5,
                  bgcolor: 'grey.50',
                  borderRadius: 1,
                  overflow: 'hidden',
                  wordBreak: 'break-word',
                }}
              >
                <Typography variant="caption" color="text.secondary">
                  月度交易次数
                </Typography>
                <Typography
                  variant="h6"
                  sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}
                >
                  {tradingPatterns.frequency_patterns.avg_monthly_trades.toFixed(1)}
                </Typography>
              </Box>
            </>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};
