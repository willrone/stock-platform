/**
 * 持仓时间分析图表组件
 */

import React from 'react';
import { Box, Card, CardHeader, CardContent, Typography } from '@mui/material';
import { useECharts } from '@/hooks/backtest/useECharts';

interface HoldingPeriodChartProps {
  holdingPeriods: any;
  isActive: boolean;
}

export const HoldingPeriodChart: React.FC<HoldingPeriodChartProps> = ({
  holdingPeriods,
  isActive,
}) => {
  const chartRef = useECharts(
    holdingPeriods,
    (periods) => ({
      title: {
        text: '持仓时间分布',
        left: 'center',
      },
      tooltip: {
        trigger: 'item',
      },
      series: [
        {
          name: '持仓时间',
          type: 'pie',
          radius: '50%',
          data: [
            {
              value: periods.short_term_positions,
              name: '短期（≤7天）',
              itemStyle: { color: '#3b82f6' },
            },
            {
              value: periods.medium_term_positions,
              name: '中期（7-30天）',
              itemStyle: { color: '#10b981' },
            },
            {
              value: periods.long_term_positions,
              name: '长期（>30天）',
              itemStyle: { color: '#f59e0b' },
            },
          ],
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
          <Typography variant="h6" component="h3" sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}>
            持仓时间分析
          </Typography>
        }
      />
      <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
        <Box ref={chartRef} sx={{ height: 400, width: '100%', overflowX: 'auto' }} />
        {/* 持仓时间统计 */}
        <Box
          sx={{
            mt: 3,
            display: 'grid',
            gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
            gap: 2,
          }}
        >
          <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1, overflow: 'hidden', wordBreak: 'break-word' }}>
            <Typography variant="caption" color="text.secondary">
              平均持仓期
            </Typography>
            <Typography variant="h6" sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}>
              {holdingPeriods.avg_holding_period.toFixed(1)}天
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1, overflow: 'hidden', wordBreak: 'break-word' }}>
            <Typography variant="caption" color="text.secondary">
              中位数持仓期
            </Typography>
            <Typography variant="h6" sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}>
              {holdingPeriods.median_holding_period.toFixed(1)}天
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1, overflow: 'hidden', wordBreak: 'break-word' }}>
            <Typography variant="caption" color="text.secondary">
              短期持仓
            </Typography>
            <Typography variant="h6" sx={{ fontWeight: 600, color: 'primary.main', fontSize: { xs: '0.95rem', md: '1.25rem' } }}>
              {holdingPeriods.short_term_positions}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              ≤7天
            </Typography>
          </Box>
          <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1, overflow: 'hidden', wordBreak: 'break-word' }}>
            <Typography variant="caption" color="text.secondary">
              长期持仓
            </Typography>
            <Typography variant="h6" sx={{ fontWeight: 600, color: 'success.main', fontSize: { xs: '0.95rem', md: '1.25rem' } }}>
              {holdingPeriods.long_term_positions}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              &gt;30天
            </Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};
