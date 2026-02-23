/**
 * 持仓权重分析图表组件
 */

import React from 'react';
import { Box, Card, CardHeader, CardContent, Typography } from '@mui/material';
import { useECharts } from '@/hooks/backtest/useECharts';

interface WeightChartProps {
  data: Array<{ name: string; value: number }> | null;
  concentrationMetrics?: any;
  isActive: boolean;
}

export const WeightChart: React.FC<WeightChartProps> = ({
  data,
  concentrationMetrics,
  isActive,
}) => {
  const chartRef = useECharts(
    data,
    (chartData) => ({
      title: {
        text: '持仓权重分布（真实权重）',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold',
        },
      },
      tooltip: {
        trigger: 'item',
        formatter: function (params: any) {
          return `${params.name}<br/>权重: ${params.value.toFixed(2)}%`;
        },
      },
      legend: {
        orient: 'vertical',
        left: 'left',
        data: chartData.map((item: any) => item.name),
      },
      series: [
        {
          name: '持仓权重',
          type: 'pie',
          radius: '50%',
          data: chartData,
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: 'rgba(0, 0, 0, 0.5)',
            },
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
          <Box>
            <Typography variant="h6" component="h3" sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}>
              持仓权重分析
            </Typography>
            <Typography variant="caption" color="text.secondary">
              基于真实持仓权重的分布
            </Typography>
          </Box>
        }
      />
      <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
        {data && data.length > 0 ? (
          <Box ref={chartRef} sx={{ height: 400, width: '100%', overflowX: 'auto' }} />
        ) : (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="body2" color="text.secondary">
              暂无持仓权重数据
            </Typography>
          </Box>
        )}
        {/* 集中度指标 */}
        {concentrationMetrics?.averages && (
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
                HHI指数
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}>
                {concentrationMetrics.averages.avg_hhi.toFixed(3)}
              </Typography>
            </Box>
            <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1, overflow: 'hidden', wordBreak: 'break-word' }}>
              <Typography variant="caption" color="text.secondary">
                有效股票数
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}>
                {concentrationMetrics.averages.avg_effective_stocks.toFixed(1)}
              </Typography>
            </Box>
            <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1, overflow: 'hidden', wordBreak: 'break-word' }}>
              <Typography variant="caption" color="text.secondary">
                前3大集中度
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}>
                {(concentrationMetrics.averages.avg_top_3_concentration * 100).toFixed(1)}%
              </Typography>
            </Box>
            <Box sx={{ textAlign: 'center', p: 1.5, bgcolor: 'grey.50', borderRadius: 1, overflow: 'hidden', wordBreak: 'break-word' }}>
              <Typography variant="caption" color="text.secondary">
                前5大集中度
              </Typography>
              <Typography variant="h6" sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}>
                {(concentrationMetrics.averages.avg_top_5_concentration * 100).toFixed(1)}%
              </Typography>
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};
