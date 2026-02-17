/**
 * 树状图组件
 */

import React from 'react';
import { Box, Card, CardHeader, CardContent, Typography } from '@mui/material';
import { useECharts } from '@/hooks/backtest/useECharts';

interface TreemapChartProps {
  data: Array<{
    name: string;
    value: number;
    originalValue: number;
    itemStyle: { color: string };
  }>;
  isActive: boolean;
}

export const TreemapChart: React.FC<TreemapChartProps> = ({ data, isActive }) => {
  const chartRef = useECharts(
    data,
    chartData => ({
      title: {
        text: '持仓权重树状图',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold',
        },
      },
      tooltip: {
        trigger: 'item',
        formatter: function (params: any) {
          const percentage = (
            (params.value / chartData.reduce((sum: number, item: any) => sum + item.value, 0)) *
            100
          ).toFixed(1);
          return `${params.name}<br/>收益: ¥${params.data.originalValue.toFixed(
            2
          )}<br/>占比: ${percentage}%`;
        },
      },
      series: [
        {
          name: '持仓权重',
          type: 'treemap',
          data: chartData,
          roam: false,
          nodeClick: false,
          breadcrumb: {
            show: false,
          },
          label: {
            show: true,
            formatter: function (params: any) {
              const percentage = (
                (params.value / chartData.reduce((sum: number, item: any) => sum + item.value, 0)) *
                100
              ).toFixed(1);
              return `${params.name}\n${percentage}%`;
            },
            color: '#fff',
            fontWeight: 'bold',
          },
          upperLabel: {
            show: false,
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
            <Typography
              variant="h6"
              component="h3"
              sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}
            >
              持仓权重树状图
            </Typography>
            <Typography variant="caption" color="text.secondary">
              绿色表示盈利，红色表示亏损，面积大小表示收益绝对值
            </Typography>
          </Box>
        }
      />
      <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
        <Box ref={chartRef} sx={{ height: 400, width: '100%', overflowX: 'auto' }} />
      </CardContent>
    </Card>
  );
};
