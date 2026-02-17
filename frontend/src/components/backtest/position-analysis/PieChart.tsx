/**
 * 饼图组件
 */

import React from 'react';
import { Box, Card, CardHeader, CardContent, Typography } from '@mui/material';
import { useECharts } from '@/hooks/backtest/useECharts';

interface PieChartProps {
  data: Array<{ name: string; value: number; originalValue: number }>;
  isActive: boolean;
}

export const PieChart: React.FC<PieChartProps> = ({ data, isActive }) => {
  const chartRef = useECharts(
    data,
    chartData => ({
      title: {
        text: '持仓权重分布',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold',
        },
      },
      tooltip: {
        trigger: 'item',
        formatter: function (params: { name: string; value: number; data: { originalValue: number } }) {
          const percentage = (
            (params.value / chartData.reduce((sum: number, item: { value: number }) => sum + item.value, 0)) *
            100
          ).toFixed(1);
          return `${params.name}<br/>收益: ¥${params.data.originalValue.toFixed(
            2
          )}<br/>占比: ${percentage}%`;
        },
      },
      legend: {
        orient: 'vertical',
        left: 'left',
        data: chartData.map((item: { name: string }) => item.name),
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
      <CardHeader title="持仓权重分布（按收益绝对值）" />
      <CardContent>
        <Box ref={chartRef} sx={{ height: 400, width: '100%' }} />
      </CardContent>
    </Card>
  );
};
