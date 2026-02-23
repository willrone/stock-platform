/**
 * 柱状图组件
 */

import React from 'react';
import {
  Box,
  Card,
  CardHeader,
  CardContent,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { useECharts } from '@/hooks/backtest/useECharts';
import { PositionData } from '@/utils/backtest/positionDataUtils';

interface BarChartProps {
  data: Array<{
    stock_code: string;
    total_return: number;
    win_rate: number;
    trade_count: number;
    avg_holding_period: number;
  }>;
  selectedMetric: keyof PositionData;
  onMetricChange: (metric: keyof PositionData) => void;
  isActive: boolean;
}

export const BarChart: React.FC<BarChartProps> = ({
  data,
  selectedMetric,
  onMetricChange,
  isActive,
}) => {
  const getDataByMetric = (chartData: typeof data) => {
    switch (selectedMetric) {
      case 'total_return':
        return chartData.map(item => item.total_return);
      case 'win_rate':
        return chartData.map(item => item.win_rate);
      case 'trade_count':
        return chartData.map(item => item.trade_count);
      case 'avg_holding_period':
        return chartData.map(item => item.avg_holding_period);
      default:
        return chartData.map(item => item.total_return);
    }
  };

  const getMetricName = () => {
    switch (selectedMetric) {
      case 'total_return':
        return '总收益';
      case 'win_rate':
        return '胜率';
      case 'trade_count':
        return '交易次数';
      case 'avg_holding_period':
        return '平均持仓期';
      default:
        return '总收益';
    }
  };

  const chartRef = useECharts(
    data,
    (chartData) => ({
      title: {
        text: '股票表现对比',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold',
        },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow',
        },
        formatter: function (params: any) {
          const param = params[0];
          const value = param.value;

          if (selectedMetric === 'total_return') {
            return `${param.name}<br/>${getMetricName()}: ¥${value.toFixed(2)}`;
          } else if (selectedMetric === 'win_rate') {
            return `${param.name}<br/>${getMetricName()}: ${value.toFixed(2)}%`;
          } else if (selectedMetric === 'avg_holding_period') {
            return `${param.name}<br/>${getMetricName()}: ${value} 天`;
          }
          return `${param.name}<br/>${getMetricName()}: ${value}`;
        },
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
        data: chartData.map((item: any) => item.stock_code),
        axisLabel: {
          rotate: 45,
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: function (value: number) {
            if (selectedMetric === 'total_return') {
              return `¥${(value / 1000).toFixed(0)}K`;
            } else if (selectedMetric === 'win_rate') {
              return `${value.toFixed(0)}%`;
            } else if (selectedMetric === 'avg_holding_period') {
              return `${value}天`;
            }
            return value.toString();
          },
        },
      },
      series: [
        {
          name: getMetricName(),
          type: 'bar',
          data: getDataByMetric(chartData),
          itemStyle: {
            color: '#3b82f6',
          },
        },
      ],
    }),
    [selectedMetric],
    isActive
  );

  return (
    <Card>
      <CardHeader
        title={
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              width: '100%',
            }}
          >
            <Typography variant="h6" component="h3" sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}>
              股票表现对比
            </Typography>
            <FormControl size="small" sx={{ minWidth: 128 }}>
              <InputLabel>指标</InputLabel>
              <Select
                value={selectedMetric}
                label="指标"
                onChange={e => onMetricChange(e.target.value as keyof PositionData)}
              >
                <MenuItem value="total_return">总收益</MenuItem>
                <MenuItem value="win_rate">胜率</MenuItem>
                <MenuItem value="trade_count">交易次数</MenuItem>
                <MenuItem value="avg_holding_period">持仓期</MenuItem>
              </Select>
            </FormControl>
          </Box>
        }
      />
      <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
        <Box ref={chartRef} sx={{ height: 400, width: '100%', overflowX: 'auto' }} />
      </CardContent>
    </Card>
  );
};
