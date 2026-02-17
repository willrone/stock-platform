/**
 * 资金分配趋势图表组件
 */

import React from 'react';
import { Box, Card, CardHeader, CardContent, Typography } from '@mui/material';
import { useECharts } from '@/hooks/backtest/useECharts';

interface CapitalChartProps {
  data: {
    dates: string[];
    totalCapital: number[];
    positionCapital: number[];
    freeCapital: number[];
  } | null;
  loading: boolean;
  isActive: boolean;
}

export const CapitalChart: React.FC<CapitalChartProps> = ({ data, loading, isActive }) => {
  const chartRef = useECharts(
    data,
    chartData => ({
      title: {
        text: '资金分配趋势',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold',
        },
      },
      tooltip: {
        trigger: 'axis',
        formatter: function (params: { axisValue: string; marker: string; seriesName: string; value: number }[]) {
          let result = `${params[0].axisValue}<br/>`;
          params.forEach((param: { marker: string; seriesName: string; value: number }) => {
            result += `${param.marker}${param.seriesName}: ¥${param.value.toLocaleString('zh-CN', {
              minimumFractionDigits: 2,
              maximumFractionDigits: 2,
            })}<br/>`;
          });
          return result;
        },
      },
      legend: {
        data: ['总资金', '持仓资金', '空闲资金'],
        top: 30,
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        top: '15%',
        containLabel: true,
      },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: chartData.dates,
        axisLabel: {
          rotate: 45,
          formatter: function (value: string) {
            return value.split('T')[0]; // 只显示日期部分
          },
        },
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: function (value: number) {
            if (value >= 10000) {
              return `¥${(value / 10000).toFixed(1)}万`;
            }
            return `¥${value.toFixed(0)}`;
          },
        },
      },
      series: [
        {
          name: '总资金',
          type: 'line',
          data: chartData.totalCapital,
          smooth: true,
          itemStyle: {
            color: '#3b82f6', // 蓝色
          },
          areaStyle: {
            opacity: 0.1,
          },
        },
        {
          name: '持仓资金',
          type: 'line',
          data: chartData.positionCapital,
          smooth: true,
          itemStyle: {
            color: '#10b981', // 绿色
          },
          areaStyle: {
            opacity: 0.1,
          },
        },
        {
          name: '空闲资金',
          type: 'line',
          data: chartData.freeCapital,
          smooth: true,
          itemStyle: {
            color: '#f59e0b', // 橙色
          },
          areaStyle: {
            opacity: 0.1,
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
              资金分配趋势
            </Typography>
            <Typography variant="caption" color="text.secondary">
              展示每天的持仓资金、空闲资金和总资金变化
            </Typography>
          </Box>
        }
      />
      <CardContent sx={{ p: { xs: 1.5, md: 2 } }}>
        {loading ? (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="body2" color="text.secondary">
              加载资金分配数据中...
            </Typography>
          </Box>
        ) : data && data.dates.length > 0 ? (
          <Box ref={chartRef} sx={{ height: 400, width: '100%', overflowX: 'auto' }} />
        ) : (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="body2" color="text.secondary">
              暂无资金分配数据
            </Typography>
          </Box>
        )}
        {/* 资金统计信息 */}
        {data && data.dates.length > 0 && (
          <Box
            sx={{
              mt: 3,
              display: 'grid',
              gridTemplateColumns: { xs: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' },
              gap: 2,
            }}
          >
            <Box
              sx={{
                textAlign: 'center',
                p: 1.5,
                bgcolor: 'primary.light',
                borderRadius: 1,
                overflow: 'hidden',
                wordBreak: 'break-word',
              }}
            >
              <Typography variant="caption" color="text.secondary">
                平均总资金
              </Typography>
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 600,
                  color: 'primary.dark',
                  fontSize: { xs: '0.95rem', md: '1.25rem' },
                }}
              >
                ¥
                {(
                  data.totalCapital.reduce((a, b) => a + b, 0) /
                  data.totalCapital.length /
                  10000
                ).toFixed(2)}
                万
              </Typography>
            </Box>
            <Box
              sx={{
                textAlign: 'center',
                p: 1.5,
                bgcolor: 'success.light',
                borderRadius: 1,
                overflow: 'hidden',
                wordBreak: 'break-word',
              }}
            >
              <Typography variant="caption" color="text.secondary">
                平均持仓资金
              </Typography>
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 600,
                  color: 'success.dark',
                  fontSize: { xs: '0.95rem', md: '1.25rem' },
                }}
              >
                ¥
                {(
                  data.positionCapital.reduce((a, b) => a + b, 0) /
                  data.positionCapital.length /
                  10000
                ).toFixed(2)}
                万
              </Typography>
            </Box>
            <Box
              sx={{
                textAlign: 'center',
                p: 1.5,
                bgcolor: 'warning.light',
                borderRadius: 1,
                overflow: 'hidden',
                wordBreak: 'break-word',
              }}
            >
              <Typography variant="caption" color="text.secondary">
                平均空闲资金
              </Typography>
              <Typography
                variant="h6"
                sx={{
                  fontWeight: 600,
                  color: 'warning.dark',
                  fontSize: { xs: '0.95rem', md: '1.25rem' },
                }}
              >
                ¥
                {(
                  data.freeCapital.reduce((a, b) => a + b, 0) /
                  data.freeCapital.length /
                  10000
                ).toFixed(2)}
                万
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
                平均持仓比例
              </Typography>
              <Typography
                variant="h6"
                sx={{ fontWeight: 600, fontSize: { xs: '0.95rem', md: '1.25rem' } }}
              >
                {(
                  (data.positionCapital.reduce((a, b) => a + b, 0) /
                    data.totalCapital.reduce((a, b) => a + b, 0)) *
                  100
                ).toFixed(1)}
                %
              </Typography>
            </Box>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};
