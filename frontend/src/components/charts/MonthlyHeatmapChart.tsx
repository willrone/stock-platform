/**
 * 月度收益热力图组件
 * 显示每月收益率的热力图，便于识别季节性模式
 */

'use client';

import React, { useEffect, useRef, useState } from 'react';
import * as echarts from 'echarts';
import {
  Card,
  CardHeader,
  CardContent,
  Select,
  MenuItem,
  Chip,
  Tooltip,
  Button,
  Box,
  Typography,
  FormControl,
  InputLabel,
  FormControlLabel,
  Checkbox,
  CircularProgress,
  IconButton,
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableCell,
  TableContainer,
  Paper,
} from '@mui/material';
import { Calendar, Info, TrendingUp, TrendingDown, Download } from 'lucide-react';

interface MonthlyHeatmapChartProps {
  taskId: string;
  data: {
    monthlyReturns: Array<{
      year: number;
      month: number;
      return: number;
      date: string;
    }>;
    years: number[];
    months: number[];
  };
  loading?: boolean;
  height?: number;
}

const MONTH_NAMES = [
  '1月',
  '2月',
  '3月',
  '4月',
  '5月',
  '6月',
  '7月',
  '8月',
  '9月',
  '10月',
  '11月',
  '12月',
];

export default function MonthlyHeatmapChart({
  taskId,
  data,
  loading = false,
  height = 400,
}: MonthlyHeatmapChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const [selectedYear, setSelectedYear] = useState<string>('all');
  const [showStatistics, setShowStatistics] = useState(true);

  // 处理年份筛选
  const getFilteredData = () => {
    if (selectedYear === 'all') {
      return data.monthlyReturns;
    }
    return data.monthlyReturns.filter(item => item.year === parseInt(selectedYear));
  };

  // 计算月度统计
  const getMonthlyStatistics = () => {
    const filteredData = getFilteredData();
    const monthlyStats = Array.from({ length: 12 }, (_, index) => {
      const month = index + 1;
      const monthData = filteredData.filter(item => item.month === month);

      if (monthData.length === 0) {
        return {
          month,
          avgReturn: 0,
          count: 0,
          positiveCount: 0,
          negativeCount: 0,
          maxReturn: 0,
          minReturn: 0,
        };
      }

      const returns = monthData.map(item => item.return);
      const positiveReturns = returns.filter(r => r > 0);
      const negativeReturns = returns.filter(r => r < 0);

      return {
        month,
        avgReturn: returns.reduce((sum, r) => sum + r, 0) / returns.length,
        count: returns.length,
        positiveCount: positiveReturns.length,
        negativeCount: negativeReturns.length,
        maxReturn: Math.max(...returns),
        minReturn: Math.min(...returns),
      };
    });

    return monthlyStats;
  };

  // 初始化和更新图表
  useEffect(() => {
    if (!chartRef.current || loading || !data.monthlyReturns.length) {
      return;
    }

    // 销毁现有图表实例
    if (chartInstance.current) {
      chartInstance.current.dispose();
    }

    // 创建新的图表实例
    chartInstance.current = echarts.init(chartRef.current);

    const filteredData = getFilteredData();

    // 准备热力图数据
    const heatmapData: any[] = [];
    const years = selectedYear === 'all' ? data.years : [parseInt(selectedYear)];

    years.forEach((year, yearIndex) => {
      for (let month = 1; month <= 12; month++) {
        const monthData = filteredData.find(item => item.year === year && item.month === month);

        if (monthData) {
          heatmapData.push([
            month - 1, // x轴：月份（0-11）
            yearIndex, // y轴：年份索引
            monthData.return, // 值：收益率
            monthData.date, // 额外信息：日期
          ]);
        }
      }
    });

    // 计算颜色范围
    const returns = heatmapData.map(item => item[2]);
    const maxReturn = Math.max(...returns);
    const minReturn = Math.min(...returns);
    const maxAbs = Math.max(Math.abs(maxReturn), Math.abs(minReturn));

    const option = {
      title: {
        text: selectedYear === 'all' ? '月度收益热力图' : `${selectedYear}年月度收益热力图`,
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'bold',
        },
      },
      tooltip: {
        position: 'top',
        formatter: function (params: any) {
          const [monthIndex, yearIndex, returnValue, date] = params.data;
          const month = MONTH_NAMES[monthIndex];
          const year = years[yearIndex];

          return `
            <div style="margin-bottom: 4px;">${year}年${month}</div>
            <div style="color: ${returnValue >= 0 ? '#10b981' : '#ef4444'};">
              收益率: ${returnValue.toFixed(2)}%
            </div>
          `;
        },
      },
      grid: {
        height: '60%',
        top: '15%',
        left: '10%',
        right: '10%',
      },
      xAxis: {
        type: 'category',
        data: MONTH_NAMES,
        splitArea: {
          show: true,
        },
      },
      yAxis: {
        type: 'category',
        data: years.map(year => `${year}年`),
        splitArea: {
          show: true,
        },
      },
      visualMap: {
        min: -maxAbs,
        max: maxAbs,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '5%',
        inRange: {
          color: ['#ef4444', '#ffffff', '#10b981'], // 红-白-绿
        },
        text: ['高收益', '低收益'],
        textStyle: {
          color: '#333',
        },
      },
      series: [
        {
          name: '月度收益率',
          type: 'heatmap',
          data: heatmapData,
          label: {
            show: true,
            formatter: function (params: any) {
              return `${params.data[2].toFixed(1)}%`;
            },
            fontSize: 10,
          },
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowColor: 'rgba(0, 0, 0, 0.5)',
            },
          },
        },
      ],
    };

    chartInstance.current.setOption(option);

    // 响应式调整
    const handleResize = () => {
      if (chartInstance.current) {
        chartInstance.current.resize();
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartInstance.current) {
        chartInstance.current.dispose();
      }
    };
  }, [data, selectedYear, loading]);

  // 导出图表
  const handleExportChart = () => {
    if (chartInstance.current) {
      const url = chartInstance.current.getDataURL({
        type: 'png',
        pixelRatio: 2,
        backgroundColor: '#fff',
      });

      const link = document.createElement('a');
      link.download = `monthly-heatmap-${taskId}-${selectedYear}.png`;
      link.href = url;
      link.click();
    }
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height }}>
            <CircularProgress size={32} />
          </Box>
        </CardContent>
      </Card>
    );
  }

  const monthlyStats = getMonthlyStatistics();
  const bestMonth = monthlyStats.reduce((best, current) =>
    current.avgReturn > best.avgReturn ? current : best
  );
  const worstMonth = monthlyStats.reduce((worst, current) =>
    current.avgReturn < worst.avgReturn ? current : worst
  );

  return (
    <Card>
      <CardHeader
        title={
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                width: '100%',
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Calendar size={20} color="#1976d2" />
                <Typography variant="h6" component="h3" sx={{ fontWeight: 600 }}>
                  月度收益热力图
                </Typography>
                <Tooltip title="显示每月收益率的分布，帮助识别季节性模式">
                  <IconButton size="small">
                    <Info size={16} />
                  </IconButton>
                </Tooltip>
              </Box>

              <Button
                size="small"
                variant="outlined"
                startIcon={<Download size={16} />}
                onClick={handleExportChart}
              >
                导出图表
              </Button>
            </Box>

            {/* 年份选择和统计开关 */}
            <Box
              sx={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                width: '100%',
              }}
            >
              <FormControl size="small" sx={{ minWidth: 128 }}>
                <InputLabel>选择年份</InputLabel>
                <Select
                  value={selectedYear}
                  label="选择年份"
                  onChange={e => setSelectedYear(e.target.value)}
                >
                  <MenuItem value="all">全部年份</MenuItem>
                  {data.years.map(year => (
                    <MenuItem key={year} value={year.toString()}>
                      {year}年
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormControlLabel
                control={
                  <Checkbox
                    checked={showStatistics}
                    onChange={e => setShowStatistics(e.target.checked)}
                    size="small"
                  />
                }
                label="显示月度统计"
              />
            </Box>

            {/* 月度统计信息 */}
            {showStatistics && (
              <Box
                sx={{
                  display: 'grid',
                  gridTemplateColumns: { xs: '1fr', md: 'repeat(2, 1fr)' },
                  gap: 2,
                }}
              >
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    p: 1.5,
                    bgcolor: 'success.light',
                    borderRadius: 1,
                  }}
                >
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      表现最佳月份
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="h6" sx={{ fontWeight: 600, color: 'success.main' }}>
                        {MONTH_NAMES[bestMonth.month - 1]}
                      </Typography>
                      <Chip
                        label={`+${bestMonth.avgReturn.toFixed(2)}%`}
                        color="success"
                        size="small"
                      />
                    </Box>
                  </Box>
                  <TrendingUp size={24} color="#2e7d32" />
                </Box>

                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    p: 1.5,
                    bgcolor: 'error.light',
                    borderRadius: 1,
                  }}
                >
                  <Box>
                    <Typography variant="caption" color="text.secondary">
                      表现最差月份
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="h6" sx={{ fontWeight: 600, color: 'error.main' }}>
                        {MONTH_NAMES[worstMonth.month - 1]}
                      </Typography>
                      <Chip
                        label={`${worstMonth.avgReturn.toFixed(2)}%`}
                        color="error"
                        size="small"
                      />
                    </Box>
                  </Box>
                  <TrendingDown size={24} color="#d32f2f" />
                </Box>
              </Box>
            )}
          </Box>
        }
      />

      <CardContent>
        <Box ref={chartRef} sx={{ height, width: '100%', minHeight: 400 }} />

        {/* 月度详细统计表格 */}
        {showStatistics && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" component="h4" sx={{ fontWeight: 600, mb: 2 }}>
              月度统计详情
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>月份</TableCell>
                    <TableCell align="right">平均收益率</TableCell>
                    <TableCell align="right">数据点数</TableCell>
                    <TableCell align="right">正收益次数</TableCell>
                    <TableCell align="right">负收益次数</TableCell>
                    <TableCell align="right">最高收益率</TableCell>
                    <TableCell align="right">最低收益率</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {monthlyStats.map(stat => (
                    <TableRow key={stat.month} hover>
                      <TableCell sx={{ fontWeight: 500 }}>{MONTH_NAMES[stat.month - 1]}</TableCell>
                      <TableCell
                        align="right"
                        sx={{
                          fontWeight: 500,
                          color: stat.avgReturn >= 0 ? 'success.main' : 'error.main',
                        }}
                      >
                        {stat.avgReturn.toFixed(2)}%
                      </TableCell>
                      <TableCell align="right">{stat.count}</TableCell>
                      <TableCell align="right" sx={{ color: 'success.main' }}>
                        {stat.positiveCount}
                      </TableCell>
                      <TableCell align="right" sx={{ color: 'error.main' }}>
                        {stat.negativeCount}
                      </TableCell>
                      <TableCell align="right" sx={{ color: 'success.main' }}>
                        {stat.maxReturn.toFixed(2)}%
                      </TableCell>
                      <TableCell align="right" sx={{ color: 'error.main' }}>
                        {stat.minReturn.toFixed(2)}%
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}
